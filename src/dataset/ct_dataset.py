from concurrent.futures import ProcessPoolExecutor
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pydicom import dcmread
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
from torchvision import transforms
from loading.get_data_from_XML import *
from typing import Union, Literal, Optional, Any
from loading.getUID import *
from abc import abstractmethod
from pathlib import Path
from enum import Enum
from torchvision.transforms.functional import crop
from dataset.base_dataset import BaseDataset
from torch.utils.data import random_split


def isNotEmptyMask(path_label_subject_mask):
    mask = path_label_subject_mask[3]
    xmin, ymin, xmax, ymax = mask
    w, h = int(xmax - xmin), int(ymax - ymin)
    return w != 0 and h != 0


class NormalizationMethods(Enum):
    SINGLE_IMAGE = "NORMALIZE_SINGLE_IMAGE"
    WHOLE_DATASET = "NORMALIZE_WHOLE_DATASET"


def calculate_meanstd(t: np.ndarray, axis=None):
    mean = np.mean(t, axis=axis, keepdims=True)
    std = np.sqrt(((t - mean) ** 2).mean(axis=axis, keepdims=True))
    return mean, std


def normalize_meanstd(t: np.ndarray, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean, std = calculate_meanstd(t, axis)
    return (t - mean) / std


PathLabelSubjectMaskList = list[tuple[str, str, str, tuple[float, float, float, float]]]


class CTDataSet(BaseDataset, Dataset):
    """Lung-PET-CT-Dx dataset."""

    color_channels = 3

    def __init__(
        self,
        all_classes: list[str],
        dataset_path: str,
        cache=True,
        subject_count=None,
        exclude_classes: Union[list[str], None] = None,
        max_size: int = -1,
        exclude_empty_bbox_samples=False,
        sampling: Optional[Literal["undersample"]] = None,
        item_filter=None,
    ):
        self.dataset_path = dataset_path
        self.cache_file = Path(f"../cache/{type(self).__name__}_metadata.pickle")

        self.exclude_classes = exclude_classes
        self.class_names = list(
            filter(lambda item: not self.isExcluded(item), all_classes)
        )
        csv_path = os.path.join(dataset_path, "metadata.csv")
        csv_file = pd.read_csv(csv_path)

        self.all_subjects = csv_file["Subject ID"].unique()
        self.filtered_subjects: list[str] = None  # type: ignore
        if subject_count:
            print(f"Only using {subject_count} subjects")
            self.filtered_subjects: list[str] = self.all_subjects[
                :subject_count
            ].tolist()

        self.paths_label_subject_mask: PathLabelSubjectMaskList = []
        self.load_paths_labels_subjects_mask(cache)

        if exclude_empty_bbox_samples:
            self.paths_label_subject_mask = list(
                filter(isNotEmptyMask, self.paths_label_subject_mask)
            )

        if max_size != -1:
            self.paths_label_subject_mask = self.paths_label_subject_mask[:max_size]

        if item_filter is not None:
            self.paths_label_subject_mask = list(
                filter(item_filter, self.paths_label_subject_mask)
            )

        if sampling == "undersample":
            label_map = list(map(lambda t: t[1], self.paths_label_subject_mask))
            from imblearn.under_sampling import RandomUnderSampler

            result = RandomUnderSampler().fit_resample(
                self.paths_label_subject_mask, label_map
            )
            new_path_label_sub: Any = result[0]
            new_path_label_sub_typed: PathLabelSubjectMaskList = new_path_label_sub
            print(
                f"Undersampled from {len(self.paths_label_subject_mask)} items to {len(new_path_label_sub)}"
            )
            self.paths_label_subject_mask = new_path_label_sub_typed

    def isExcluded(self, label: str):
        return label in self.exclude_classes if self.exclude_classes != None else False

    def split(self, test_size: float, mode: Literal["subject", "random"] = "subject"):
        if mode == "subject":
            return self.subject_split(test_size)
        else:
            return random_split(self, [0.8, 0.2])

    def subject_split(self, test_size: float):
        subj_train, subj_test = train_test_split(
            self.all_subjects
            if self.filtered_subjects is None
            else self.filtered_subjects,
            test_size=test_size,
            random_state=42,
        )
        idx_train = []
        idx_test = []
        for idx, t in enumerate(self.paths_label_subject_mask):
            if t[2] in subj_test:
                idx_test.append(idx)
            else:
                idx_train.append(idx)
        train_dataset_split = Subset(self, idx_train)
        test_dataset_split = Subset(self, idx_test)

        return train_dataset_split, test_dataset_split

    # Returns weights between 0 and 1 inverse to the amount they take up in the dataset
    def get_class_weights_inverse(self):
        _, labels, *_ = list(zip(*self.paths_label_subject_mask))
        labels = list(map(lambda l: self.to_one_hot(l), labels))
        weights = np.sum(labels, axis=0, dtype="float32")
        weights = weights / weights.sum()
        # TODO: check this
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return torch.tensor(weights)

    def load_paths_labels_subjects_mask(self, cache):
        if cache and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.paths_label_subject_mask = pickle.load(f)
        else:
            self.paths_label_subject_mask = self._get_paths_labels_subjects_mask()

            if cache:
                os.makedirs(self.cache_file.parent, exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(
                        self.paths_label_subject_mask, f, pickle.HIGHEST_PROTOCOL
                    )
        if self.filtered_subjects != None:
            self.paths_label_subject_mask = list(
                filter(
                    lambda item: item[2] in self.filtered_subjects,
                    self.paths_label_subject_mask,
                )
            )

        if not self.exclude_classes == None:
            self.paths_label_subject_mask = list(
                filter(
                    lambda item: not self.isExcluded(item[1]),
                    self.paths_label_subject_mask,
                )
            )

    @abstractmethod
    def _get_paths_labels_subjects_mask(
        self,
    ) -> list[tuple[str, str, str, tuple[float, float, float, float]]]:
        pass

    def __len__(self):
        return len(self.paths_label_subject_mask)

    def to_one_hot(self, label):
        return np.eye(len(self.class_names))[self.class_names.index(label)]

    @abstractmethod
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        pass


class CTDataSet2D(CTDataSet):
    """Lung-PET-CT-Dx dataset."""

    color_channels = 3

    def __init__(
        self,
        all_classes: list[str],
        dataset_path: str,
        post_normalize_transform=None,
        cache=True,
        subject_count=None,
        exclude_classes: Union[list[str], None] = None,
        normalize: Union[NormalizationMethods, None] = None,
        max_size: int = -1,
        crop_to_tumor: bool = False,
        image_resolution=128,
        exclude_empty_bbox_samples=False,
        sampling=None,
    ):
        super().__init__(
            all_classes,
            dataset_path,
            cache,
            subject_count,
            exclude_classes,
            max_size,
            exclude_empty_bbox_samples,
            sampling,
        )

        self.data_distribution_cache_file = Path(
            f"../cache/{type(self).__name__}_dataset_distribution.pickle"
            if not crop_to_tumor
            else f"../cache/{type(self).__name__}_cropped_dataset_distribution.pickle"
        )

        self.post_normalize_transform = post_normalize_transform
        self._disable_transform_and_norm = False
        self._force_normalize_for_dist_calc = False
        self._normalization_transform = None
        self.crop_to_tumor = crop_to_tumor
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(image_resolution),
                transforms.CenterCrop(image_resolution),
            ]
        )

        self.normalize = normalize

        if normalize == NormalizationMethods.WHOLE_DATASET:
            mean, sd = None, None
            if cache and os.path.exists(self.data_distribution_cache_file):
                with open(self.data_distribution_cache_file, "rb") as f:
                    mean, sd = pickle.load(f)
            else:
                mean, sd = self.calculateMeanAndSD()
                if cache:
                    with open(self.data_distribution_cache_file, "wb") as f:
                        pickle.dump((mean, sd), f, pickle.HIGHEST_PROTOCOL)
            self._normalization_transform = transforms.Normalize(mean=mean, std=sd)

    @abstractmethod
    def _get_paths_labels_subjects_mask(
        self,
    ) -> list[tuple[str, str, str, tuple[float, float, float, float]]]:
        pass

    def __getitem__(self, idx):

        path, label, subject, mask = self.paths_label_subject_mask[idx]

        # Read from path
        img = dcmread(path).pixel_array
        # Add channel dimension if greyscale by repeating gray channel
        if len(img.shape) == 2:
            img = np.array(img, dtype=np.float32)[np.newaxis]
            img = np.repeat(img, self.color_channels, 0)
        elif len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        else:
            raise Exception(f"Unknown shape of dicom: {img.shape}")

        if self.normalize == NormalizationMethods.SINGLE_IMAGE:
            img = normalize_meanstd(img.astype(np.float64), axis=(1, 2))
        # Convert to tensor with now proper Channel x Height x Width dimensions
        img = torch.from_numpy(img.astype(np.float32))

        # Normalize tumor bounding box before transformation
        _channels, sx, sy = img.shape

        if self.crop_to_tumor:
            xmin, ymin, xmax, ymax = mask
            w, h = int(xmax - xmin), int(ymax - ymin)
            size = max(w, h)
            x = xmin + (xmax - xmin) / 2 - size / 2
            y = ymin + (ymax - ymin) / 2 - size / 2

            if w == 0 and h == 0:
                print("width and height of masked image are 0!")
            else:
                img = crop(img, int(y), int(x), int(size), int(size))

            mask = np.array([0, 0, 1, 1])
        else:

            mask = (
                np.array([mask[0] / sx, mask[1] / sy, mask[2] / sx, mask[3] / sy])
                if mask is not None
                else np.array([0, 0, 1, 1], dtype=np.float32)
            )

        img = self.resize_transform(img)
        if self._normalization_transform:
            img = self._normalization_transform(img)

        if self.post_normalize_transform and (
            not self._disable_transform_and_norm or self._force_normalize_for_dist_calc
        ):
            img = self.post_normalize_transform(img)

        img = img

        label_one_hot = self.to_one_hot(label)
        return (img, label_one_hot, mask)

    def calculateMeanAndSD(self, batch_size=64, normalize=False):
        self._disable_transform_and_norm = True
        if normalize:
            self._force_normalize_for_dist_calc = True

        loader = DataLoader(self, batch_size=batch_size, num_workers=8, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mean = torch.zeros(self.color_channels).to(device)
        var = torch.zeros(self.color_channels).to(device)

        numBatches = len(loader)
        for i, (images, _, _) in enumerate(loader):
            images = images.to(device)
            batch_mean = images.mean((2, 3)).transpose(0, 1).mean(1)
            mean += batch_mean
            width = images.size(3)
            height = images.size(2)
            batch_size = images.size(0)
            color_channels = images.size(1)

            batch_var = (
                images.transpose(1, 3)
                .sub(batch_mean)
                .reshape((batch_size, width * height, color_channels))
                .square()
                .transpose(1, 2)
                .transpose(0, 1)
                .reshape((color_channels, batch_size * width * height))
                .mean(1)
            )
            var += batch_var
            print(
                f"Calculating mean and sd of ds: {i}/{numBatches} batches done. ",
                end="\r",
            )

        print(f"Done with {numBatches}")
        mean, sd = mean.cpu().div(numBatches), var.cpu().div(numBatches).sqrt()

        print(f"Calculated mean: {mean}, sd: {sd}")

        self._disable_transform_and_norm = False
        self._force_normalize_for_dist_calc = False
        return mean, sd
