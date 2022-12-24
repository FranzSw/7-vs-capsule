from concurrent.futures import ProcessPoolExecutor
import torch
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread
from pydicom.data import get_testdata_file
import pandas as pd
import os
import pickle
from torchvision import transforms
from loading.get_data_from_XML import *
from typing import Union
from loading.getUID import *

from pathlib import Path
dataset_path = "/dhc/dsets/Lung-PET-CT-Dx"
num_classes = 4
all_class_names = ["Adenocarcinoma", "Small Cell Carcinoma",
                   "Large Cell Carcinoma", "Squamous Cell Carcinoma"]


def shortId(ext_str: str):
    return ext_str.replace('Lung_Dx-', '')


def get_xml_files(subject_ids):
    ids_folders = list(map(lambda s: (s, os.path.join(
        dataset_path, 'labels', shortId(s))), subject_ids))
    return list(filter(lambda tuple: os.path.exists(tuple[1]), ids_folders))


def load_files_for_subject(subject_id: str) -> list[tuple[str, str, str]]:
    print(f'Loading subject {subject_id}')
    folder = os.path.join(dataset_path, 'labels', shortId(subject_id))
    if not os.path.exists(folder):
        return []
    annotations = XML_preprocessor(folder, num_classes=num_classes).data

    available_dicom_files = getUID_path(
        os.path.join(dataset_path, 'data', subject_id))

    paths_label_subject = []
    for k, v in annotations.items():
        dcm_path, dcm_name = available_dicom_files.get(k[:-4], (None, None))
        if dcm_path is None:
            continue
        label = all_class_names[np.argmax(v[0][-num_classes:])]
        paths_label_subject.append((dcm_path, label, subject_id))

    return paths_label_subject


class LungPetCtDxDataset(Dataset):
    """Lung-PET-CT-Dx dataset."""
    color_channels = 3

    def __init__(self, dataset_path: str = dataset_path, post_normalize_transform=None, cache=True, subject_count=None, exclude_classes: Union[list[str], None] = None, normalize=False):
        # dirs = [d for d in os.listdir(datasetPath) if os.isdir(d)]
        self.cache_file = Path(
            f'../cache/{type(self).__name__}_metadata.pickle')
        self.data_distribution_cache_file = Path(
            f'../cache/{type(self).__name__}_dataset_distribution.pickle')
        self.exclude_classes = exclude_classes
        self.class_names = list(
            filter(lambda item: not self.isExcluded(item), all_class_names))
        csv_path = os.path.join(dataset_path, 'metadata.csv')
        csv_file = pd.read_csv(csv_path)

        self.all_subjects = csv_file['Subject ID'].unique()
        if subject_count:
            print(f'Only using {subject_count} subjects')
            self.filtered_subjects = self.all_subjects[:subject_count]

        self.paths_label_subject = []

        self.post_normalize_transform = post_normalize_transform
        self._disable_transform_and_norm = False
        self._force_normalize_for_dist_calc = False
        self._normalization_transform = None
        self.load_metadata(cache, normalize)

    def isExcluded(self, label: str):
        return label in self.exclude_classes if self.exclude_classes != None else False

    def load_metadata(self, cache, normalize):
        if cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.paths_label_subject = pickle.load(f)
        else:
            with ProcessPoolExecutor(max_workers=8) as executor:
                for r in executor.map(load_files_for_subject, self.all_subjects):
                    self.paths_label_subject += r

            if cache:
                os.makedirs(self.cache_file.parent, exist_ok=True)
                with open(self.cache_file, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.paths_label_subject,
                                f, pickle.HIGHEST_PROTOCOL)
        if not self.filtered_subjects == None:
            self.paths_label_subject = list(
                filter(lambda item: item[2] in self.filtered_subjects, self.paths_label_subject))

        if not self.exclude_classes == None:
            self.paths_label_subject = list(
                filter(lambda item: not self.isExcluded(item[1]), self.paths_label_subject))

        if normalize:
            mean, sd = None, None
            if cache and os.path.exists(self.data_distribution_cache_file):
                with open(self.data_distribution_cache_file, 'rb') as f:
                    mean, sd = pickle.load(f)
            else:
                mean, sd = self.calculateMeanAndSD()
                if cache:
                    with open(self.data_distribution_cache_file, 'wb') as f:
                        pickle.dump((mean, sd), f,
                                    pickle.HIGHEST_PROTOCOL)
            self._normalization_transform = transforms.Normalize(
                mean=mean, std=sd)

    def __len__(self):
        return len(self.paths_label_subject)

    def __getitem__(self, idx):

        path, label, subject = self.paths_label_subject[idx]

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

        # Convert to tensor with now proper Channel x Height x Width dimensions
        img = torch.from_numpy(img.astype(np.float32))

        if self._normalization_transform:
            img = self._normalization_transform(img)

        if self.post_normalize_transform and (not self._disable_transform_and_norm or self._force_normalize_for_dist_calc):
            img = self.post_normalize_transform(img)

        label_one_hot = np.eye(len(self.class_names))[
            self.class_names.index(label)]
        return (img, label_one_hot)

    def calculateMeanAndSD(self, batch_size=32, normalize=False):
        self._disable_transform_and_norm = True
        if normalize:
            self._force_normalize_for_dist_calc = True

        loader = DataLoader(self,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=False)

        mean = torch.zeros(self.color_channels)
        var = torch.zeros(self.color_channels)

        numBatches = len(loader)
        for i, (images, _) in enumerate(loader):
            batch_mean = images.mean((2, 3)).transpose(0, 1).mean(1)
            mean += batch_mean
            width = images.size(3)
            height = images.size(2)
            batch_size = images.size(0)
            color_channels = images.size(1)

            batch_var = images.transpose(1, 3).sub(batch_mean).reshape((batch_size, width*height, color_channels)).square(
            ).transpose(1, 2).transpose(0, 1).reshape((color_channels, batch_size*width*height)).mean(1)
            var += batch_var
            print(
                f'Calculating mean and sd of ds: {i}/{numBatches} batches done. ', end='\r')

        print()
        mean, sd = mean.div(numBatches), var.div(numBatches).sqrt()

        print(f'Calculated mean: {mean}, sd: {sd}')

        self._disable_transform_and_norm = False
        self._force_normalize_for_dist_calc = False
        return mean, sd
