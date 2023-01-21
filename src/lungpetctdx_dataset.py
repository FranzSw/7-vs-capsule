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
from typing import Union
from loading.getUID import *
from ct_dataset import CTDataSet
from pathlib import Path

dataset_path = "/dhc/dsets/Lung-PET-CT-Dx"
num_classes = 4



def shortId(ext_str: str):
    return ext_str.replace("Lung_Dx-", "")


def get_xml_files(subject_ids):
    ids_folders = list(
        map(
            lambda s: (s, os.path.join(dataset_path, "labels", shortId(s))), subject_ids
        )
    )
    return list(filter(lambda tuple: os.path.exists(tuple[1]), ids_folders))


def load_files_for_subject(subject_id: str) -> list[tuple[str, str, str, list[int]]]:
    print(f"Loading subject {subject_id}")
    folder = os.path.join(dataset_path, "labels", shortId(subject_id))
    if not os.path.exists(folder):
        return []
    annotations = XML_preprocessor(folder, num_classes=num_classes).data

    available_dicom_files = getUID_path(os.path.join(dataset_path, "data", subject_id))

    paths_label_subject_mask = []
    for k, v in annotations.items():
        dcm_path, dcm_name = available_dicom_files.get(k[:-4], (None, None))
        if dcm_path is None:
            continue
        label = LungPetCtDxDataset_TumorClass.all_tumor_class_names[np.argmax(v[0][-LungPetCtDxDataset_TumorClass.num_classes:])]
        bounding_box = v[0][:4]
        paths_label_subject_mask.append((dcm_path, label, subject_id, bounding_box))

    return paths_label_subject_mask


def load_binary_files_for_subject(subject_id: str) -> list[tuple[str, str, str, list[int]]]:
    print(f"Loading subject {subject_id}")
    folder = os.path.join(dataset_path, "labels", shortId(subject_id))
    if not os.path.exists(folder):
        return []
    annotations = XML_preprocessor(folder, num_classes=num_classes).data

    available_dicom_files = getUID_path(os.path.join(dataset_path, "data", subject_id))

    paths_label_subject_mask = []
    for k, v in available_dicom_files.items():
        isTumor =  k in annotations
        label = "Tumor" if isTumor else "No Tumor"


        dcm_path, dcm_name = v
        if dcm_path is None:
            continue
        bounding_box = annotations[k][0][:4] if isTumor else None
        paths_label_subject_mask.append((dcm_path, label, subject_id, bounding_box))

    return paths_label_subject_mask

def isNotEmptyMask(path_label_subject_mask):
    mask = path_label_subject_mask[3]
    xmin, ymin, xmax, ymax = mask
    w, h = int(xmax - xmin), int(ymax - ymin)
    return w != 0 and h != 0


class LungPetCtDxDataset_TumorClass(CTDataSet):
    """Lung-PET-CT-Dx dataset."""
    all_tumor_class_names = [
        "Adenocarcinoma",
        "Small Cell Carcinoma",
        "Large Cell Carcinoma",
        "Squamous Cell Carcinoma",
    ]
    num_classes = 4
    color_channels = 3

    def __init__(
        self,
        dataset_path: str = dataset_path,
        post_normalize_transform=None,
        cache=True,
        subject_count=None,
        exclude_classes: Union[list[str], None] = None,
        normalize=False,
        max_size: int = -1,
        crop_to_tumor: bool = False,
        cropped_tumor_size=128,
        exclude_empty_bbox_samples=False,
    ):
        super().__init__(
            LungPetCtDxDataset_TumorClass.all_tumor_class_names,
            dataset_path,
            post_normalize_transform,
            cache,
            subject_count,
            exclude_classes,
            normalize,
            max_size,
            crop_to_tumor,
            cropped_tumor_size,
            exclude_empty_bbox_samples,
        )
    

    def _get_paths_labels_subjects_mask(self):
        paths_label_subject_mask = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            for r in executor.map(load_files_for_subject, self.all_subjects):
                paths_label_subject_mask += r
        return paths_label_subject_mask
    

class LungPetCtDxDataset_TumorPresence(CTDataSet):
    """Lung-PET-CT-Dx dataset."""

    color_channels = 3

    def __init__(
        self,
        dataset_path: str = dataset_path,
        post_normalize_transform=None,
        cache=True,
        subject_count=None,
        normalize=False,
        max_size: int = -1
    ):
        super().__init__(
            ["Tumor", "No Tumor"],
            dataset_path,
            post_normalize_transform,
            cache,
            subject_count,
            normalize = normalize,
            max_size = max_size,
            crop_to_tumor= False,
            exclude_empty_bbox_samples=False,
        )
    

    def _get_paths_labels_subjects_mask(self):
        paths_label_subject_mask = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            for r in executor.map(load_binary_files_for_subject, self.all_subjects):
                paths_label_subject_mask += r
        return paths_label_subject_mask