from concurrent.futures import ProcessPoolExecutor
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pydicom import dcmread
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
from torchvision import transforms
from load_dicom_vol import load_volume
from loading.get_data_from_XML import *
from typing import Union
from loading.getUID import *
from ct_dataset import CTDataSet2D, CTDataSet
from pathlib import Path
from ct_dataset import NormalizationMethods
from typing import cast
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

import sys
def load_3d_files_for_subject(subject_id: str) -> list[tuple[str, str, str, list[Union[int, float]]]]:
    print(f"Loading subject {subject_id}")
    folder = os.path.join(dataset_path, "labels", shortId(subject_id))
    if not os.path.exists(folder):
        return []
    
    annotations = XML_preprocessor(folder, num_classes=num_classes).data
    
    
    available_dicom_files = getUID_path(os.path.join(dataset_path, "data", subject_id))

    label = None
    min = -sys.maxsize+1
    max = sys.maxsize
    xmin, ymin, zmin, xmax, ymax, zmax = float(max), float(max), max, float(min), float(min), min 
    for k, v in annotations.items():
        dcm_path, _ = available_dicom_files.get(k[:-4], (None, None))
        if dcm_path is None:
            continue
        

        current_label = LungPetCtDxDataset_TumorClass3D.all_tumor_class_names[np.argmax(v[0][-LungPetCtDxDataset_TumorClass3D.num_classes:])]
        if current_label != None:
            if label != None and current_label != label:
                raise Exception(f"Found different labels in same scan ({dcm_path}): {label} and {current_label}")
            label = current_label

        z = int(Path(dcm_path).stem.split("-")[1])
        _xmin, _ymin, _xmax, _ymax  = cast(tuple[float, float, float, float],v[0][:4] )
        if _xmin < xmin:
            xmin = _xmin
        if _ymin < ymin:
            ymin = _ymin
        if z < zmin:
            zmin = z

        if _xmax > xmax:
            xmax = _xmax
        if _ymax < ymax:
            ymax = _ymax
        if z > zmax:
            zmax = z     

    if label == None:
        print(f"Scan for subject {subject_id} has no label. Skipping")
        return []
        #raise Exception(f"Scan for subject {subject_id} has no label.")   
    bbox = [xmin, ymin, zmin, xmax, ymax, zmax]
    return  [(folder, label, subject_id, bbox)]

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

    annotation_reverse_lookup = {k[:-4]: k for k,_ in annotations.items()}
    paths_label_subject_mask = []
    for dcm_id, (dcm_path, dcm_name) in available_dicom_files.items():
        isTumor =  dcm_id in annotation_reverse_lookup.keys()
        annotation_id = annotation_reverse_lookup[dcm_id] if isTumor else None
        label = "Tumor" if isTumor else "No Tumor"
        if dcm_path is None:
            continue
        bounding_box = annotations[annotation_id][0][:4] if isTumor else None
        paths_label_subject_mask.append((dcm_path, label, subject_id, bounding_box))

    return paths_label_subject_mask

def isNotEmptyMask(path_label_subject_mask):
    mask = path_label_subject_mask[3]
    xmin, ymin, xmax, ymax = mask
    w, h = int(xmax - xmin), int(ymax - ymin)
    return w != 0 and h != 0


class LungPetCtDxDataset_TumorClass(CTDataSet2D):
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
        normalize: Union[NormalizationMethods, None] = None,
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
    

class LungPetCtDxDataset_TumorPresence(CTDataSet2D):
    """Lung-PET-CT-Dx dataset."""

    color_channels = 3

    def __init__(
        self,
        dataset_path: str = dataset_path,
        post_normalize_transform=None,
        cache=True,
        subject_count=None,
        normalize: Union[NormalizationMethods, None] = None,
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

        #  Not parallel (for debugging):
        # paths_label_subject_mask = list(map(load_binary_files_for_subject, self.all_subjects))

        # Parallel:
        with ProcessPoolExecutor(max_workers=8) as executor:
            for r in executor.map(load_binary_files_for_subject, self.all_subjects):
                paths_label_subject_mask += r
        return paths_label_subject_mask


class LungPetCtDxDataset_TumorClass3D(CTDataSet):
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
        cache=True,
        subject_count=None,
        exclude_classes: Union[list[str], None] = None,
        max_size: int = -1,
        exclude_empty_bbox_samples=False,
        samples_per_scan=4,
        slices_per_sample=30,
    ):
        super().__init__(
            LungPetCtDxDataset_TumorClass3D.all_tumor_class_names,
            dataset_path,
            cache,
            subject_count,
            exclude_classes,
            max_size,
            exclude_empty_bbox_samples,
        )

        self.samples_per_scan = samples_per_scan
        self.slices_per_sample = slices_per_sample
    

    def _get_paths_labels_subjects_mask(self):
        paths_label_subject_mask = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            for r in executor.map(load_3d_files_for_subject, self.all_subjects):
                paths_label_subject_mask += r
        return paths_label_subject_mask

    def __len__(self):
        return super().__len__() * self.samples_per_scan

    def __getitem__(self, idx):
        item_idx, sample_idx = divmod(idx, len(self.paths_label_subject_mask))

        path, label, subject, mask = self.paths_label_subject_mask[item_idx]
        x_min, y_min, z_min, x_max, y_max, z_max = mask
        offset_per_slice = (self.slices_per_sample - (z_max - z_min)) / self.samples_per_scan

        z_start = max(0, z_max - self.slices_per_sample + offset_per_slice*sample_idx)
        # TOOD: Chheck if z_end valid
        z_end = z_start + self.slices_per_sample

        volume = load_volume(path)
        if z_end > volume.shape[-1]:
            raise Exception('z_end out of bounds of volume.')
        
        volume = torch.tensor(volume[...,z_start:z_end])
        label_one_hot = torch.tensor(self.to_one_hot(label), dtype=torch.float32)
        return volume, label_one_hot