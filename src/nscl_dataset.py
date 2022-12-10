import torch
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread
from pydicom.data import get_testdata_file
import pandas as pd
import os

from loading.get_data_from_XML import *
from loading.getUID import *

dataset_path = "/dhc/dsets/Lung-PET-CT-Dx/Lung-PET-CT-Dx"
num_classes = 4

def shortId(ext_str: str):
    return ext_str.replace('Lung_Dx-', '')

def get_xml_files(subject_ids):
    ids_folders = list(map(lambda s: (s, os.path.join(dataset_path, 'labels', shortId(s))), subject_ids))
    return list(filter(lambda tuple: os.path.exists(tuple[1]), ids_folders))


class NSCLDataSet(Dataset):
    """Face Landmarks dataset."""

    def load_files_for_subject(subject_id: str) -> dict[str, list[int], str]:
        folder = os.path.join(dataset_path, 'labels', shortId(subject_id))
        if not os.path.exists(folder):
            return []
        annotations = XML_preprocessor(folder, num_classes=num_classes).data

        available_dicom_files = getUID_path(os.path.join(dataset_path, 'data', subject_id))

        paths_label_subject = []
        for k, v in annotations.items():
            dcm_path, dcm_name = available_dicom_files[k[:-4]]
            label = v[-num_classes]
            paths_label_subject.append((dcm_path, label, subject_id))

        return paths_label_subject

    def __init__(self, dataset_path: str = dataset_path, transform=None):
        # dirs = [d for d in os.listdir(datasetPath) if os.isdir(d)]

        csv_path = os.path.join(dataset_path, 'metadata.csv')
        csv_file = pd.read_csv(csv_path)

        self.subjects = csv_file['Subject ID'].unique()
        self.paths_label_subject = []
        for subject in self.subjects:
            subject_data = self.load_files_for_subject(subject)
            self.paths_label_subject.append(subject_data)

        self.transform = transform


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        
        path, label, subject = self.paths_label_subject[idx]
        
        # Read from path
        img = dcmread(path).pixel_array

        if self.transform:
            img = self.transform(img)

        return (img, label)