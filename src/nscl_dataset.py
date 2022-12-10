from concurrent.futures import ProcessPoolExecutor
import torch
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread
from pydicom.data import get_testdata_file
import pandas as pd
import os
import pickle

from loading.get_data_from_XML import *
from loading.getUID import *

dataset_path = "/Volumes/X/datasets/Lung-PET-CT-Dx"
num_classes = 4
cache_file = 'ds_cache.pickle'

def shortId(ext_str: str):
    return ext_str.replace('Lung_Dx-', '')

def get_xml_files(subject_ids):
    ids_folders = list(map(lambda s: (s, os.path.join(dataset_path, 'labels', shortId(s))), subject_ids))
    return list(filter(lambda tuple: os.path.exists(tuple[1]), ids_folders))


def load_files_for_subject(subject_id: str) -> dict[str, list[int], str]:
    print(f'Loading subject {subject_id}')
    folder = os.path.join(dataset_path, 'labels', shortId(subject_id))
    if not os.path.exists(folder):
        return []
    annotations = XML_preprocessor(folder, num_classes=num_classes).data

    available_dicom_files = getUID_path(os.path.join(dataset_path, 'data', subject_id))

    paths_label_subject = []
    for k, v in annotations.items():
        dcm_path, dcm_name = available_dicom_files.get(k[:-4], (None, None))
        if dcm_path is None:
            continue
        label = v[0][-num_classes:]
        paths_label_subject.append((dcm_path, label, subject_id))

    return paths_label_subject

class NSCLDataSet(Dataset):
    """Lung-PET-CT-Dx dataset."""

    def __init__(self, dataset_path: str = dataset_path, transform=None, cache=True):
        # dirs = [d for d in os.listdir(datasetPath) if os.isdir(d)]

        csv_path = os.path.join(dataset_path, 'metadata.csv')
        csv_file = pd.read_csv(csv_path)

        self.subjects = csv_file['Subject ID'].unique()
        self.paths_label_subject = []

        self.load_metadata(cache)

        self.transform = transform

    def load_metadata(self, cache):
        if cache and os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.paths_label_subject = pickle.load(f)
        else:
            with ProcessPoolExecutor(max_workers=8) as executor:
                for r in executor.map(load_files_for_subject, self.subjects):
                    self.paths_label_subject.append(r)
                    
            if cache:
                with open(cache_file, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.paths_label_subject, f, pickle.HIGHEST_PROTOCOL)


    def __len__(self):
        return len(self.paths_label_subject)

    def __getitem__(self, idx):
        
        path, label, subject = self.paths_label_subject[idx]
        
        # Read from path
        img = dcmread(path).pixel_array

        if self.transform:
            img = self.transform(img)

        return (img, label)