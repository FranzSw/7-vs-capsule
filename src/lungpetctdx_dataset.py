from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import random
import torch
from torch.utils.data import Dataset, Subset
from pydicom import dcmread
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle

from loading.get_data_from_XML import *
from typing import Union
from loading.getUID import *

from pathlib import Path
dataset_path = "/dhc/dsets/Lung-PET-CT-Dx"
num_classes = 4
all_class_names =["Adenocarcinoma", "Small Cell Carcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma"]

def shortId(ext_str: str):
    return ext_str.replace('Lung_Dx-', '')

def get_xml_files(subject_ids):
    ids_folders = list(map(lambda s: (s, os.path.join(dataset_path, 'labels', shortId(s))), subject_ids))
    return list(filter(lambda tuple: os.path.exists(tuple[1]), ids_folders))


def load_files_for_subject(subject_id: str) -> list[tuple[str, str, str]]:
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
        label = all_class_names[np.argmax(v[0][-num_classes:])]
        paths_label_subject.append((dcm_path, label, subject_id))

    return paths_label_subject

class LungPetCtDxDataset(Dataset):
    """Lung-PET-CT-Dx dataset."""
    
    def __init__(self, dataset_path: str = dataset_path, transform=None, cache=True, subject_count=None, exclude_classes: Union[list[str],None]=None):
        # dirs = [d for d in os.listdir(datasetPath) if os.isdir(d)]
        self.cache_file = Path(f'../cache/{type(self).__name__}_metadata.pickle')
        self.exclude_classes = exclude_classes
        self.class_names = list(filter(lambda item: not self.isExcluded(item), all_class_names))
        csv_path = os.path.join(dataset_path, 'metadata.csv')
        csv_file = pd.read_csv(csv_path)

        self.subjects = csv_file['Subject ID'].unique()
        if subject_count:
            print(f'Only using {subject_count} subjects')
            self.subjects = self.subjects[:subject_count]

        self.paths_label_subject = []
        
        self.load_metadata(cache)

        self.transform = transform

    def subject_split(self, test_size: float):
        subj_train, subj_test = train_test_split(self.subjects, test_size=test_size, random_state=42)
        idx_train = []
        idx_test = []
        for idx, t in enumerate(self.paths_label_subject):
            if t[2] in subj_test:
                idx_test.append(idx)
            else:
                idx_train.append(idx)
        train_dataset_split = Subset(self, idx_train)
        test_dataset_split = Subset(self, idx_test)
        
        return train_dataset_split, test_dataset_split

    # Returns weights between 0 and 1 inverse to the amount they take up in the dataset
    def get_class_weights_inverse(self):
        _, labels, _ = list(zip(*self.paths_label_subject))
        labels = list(map(lambda l: self.to_one_hot(l), labels))
        weights = np.sum(labels, axis=0, dtype='float32')
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return torch.tensor(weights)

        

    def isExcluded(self, label: str):
        if self.exclude_classes == None:
            return False
        return label in self.exclude_classes

    def load_metadata(self, cache):
        if cache and os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.paths_label_subject = pickle.load(f)
        else:
            with ProcessPoolExecutor(max_workers=8) as executor:
                for r in executor.map(load_files_for_subject, self.subjects):
                    self.paths_label_subject += r
                    
            if cache:
                os.makedirs(self.cache_file.parent ,exist_ok=True)
                with open(self.cache_file, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.paths_label_subject, f, pickle.HIGHEST_PROTOCOL)
        if not self.exclude_classes == None:
            self.paths_label_subject = list(filter(lambda item: not self.isExcluded(item[1]), self.paths_label_subject))

    def __len__(self):
        return len(self.paths_label_subject)

    def to_one_hot(self, label):
        return np.eye(len(self.class_names))[self.class_names.index(label)]

    def __getitem__(self, idx):
        
        path, label, subject = self.paths_label_subject[idx]
        
        # Read from path
        img = dcmread(path).pixel_array
        # Add channel dimension if greyscale by repeating gray channel
        if len(img.shape) == 2:
            img = np.array(img, dtype=np.float32)[np.newaxis]
            img = np.repeat(img, 3, 0)
        elif len(img.shape) == 3:
            img = img.transpose((2,0,1))
        else:
            raise Exception(f"Unknown shape of dicom: {img.shape}")
        
        # Convert to tensor with now proper Channel x Height x Width dimensions
        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)
        label_one_hot = self.to_one_hot(label)
        return (img, label_one_hot)