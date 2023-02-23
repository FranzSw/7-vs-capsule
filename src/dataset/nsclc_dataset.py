import os
from typing import Union
import torch
from sklearn.model_selection import train_test_split
from ct_dataset import CTDataSet
from load_dicom_vol import load_volume
import numpy as np
import pandas as pd


class NsclcDataset_TumorClass3D(CTDataSet):
    """Lung-PET-CT-Dx dataset."""
    all_tumor_class_names = [
        "Adenocarcinoma",
        "Squamous cell carcinoma",
    ]
    num_classes = 4
    color_channels = 3

    def __init__(
        self,
        dataset_path: str,
        cache=True,
        subject_count=None,
        exclude_classes: Union[list[str], None] = None,
        max_size: int = -1,
        exclude_empty_bbox_samples=False,
        postprocess=None,
    ):
        super().__init__(
            self.all_tumor_class_names,
            dataset_path,
            cache,
            subject_count,
            exclude_classes,
            max_size,
            exclude_empty_bbox_samples,
        )

        self.postprocess = postprocess
    

    def _get_paths_labels_subjects_mask(self):
        metadata = pd.read_csv(os.path.join(self.dataset_path, 'metadata.csv'))
        subjects_data = pd.read_csv(os.path.join(self.dataset_path, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv'))

        fusion_data = metadata[metadata['Series Description'] == 'CT FUSION']
        paths_label_subject_mask = []
        for row in fusion_data.to_dict(orient="records"):
            subject_id = row['Subject ID']
            subject_data = subjects_data[subjects_data['Case ID'] == subject_id]
            label = subject_data['Histology'].iloc[0]
            if label in self.all_tumor_class_names:
                path = row['File Location'].replace('./NSCLC Radiogenomics/', '')
                path = os.path.join(self.dataset_path, 'data', path)
                paths_label_subject_mask += [(path, label, subject_id, [0,0,0,1,1,1])]
        
        return paths_label_subject_mask
    
    def __getitem__(self, idx):
        path, label, subject, mask = self.paths_label_subject_mask[idx]

        volume = load_volume(path)

        if len(volume.shape) == 3:
            volume = np.expand_dims(np.array(volume, dtype=np.float32), 0)
            volume = np.repeat(volume, self.color_channels, 0)
        elif len(volume.shape) == 4:
            pass
        else:
            raise Exception(f"Unknown shape of dicom: {volume.shape}")

        if self.postprocess is not None:
            volume = torch.tensor(volume, dtype=torch.float32)
            volume = self.postprocess(volume)

        label_one_hot = torch.tensor(self.to_one_hot(label), dtype=torch.float32)
        return volume, label_one_hot, mask