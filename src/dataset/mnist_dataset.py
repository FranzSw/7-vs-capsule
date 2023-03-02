from dataset.base_dataset import BaseDataset
from torchvision.datasets import MNIST
import os
import numpy as np
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale
from torch.nn.functional import one_hot
import torch
class MNISTDataset(BaseDataset):
    class_names = ["0","1","2","3","4","5","6","7","8","9"]
    def __init__(self, image_width=28, color_channels = 3):
        root_dir = "../cache/mnist"
        os.makedirs(root_dir, exist_ok=True)

        # Normalization parameters taken from https://github.com/pytorch/examples/blob/main/mnist/main.py 
        transform = Compose([Resize((image_width,image_width)), Grayscale(color_channels),ToTensor(), Normalize((0.1307,0.1307,0.1307) if color_channels==3 else (0.1307,), (0.3081,0.3081,0.3081) if color_channels==3 else (0.3081,))])
        
        def to_one_hot(target_index):
            return torch.eye(10)[target_index]

        self.train = MNIST(root=root_dir, train=True, download=True, transform=transform, target_transform=to_one_hot)
        self.test = MNIST(root=root_dir, train=False, download=True, transform=transform, target_transform=to_one_hot)
    
    def get_class_weights_inverse(self):
        counts = np.zeros(10)

        for (_, label_index) in self.train:
            counts[label_index] +=1
        
        relative = counts/ np.sum(counts)
        return 1-relative
    
    def split(self, _):
        return self.train, self.test

