import torch
import numpy as np
from torchvision import transforms


class Resize3D(torch.nn.Module):
    def __init__(self, size=64):
        super().__init__()
        self.size = size
        self.resize = transforms.Resize(size)

    def forward(self, img):
        new_shape = list(img.shape)
        new_shape[-2] = self.size
        new_shape[-3] = self.size
        out = torch.tensor(np.zeros(shape=new_shape, dtype=np.float32))
        for i in range(img.shape[-1]):
            tmp = self.resize(img[..., i])
            out[..., i] = tmp
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class Grayscale3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.PILToTensor(),
            ]
        )

    def forward(self, img):
        new_shape = list(img.shape)
        new_shape[0] = 1
        out = torch.tensor(np.zeros(shape=new_shape, dtype=np.float32))
        for i in range(img.shape[-1]):
            tmp = self.transform(img[..., i])
            out[..., i] = tmp
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
