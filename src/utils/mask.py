import numpy as np
import torch

def mask_image(img, bounding_box):
    mask = np.zeros(img.shape, dtype=np.float32)
    x_min, y_min, x_max, y_max = np.rint(bounding_box*img.shape[-1]).to(torch.int32)
    mask[:, x_min:x_max, y_min:y_max] = img[:, x_min:x_max, y_min:y_max]
    return mask