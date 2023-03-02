import numpy as np
import torch


def mask_image(img, bounding_box):
    mask = np.full(img.shape, img.min().item(), dtype=np.float32)
    x_min, y_min, x_max, y_max = np.rint(bounding_box * img.shape[-1]).to(torch.int32)
    mask[:, y_min:y_max, x_min:x_max] = img[:, y_min:y_max, x_min:x_max]
    return mask
