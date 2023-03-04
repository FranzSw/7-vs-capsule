import torchvision
import torch
import shap
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(nchw_to_nhwc),
    ]
)
inv_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(nhwc_to_nchw),
        torchvision.transforms.Lambda(nchw_to_nhwc),
    ]
)


def softmax_output(outputs):
    classes = torch.sqrt((outputs**2).sum(2))
    output = torch.nn.Softmax(dim=1)(classes)
    return torch.squeeze(output.detach().cpu(), -1)


def write_bbox_in_matrix(data, bbox):
    width = data.shape[0]
    max = np.max(data)
    x_min, y_min, x_max, y_max = np.floor(bbox * width).numpy().astype(int)
    if x_max == width or y_max == width:
        return data
    data[x_min:x_max, y_min] = max
    data[x_max, y_min:y_max] = max
    data[x_min:x_max, y_max] = max
    data[x_min, y_min:y_max] = max
    return data


class ShapEvaluation:
    def __init__(self, model: nn.Module, input_sample_shape: tuple, class_names):
        self.model = model
        masker = shap.maskers.Image("blur(32, 32)", input_sample_shape)  # type: ignore

        self.explainer = shap.Explainer(
            self._predict, masker=masker, output_names=class_names
        )
        self.class_names = class_names

    def _predict(self, img_batch: torch.Tensor):
        img_batch = nhwc_to_nchw(torch.tensor(img_batch))
        is_single = False
        if img_batch.shape[0] == 1:
            is_single = True
            img_batch = img_batch.repeat(2, 1, 1, 1)

        img_batch = img_batch.to("cuda:0")
        output, _, _ = self.model(img_batch)
        if is_single:
            output = output[0:1, :]
        output = softmax_output(output)
        return output.detach().cpu()

    def evaluate(
        self,
        input_batch: torch.Tensor,
        bounding_boxes: Optional[list[np.ndarray]],
        output_path: str,
    ):
        input_images = transform(input_batch).cpu().numpy()
        shap_values = self.explainer(input_images, max_evals=500, batch_size=input_images.shape[0], outputs=shap.Explanation.argsort.flip[:2])  # type: ignore

        if bounding_boxes is not None:
            shap_values.data = np.array(
                list(map(write_bbox_in_matrix, shap_values.data, bounding_boxes))
            )

        shap.image_plot(shap_values, labels=[self.class_names] * input_images.shape[0])
        plt.savefig(output_path)
