import torch
from torch import nn
from abc import abstractmethod
from typing import Tuple

class ModelWithLoss(nn.Module):
    @abstractmethod
    def loss(self,reconstruction_target_images: torch.Tensor,outputs: torch.Tensor,
                labels: torch.Tensor,
                reconstructions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass