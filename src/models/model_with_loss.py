import torch
from torch import nn
from abc import abstractmethod
from typing import Tuple
from abc import ABC


class ModelWithLoss(ABC, nn.Module):
    @abstractmethod
    def loss(
        self,
        reconstruction_target_images: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
