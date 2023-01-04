

from dataclasses import dataclass
from model_config.Config import ModelConfig


@dataclass
class ResnetConfig(ModelConfig):
    model_name = 'resnet152'
    source = 'pytorch/vision:v0.10.0'

    weighted_loss = False

    num_outputs: int