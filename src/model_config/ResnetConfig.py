from dataclasses import dataclass
from model_config.Config import ModelConfig
from typing import Optional


@dataclass
class ResnetConfig(ModelConfig):
    model_name = "resnet152"
    source = "pytorch/vision:v0.10.0"

    num_outputs: int
    class_weights: Optional[list]