from dataclasses import dataclass
from typing import Optional, Union
from torch import Tensor

@dataclass
class ResnetConfig():
    model_name = "resnet152"
    source = "pytorch/vision:v0.10.0"
   
    num_outputs: int
    class_weights: Optional[Union[list, Tensor]]