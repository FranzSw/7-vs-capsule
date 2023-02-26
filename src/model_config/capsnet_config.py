from dataclasses import dataclass
from typing import Union
import torch
@dataclass
class Config:
    model_name: str = 'Capsnet'
    def __init__(self, dc_num_capsules=4, input_width=32, input_height= 32, reconstruction_loss_factor = 0.0005, cnn_in_channels=3, dc_in_channels=8, out_capsule_size=16, class_weights: Union[torch.Tensor, None] = None, num_iterations: int = 3):
       
        # CNN 
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = 256
        self.cnn_kernel_size = 9
        # Primary Capsule (pc)
        self.pc_num_capsules = 8
        self.pc_in_channels = 256
        self.pc_out_channels = 32
        self.pc_kernel_size = 9
        self.pc_num_routes = 32 * 8 * 8
        
        # Digit Capsule (dc)
        self.dc_num_capsules = dc_num_capsules
        self.dc_num_routes = 32 * 8 * 8
        self.dc_in_channels = dc_in_channels
        self.dc_out_channels = out_capsule_size
        self.dc_num_iterations = num_iterations

        # Decoder
        self.input_width = input_width
        self.input_height = input_height  
        self.reconstruction_loss_factor = reconstruction_loss_factor
        
        self.class_weights = class_weights
      
    