from dataclasses import dataclass
from typing import Union
import numpy as np
import torch

@dataclass
class Config:
    model_name: str = 'Capsnet'
    def __init__(self, dc_num_capsules=4, input_width=32, input_height= 32, input_slices=30, reconstruction_loss_factor = 0.0005, cnn_in_channels=3, out_capsule_size=16, class_weights: Union[torch.Tensor, None] = None, num_iterations: int = 3):
       
        # CNN 
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = 256
        self.cnn_kernel_size = 9
        # Primary Capsule (pc)
        self.pc_num_capsules = 8
        self.pc_in_channels = 256
        self.pc_out_channels = 32
        # currently hard coded in model
        pc_stride_size = 2
        self.pc_kernel_size = 9
        self.pc_num_routes = 32 * 8 * 8
        
        # Digit Capsule (dc)
        self.dc_num_capsules = dc_num_capsules
        self.dc_num_routes = 32 * 8 * 8

        # Conv Out = out_channels + (64, 64, 30) - kernel_size + 1 -> (256, 56, 56, 22)
        # Conv 3D Out = out_channels, Conv_Out - kernel_size + 1 / stride_size -> (32, 24, 24, 7)
        # Primary Out = Conv 3D Out, pc_num_capsules * Conv 3D Out, squashed
        conv_out = np.array([input_width, input_height, input_slices]) - self.cnn_kernel_size + 1
        pc_conv_out = (conv_out - self.pc_kernel_size + 1) / pc_stride_size
        pc_out_total = self.pc_num_capsules * self.pc_out_channels * np.prod(pc_conv_out)
        pc_out = (self.dc_num_routes, int(pc_out_total / self.dc_num_routes))
        self.dc_in_channels = pc_out[1]
        self.dc_out_channels = out_capsule_size
        self.dc_num_iterations = num_iterations

        # Decoder
        self.input_width = input_width
        self.input_height = input_height  
        self.reconstruction_loss_factor = reconstruction_loss_factor
        
        self.class_weights = class_weights
      
    