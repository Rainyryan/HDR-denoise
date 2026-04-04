import torch
import torch.nn as nn
from blocks_Restormer import RestormerBlock
from CoDe_Container import CoDe_Container

class CoDeBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 1. Initialize Restormer as the high-performance base_model
        # Restormer is ideal for high-res HDR because of its linear complexity [cite: 983, 1214]
        self.base_restormer = RestormerBlock(dim=dim, num_heads=num_heads)
        
        # 2. Wrap it in the CoDe_Container [cite: 13, 61, 67]
        # This allows the model to handle smooth and textured regions differently [cite: 47, 49]
        self.code_wrapper = CoDe_Container(base_model=self.base_restormer, n_components=2)

    def forward(self, x):
        # The container handles frequency decoupling, processing, and summation [cite: 165, 167, 190]
        return self.code_wrapper(x)
