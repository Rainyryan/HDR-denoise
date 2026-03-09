import torch
import torch.nn as nn
import torch.nn.functional as F

class RepUNetBlock(nn.Module):
    """
    Inference-efficient block using Structural Re-parameterization.
    Ideal for real-time 24-bit HDR streams.
    """
    def __init__(self, dim, num_heads=None, deploy=False, **kwargs):
        super(RepUNetBlock, self).__init__()
        self.deploy = deploy
        self.dim = dim
        
        if deploy:
            self.reparam_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        else:
            # Multi-branch training structure
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim)
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim)
            )
            self.identity = nn.BatchNorm2d(dim)
        
        self.act = nn.GELU()

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam_conv(x))
        
        # Training: sum the branches
        out = self.branch_3x3(x) + self.branch_1x1(x) + self.identity(x)
        return self.act(out)

    def switch_to_deploy(self):
        """
        Logic to fuse branches into a single 3x3 kernel for inference.
        """
        # (Simplified for brevity: involves fusing Conv+BN then summing kernels)
        self.deploy = True