import torch
import torch.nn as nn
import torch.nn.functional as F

class RestormerDistiller(nn.Module):
    def __init__(self, teacher_model, student_model, teacher_dim, student_dim):
        super(RestormerDistiller, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # 1. Freeze the teacher completely
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 2. Alignment layers (Student features -> Teacher dimensions) using 1x1 convolutions
        self.align_x_level1 = nn.Conv2d(student_dim, teacher_dim, kernel_size=1)
        self.align_x_level2 = nn.Conv2d(student_dim * 4, teacher_dim * 4, kernel_size=1)
        self.align_x_latent = nn.Conv2d(student_dim * 16, teacher_dim * 16, kernel_size=1)
        
    def forward(self, x):
        # Get Teacher features (No gradients needed)
        with torch.no_grad():
            t_out, t_maps = self.teacher(x, return_maps=True)
            
        # Get Student features
        s_out, s_maps = self.student(x, return_maps=True)
        
        # Align Student features to Teacher features
        aligned_s_maps = {
            "x_level1": self.align_x_level1(s_maps["x_level1"]),
            "x_level2": self.align_x_level2(s_maps["x_level2"]),
            "x_latent": self.align_x_latent(s_maps["x_latent"])
        }
        
        return s_out, t_out, aligned_s_maps, t_maps