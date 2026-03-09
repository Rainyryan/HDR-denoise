import torch
import torch.nn as nn
import torch.fft

class ContentDecouplingModule(nn.Module):
    def __init__(self, n_components=2):
        super(ContentDecouplingModule, self).__init__()
        self.n = n_components
        # alpha is learnable and determines the radius for the frequency mask 
        self.alphas = nn.Parameter(torch.sort(torch.rand(self.n - 1))[0])

    def forward(self, x):
        N, C, H, W = x.shape
        dct_spec = self.apply_dct2(x)
        
        # 1. Generate all raw soft masks first
        raw_masks = []
        for i in range(self.n):
            raw_masks.append(self.generate_frequency_mask(dct_spec.shape, i))
        
        # 2. Normalize: Ensure sum(masks) == 1.0 at every pixel
        # This prevents the "darkening" effect
        mask_stack = torch.stack(raw_masks, dim=0) # [n, 1, 1, H, W_reduced]
        total_mask_sum = mask_stack.sum(dim=0)
        normalized_masks = mask_stack / (total_mask_sum + 1e-8)
        
        sub_images = []
        for i in range(self.n):
            # Apply Inverse DCT to each normalized masked spectrum
            sub_images.append(self.apply_idct2(dct_spec * normalized_masks[i]))
            
        return sub_images
    
    def generate_frequency_mask(self, shape, index):
        N, C, H, W_reduced = shape # W is reduced due to rfft2
        # Create a coordinate grid for the DCT spectrum [cite: 140]
        # The DC component (0,0) is at the top-left [cite: 152]
        u = torch.linspace(0, 1, H, device=self.alphas.device)
        v = torch.linspace(0, 1, W_reduced, device=self.alphas.device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        
        # Calculate radius from the top-left (diagonal direction) [cite: 153]
        radius_grid = torch.sqrt(grid_u**2 + grid_v**2)
        
        # Define boundaries using learnable alphas [cite: 147, 160]
        # r0=0, rn=1.0 [cite: 150]
        boundaries = torch.cat([
            torch.tensor([0.0], device=self.alphas.device),
            self.alphas,
            torch.tensor([1.0], device=self.alphas.device)
        ])
        
        # Create mask for the specific subband [cite: 154]
        # mask = (radius_grid >= boundaries[index]) & (radius_grid < boundaries[index+1])
        
        # With a soft transition (Conceptual):
        temperature = 0.1  # Fixed temperature for soft masking [cite: 158]
        mask = torch.sigmoid((radius_grid - boundaries[index]) / temperature)
        if index < self.n - 1: # If not the last band, subtract the next step
            mask = mask - torch.sigmoid((radius_grid - boundaries[index+1]) / temperature)
        
        return mask.float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W_reduced]

    def apply_dct2(self, x):
        # Implementation of Eq. 1 [cite: 136]
        return torch.fft.rfft2(x, norm='ortho')

    def apply_idct2(self, spec):
        # Implementation of Eq. 2 [cite: 136]
        return torch.fft.irfft2(spec, norm='ortho')
        
import copy

class CoDe_Container(nn.Module):
    def __init__(self, base_model, n_components=2):
        super(CoDe_Container, self).__init__()
        self.cdm = ContentDecouplingModule(n_components)
        
        # Subnet 2 is typically the full-strength base model [cite: 176]
        self.subnet2 = base_model 
        
        # Subnet 1 is the 'modulated' version for smooth content [cite: 175, 176]
        # For your test, we can deepcopy the base_model or create a simpler block
        self.subnet1 = copy.deepcopy(base_model) 
        self.apply_modulation(self.subnet1, ratio=0.5)

    def apply_modulation(self, model, ratio):
        """
        Streamlines the sub-net by reducing layers or channels.
        For a block-level test, we ensure it is callable.
        """
        # In a real scenario, you would modify the internal layers here.
        # For shape verification, simply ensuring it's a valid module is enough.
        return model

    def forward(self, x):
        lq_sub_images = self.cdm(x)
        
        # Now these are actual modules, not None [cite: 167]
        hq1 = self.subnet1(lq_sub_images[0])
        hq2 = self.subnet2(lq_sub_images[1])
        
        # Summation for final reconstruction 
        return hq1 + hq2