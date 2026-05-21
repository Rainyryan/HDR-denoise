import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def add_photon_noise(image, nbits=10, alpha=1, random_alpha=True):
    """
    Adds Poisson-Gaussian noise to a [C, H, W] float32 image array.
    Preserves white-balance scaling by operating on color planes safely.
    """
    pix_max = 2**nbits - 1
    
    # Work on a copy to prevent overriding cache states
    image = image.copy()
    
    # Normalize securely to prevent dividing by zero variations
    img_min = image.min()
    img_max = image.max()
    if (img_max - img_min) > 1e-6:
        image = (image - img_min) / (img_max - img_min)
    image *= pix_max
    
    # Apply dynamic scene intensity shift to mimic exposure changes
    if random_alpha:
        alpha = np.random.random() * (1.0 - 0.25) + 0.25
        
    clean = image * alpha
    gt = image * alpha
    
    # Synthesize sensor characteristics based on Enze's calibrated parameters
    # Noise Variance = 14 * Signal + Uniform(135, 160)
    rand_read_floor = np.random.random() * (160.0 - 135.0) + 135.0
    noise_sigma = np.sqrt(14.0 * clean + rand_read_floor)
    
    # Generate Gaussian distribution mirroring the precise matrix shape
    noisy = clean + noise_sigma * np.random.randn(*clean.shape)
    
    # Standardize integer discretization and normalize back to float workspace [0, 1]
    noisy = np.clip(np.round(noisy), 0, pix_max) / pix_max
    gt = np.clip(gt, 0, pix_max) / pix_max
    
    return noisy.astype(np.float32), gt.astype(np.float32)


class MobileHDRDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, nbits=10, random_alpha=True):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.nbits = nbits
        self.random_alpha = random_alpha
        self.do_expand = False 
        
        if self.split == "train":
            self.tensor_dir = os.path.join(base_dir, "train", "tensors")
            self.file_list = glob.glob(os.path.join(self.tensor_dir, "**", "*.npz"), recursive=True)
        elif self.split == "test":
            self.tensor_dir = os.path.join(base_dir, "test", "tensors", "with_gt")
            self.file_list = glob.glob(os.path.join(self.tensor_dir, "*.npz"))
            
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npz arrays found in {self.tensor_dir}.")

    def __len__(self):
        return len(self.file_list)

    def regen_crops(self):
        pass 

    def regen_noise(self):
        pass 

    def __getitem__(self, idx):
        tensor_path = self.file_list[idx]
        
        # 1. Read only the ground-truth HDR target matrix
        with np.load(tensor_path) as data:
            clean_hdr = data['hdr'].astype(np.float32)
            
        # 2. Check and normalize spatial configuration formatting to [C, H, W]
        if clean_hdr.shape[-1] == 3:
            clean_hdr = np.transpose(clean_hdr, (2, 0, 1))

        # 3. Process the ground-truth image through your sensor noise generator
        # Only apply synthetic noise loops during training cycles to mimic physical conditions
        if self.split == "train":
            noisy_hdr, gt_hdr = add_photon_noise(
                clean_hdr, 
                nbits=self.nbits, 
                random_alpha=self.random_alpha
            )
        else:
            # During evaluation cycles, match baseline exposure profiles deterministically
            noisy_hdr, gt_hdr = add_photon_noise(
                clean_hdr, 
                nbits=self.nbits, 
                random_alpha=False
            )
        
        # 4. Cast directly to PyTorch tensors safely
        x_tensor = torch.from_numpy(noisy_hdr).float()
        y_tensor = torch.from_numpy(gt_hdr).float()
        
        # 5. Apply spatial augmentations (e.g., cropping, rotations)
        if self.transform:
            x_tensor, y_tensor = self.transform(x_tensor, y_tensor)

        return {
            'x': x_tensor,
            'xm': x_tensor,
            'y': y_tensor,
        }