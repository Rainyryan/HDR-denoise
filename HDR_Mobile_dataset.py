import os
import glob
import torch
from torch.utils.data import Dataset

def add_photon_noise(image: torch.Tensor, nbits: int = 10, random_alpha: bool = True, do_expand: bool = False) -> tuple:
    """
    Adds Poisson-Gaussian noise to a [4, H, W] packed Bayer float32 tensor.
    Channel order: B, G1, G2, R.
    Now includes Triangular low-light biasing and Expand Mode offsets.
    """
    pix_max = float(2 ** nbits - 1)
    image = image.clone()

    img_min = image.min()
    img_max = image.max()
    if (img_max - img_min) > 1e-6:
        image = (image - img_min) / (img_max - img_min)
    image = image * pix_max

    # 1. Restore the Triangular Distribution (Biased towards extreme low-light)
    if random_alpha:
        # PyTorch doesn't have a native triangular distribution, 
        # but we can approximate it by combining two uniform tensors
        u1 = torch.rand(1).item()
        u2 = torch.rand(1).item()
        # Roughly mimics your old (1, 2, 8) curve scaled to [0.1, 1.0]
        alpha = 0.1 + 0.9 * abs(u1 - u2) 
    else:
        alpha = 1.0

    clean = image * alpha
    gt = image * alpha

    # 2. Restore Expand Mode (Extreme Highlights)
    if do_expand and torch.rand(1).item() < 0.3:
        # Adds 5-bit to 11-bit global offset to 30% of data
        offset = 2 ** (torch.rand(1).item() * 6.0 + 5.0)
        gt = gt + offset
        clean = clean + offset

    # 3. Sensor noise model (Var = 14 * Signal + Read Noise)
    iso = torch.empty(1).uniform_(100, 6400).item()
    shot_noise = clean / pix_max          
    read_noise = (iso / 1600) * 0.01      
    noise_sigma = torch.sqrt(shot_noise * 14.0 + read_noise * pix_max)
    
    noisy = clean + noise_sigma * torch.randn_like(clean)

    noisy = torch.clamp(torch.round(noisy), 0.0, pix_max) / pix_max
    gt = torch.clamp(gt, 0.0, pix_max) / pix_max

    return noisy, gt


class MobileHDRDataset(Dataset):
    # Added num_patch to the inputs (default 64)
    def __init__(self, base_dir: str, split: str = "train", transform=None, nbits: int = 10, random_alpha: bool = True, num_patch: int = 16):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.nbits = nbits
        self.random_alpha = random_alpha
        self.do_expand = False
        self.num_patch = num_patch # Store the multiplier

        if self.split == "train":
            self.tensor_dir = os.path.join(base_dir, "train", "tensors")
            self.file_list = glob.glob(os.path.join(self.tensor_dir, "**", "*.pt"), recursive=True)
        elif self.split == "test":
            self.tensor_dir = os.path.join(base_dir, "test", "tensors", "with_gt")
            self.file_list = glob.glob(os.path.join(self.tensor_dir, "*.pt"))

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .pt tensors found in {self.tensor_dir}.")

    def __len__(self) -> int:
        # THE TRICK: Multiply the length for training so an epoch is the correct size!
        if self.split == "train":
            return len(self.file_list) * self.num_patch
        return len(self.file_list) # Testing should just be 1-to-1

    def regen_crops(self):
        pass

    def regen_noise(self):
        pass

    def __getitem__(self, idx: int) -> dict:
        # Use modulo arithmetic to wrap the virtual index back to the real file list
        actual_idx = idx % len(self.file_list)
        tensor_path = self.file_list[actual_idx]

        clean_hdr = torch.load(tensor_path, weights_only=True)

        assert clean_hdr.dim() == 3 and clean_hdr.shape[0] == 4, \
            f"Expected [4, H, W] Bayer tensor, got {clean_hdr.shape} in {tensor_path}"

        noisy_hdr, gt_hdr = add_photon_noise(
            clean_hdr,
            nbits=self.nbits,
            random_alpha=self.random_alpha if self.split == "train" else False,
            do_expand=self.do_expand
        )

        if self.transform:
            noisy_hdr, gt_hdr = self.transform(noisy_hdr, gt_hdr)

        return {
            'x': noisy_hdr,
            'xm': noisy_hdr,
            'y': gt_hdr,
        }