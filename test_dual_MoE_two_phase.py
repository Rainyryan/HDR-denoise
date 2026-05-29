"""
test_DualMoE.py — Comprehensive Benchmark for DualSNR HDR Denoiser
====================================================================
Features:
  - Patch-based inference (256x256) with reflection padding for any image size
  - Metrics in both RAW Bayer domain and RGB domain (after GBTF demosaicing)
    • PSNR-linear, PSNR-µ (µ=5000), SSIM
  - FLOPs estimation (via torchinfo if available, else skipped gracefully)
  - Per-image wall-clock timing
  - Saves side-by-side RGB comparison: Noisy | Denoised | GT
  - CSV results log for easy analysis
"""

import os
import csv
import sys
import time
import math
from numpy import size
from matplotlib.pylab import size
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from HDR_model_hybrid_Teacher import TransUNet_Teacher_HDR as HDR_model
from HDR_Mobile_dataset import MobileHDRDataset
from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR


# ─────────────────────────────────────────────────────────────────────────────
# Tee: mirror all print() output to both stdout and a log file simultaneously
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    """
    Replaces sys.stdout so every print() writes to both the terminal and
    a log file.  Call Tee.close() (or use as a context manager) when done.

    Usage:
        sys.stdout = Tee(log_path)
        ...
        sys.stdout.close()
    """
    def __init__(self, path: str):
        self._terminal = sys.__stdout__
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._log = open(path, "w", buffering=1)   # line-buffered

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper (must match training)
# ─────────────────────────────────────────────────────────────────────────────
class DualSNRDenoiser(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser_low_snr  = HDR_model(**kwargs)
        self.denoiser_high_snr = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        out_low  = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)
        return (1.0 - snr_map) * out_low + snr_map * out_high, out_low, out_high


class SingleDenoiser(nn.Module):
    """Baseline: one denoiser, no SNR routing. Matches training wrapper exactly."""
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        out = self.denoiser(x)
        return out, out, out


def load_model_from_checkpoint(checkpoint_path, model_kwargs, device):
    """
    Reads the 'mode' field saved in the checkpoint to auto-select
    DualSNRDenoiser vs SingleDenoiser, then loads weights.
    Falls back to DualSNRDenoiser if the field is absent (legacy checkpoints).
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mode  = ckpt.get("mode", "dual")
    print(f"  Checkpoint mode: '{mode}'")

    model = (DualSNRDenoiser(**model_kwargs) if mode == "dual"
             else SingleDenoiser(**model_kwargs)).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    # Strip torch.compile prefix if checkpoint was saved while compiled
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model, mode


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def hdr_tonemap(x, mu=5000):
    """
    µ-law tonemapping: log1p(µ·x) / log1p(µ).
    mu=5000 matches training and the HDR imaging literature standard
    (Kalantari SIGGRAPH 2017). All metrics and visualisations use this.
    """
    mu_t = torch.tensor(mu, dtype=x.dtype, device=x.device)
    return torch.log1p(mu_t * x) / torch.log1p(mu_t)

def psnr(pred, gt, data_range=1.0):
    """Per-image PSNR, returns mean over batch."""
    with torch.no_grad():
        mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
        p = torch.where(mse == 0,
                        torch.tensor(100.0, device=pred.device),
                        10.0 * torch.log10(data_range ** 2 / mse))
        return p.mean().item()


def ssim(pred, gt, window_size=11, data_range=1.0):
    """
    Structural Similarity — averaged over batch and channels.
    Pure PyTorch, no external dependency.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C  = pred.shape[1]

    # Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1  = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    s2  = F.conv2d(gt   * gt,   kernel, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(pred * gt,   kernel, padding=pad, groups=C) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()


def estimate_local_snr_map(x, window_size=5, eps=1e-5):
    unbatched = x.dim() == 3
    if unbatched:
        x = x.unsqueeze(0)
    pad = window_size // 2
    local_mean    = F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=pad)
    local_sq_mean = F.avg_pool2d(x ** 2, kernel_size=window_size, stride=1, padding=pad)
    local_var     = torch.clamp(local_sq_mean - local_mean ** 2, min=0.0)
    local_std     = torch.sqrt(local_var + eps)
    raw_snr       = local_mean / local_std
    spatial_snr   = raw_snr.mean(dim=1, keepdim=True)
    batch_max     = spatial_snr.amax(dim=(2, 3), keepdim=True)
    snr_norm      = spatial_snr / (batch_max + eps)
    if unbatched:
        snr_norm = snr_norm.squeeze(0)
    return snr_norm


def packed_bayer_to_mosaic(packed):
    """
    [B, 4, h, w] packed Bayer (B, G1, G2, R) -> [B, 1, 2h, 2w] full-res BGGR mosaic.
    Used as input to DifferentiableGBTF_BGGR.
    """
    B, _, h, w = packed.shape
    mosaic = torch.zeros((B, 1, h * 2, w * 2), device=packed.device, dtype=packed.dtype)
    mosaic[:, 0, 0::2, 0::2] = packed[:, 0]   # B
    mosaic[:, 0, 0::2, 1::2] = packed[:, 1]   # G1
    mosaic[:, 0, 1::2, 0::2] = packed[:, 2]   # G2
    mosaic[:, 0, 1::2, 1::2] = packed[:, 3]   # R
    return mosaic

def save_jpg(tensor, path, quality=92):
    """Save a [C, H, W] float tensor as JPEG. save_image doesn't support quality kwarg."""
    to_pil_image(tensor.clamp(0, 1).cpu()).save(path, format="JPEG", quality=quality)

# ─────────────────────────────────────────────────────────────────────────────
# Patch-based inference
# ─────────────────────────────────────────────────────────────────────────────
def infer_patches(model, noisy_bayer, patch_size=256, overlap=32, device='cuda'):
    """
    Runs the model on a single [1, 4, H, W] image using overlapping patches.
    Handles any image size by reflection-padding to a multiple of patch_size.

    overlap: pixel overlap between adjacent patches (reduces boundary artifacts).
    Uses a cosine blending window to feather patch seams.

    Returns: (blended, out_low_snr, out_high_snr) — all [1, 4, H, W].
    """
    _, C, H, W = noisy_bayer.shape
    stride = patch_size - overlap

    # Pad to ensure full patch coverage
    pad_h = (math.ceil((H - overlap) / stride) * stride + overlap) - H
    pad_w = (math.ceil((W - overlap) / stride) * stride + overlap) - W
    x_pad = F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = x_pad.shape

    output     = torch.zeros_like(x_pad)
    out_low    = torch.zeros_like(x_pad)   # low-SNR expert accumulator
    out_high   = torch.zeros_like(x_pad)   # high-SNR expert accumulator
    weight_sum = torch.zeros((1, 1, Hp, Wp), device=device)

    # # Cosine blending window — smooth feathering at patch edges
    # def cosine_window(size):
    #     ramp = torch.hann_window(size, device=device, periodic=False)
    #     return (ramp.unsqueeze(0) * ramp.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    def cosine_window(size):
        # Simple average over overlapping patches — no feathering, no dark edges
        return torch.ones((1, 1, size, size), device=device)

    win = cosine_window(patch_size)  # [1, 1, P, P]

    snr_full = estimate_local_snr_map(x_pad, window_size=5)  # computed once on full image

    for y in range(0, Hp - patch_size + 1, stride):
        for x in range(0, Wp - patch_size + 1, stride):
            patch    = x_pad[:, :, y:y+patch_size, x:x+patch_size]
            snr_crop = snr_full[:, :, y:y+patch_size, x:x+patch_size]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred, pred_low, pred_high = model(patch, snr_crop)

            output  [:, :, y:y+patch_size, x:x+patch_size] += pred      * win
            out_low [:, :, y:y+patch_size, x:x+patch_size] += pred_low  * win
            out_high[:, :, y:y+patch_size, x:x+patch_size] += pred_high * win
            weight_sum[:, :, y:y+patch_size, x:x+patch_size] += win

    denom = weight_sum + 1e-8
    return (
        output  [:, :, :H, :W] / denom[:, :, :H, :W],
        out_low [:, :, :H, :W] / denom[:, :, :H, :W],
        out_high[:, :, :H, :W] / denom[:, :, :H, :W],
    )

def infer_full(model, noisy_bayer, device='cuda'):
    """
    Full-resolution inference — no patches, no blending artifacts.

    The model's PixelUnshuffle chain requires H and W divisible by 8.
    We reflect-pad if needed and crop back after inference.
    For MobileHDR (2040×1528 packed) no padding is required.

    Returns: (pred, pred_low, pred_high) — all [1, 4, H, W]
    """
    _, _, H, W = noisy_bayer.shape

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x = (F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='reflect')
         if pad_h or pad_w else noisy_bayer)

    snr_map = estimate_local_snr_map(x, window_size=5)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred, pred_low, pred_high = model(x, snr_map)

    return pred[:, :, :H, :W], pred_low[:, :, :H, :W], pred_high[:, :, :H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs estimation
# ─────────────────────────────────────────────────────────────────────────────
def estimate_flops(model, patch_size=256, device='cuda'):
    """
    Estimates GFLOPs for one 256x256 patch using torchinfo.
    Runs on the underlying uncompiled model to avoid TorchDynamo
    cache thrashing from torchinfo's multi-shape probing.
    Gracefully skipped if torchinfo is not installed.
    """
    try:
        from torchinfo import summary
        import torch._dynamo as dynamo

        # Unwrap torch.compile wrapper if present — torchinfo probes many
        # input shapes which exhausts dynamo's cache_size_limit
        raw_model = dynamo.disable(model) if hasattr(dynamo, 'disable') else model

        dummy_x   = torch.zeros(1, 4, patch_size, patch_size, device=device)
        dummy_snr = torch.zeros(1, 1, patch_size, patch_size, device=device)

        stats = summary(raw_model, input_data=(dummy_x, dummy_snr),
                        verbose=0, mode='eval')
        gflops = stats.total_mult_adds / 1e9
        params_m = stats.total_params / 1e6
        print(f"  Parameters:             {params_m:.2f} M")
        print(f"  FLOPs (one {patch_size}x{patch_size} patch): {gflops:.2f} GFLOPs")
        return gflops
    except ImportError:
        print("  [torchinfo not installed — skipping FLOPs. Run: pip install torchinfo]")
        return None
    except Exception as e:
        print(f"  [FLOPs estimation failed: {e}]")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Performance flags ──────────────────────────────────────────────────
    # TF32 gives ~2x throughput on A100 tensor cores with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision('high')

    # ── Configuration ──────────────────────────────────────────────────────
    DATASET_DIR    = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    # Update this to the phase1_best.pth from your latest training run.
    # New naming convention: models_p{PHASE}_{mode}_Teacher_MobileHDR_{timestamp}/phase{PHASE}_best.pth
    CHECKPOINT     = "models_p1_dual_Teacher_MobileHDR_20260525_0740/phase1_best.pth"
    OUTPUT_DIR     = f"test_results/{CHECKPOINT.split('/')[0]}"
    PATCH_SIZE     = 1024
    PATCH_OVERLAP  = PATCH_SIZE // 4
    SAVE_EVERY     = 1
    # mu=5000 matches training (train_two_phase.py uses mu=5000).
    # Using a different value makes test PSNR-µ incomparable to W&B training curves.
    METRIC_MU      = 5000
    DEVICE         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MODEL_KWARGS = {
        "dim":                   32,
        "num_blocks":            [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads":                 [1, 2, 4, 8],
        "se_reduction":          8,    # required by SEResidualBlock in new model
    }
    # ───────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "rgb"), exist_ok=True)

    # ── Log file — mirrors all print() output to OUTPUT_DIR/log.txt ───────
    log_path   = os.path.join(OUTPUT_DIR, "log.txt")
    sys.stdout = Tee(log_path)
    print(f"Logging to: {log_path}")

    # ── Load model (auto-detects dual vs single from checkpoint) ───────────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model, mode_tag = load_model_from_checkpoint(CHECKPOINT, MODEL_KWARGS, DEVICE)
    print("  Weights loaded.")

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("  Model compiled with TorchInductor.")
        except Exception as e:
            print(f"  Skipping compile: {e}")

    model.eval()

    # ── Differentiable GBTF demosaicing ────────────────────────────────────
    gbtf = DifferentiableGBTF_BGGR().to(DEVICE)
    gbtf.eval()

    # ── FLOPs ──────────────────────────────────────────────────────────────
    print("\nEstimating model complexity...")
    gflops = estimate_flops(model, patch_size=PATCH_SIZE, device=DEVICE)

    # ── Dataset ────────────────────────────────────────────────────────────
    print("\nLoading test dataset...")
    test_dataset = MobileHDRDataset(
        base_dir=DATASET_DIR,
        split="test",
        transform=None      # no augmentation at test time
    )
    # batch_size=1: process one full image at a time for patch inference
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  {len(test_dataset)} test samples found.\n")

    # ── Metrics accumulators ───────────────────────────────────────────────
    metrics = {
        # Noisy input baseline (so we can report Δ PSNR)
        "psnr_noisy_raw_mu":    [],
        "psnr_noisy_rgb_mu":    [],
        # Denoised output
        "psnr_raw_linear":  [],
        "psnr_raw_mu":      [],
        "ssim_raw_linear":  [],
        "ssim_raw_mu":      [],
        "psnr_rgb_linear":  [],
        "psnr_rgb_mu":      [],
        "ssim_rgb_linear":  [],
        "ssim_rgb_mu":      [],
        # Expert routing (dual mode only; filled with None for single mode)
        "pct_low_snr_pixels":   [],
        "psnr_expert_low_mu":   [],
        "psnr_expert_high_mu":  [],
        "time_sec":         [],
    }

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "sample_idx",
        "psnr_noisy_raw_mu", "psnr_noisy_rgb_mu",
        "psnr_raw_linear", "psnr_raw_mu", "ssim_raw_linear", "ssim_raw_mu",
        "psnr_rgb_linear", "psnr_rgb_mu", "ssim_rgb_linear", "ssim_rgb_mu",
        "pct_low_snr_pixels", "psnr_expert_low_mu", "psnr_expert_high_mu",
        "time_sec",
    ])

    print("=" * 80)
    print(f"{'#':>4}  {'PSNR-raw-µ(noisy)':>18}  {'PSNR-raw-µ':>10}  "
          f"{'PSNR-rgb-µ':>10}  {'%LowSNR':>8}  {'Time(s)':>8}")
    print("=" * 80)

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            x  = sample["x"].to(DEVICE, non_blocking=True)   # [1, 4, H, W] noisy
            y  = sample["y"].to(DEVICE, non_blocking=True)   # [1, 4, H, W] clean GT

            # ── SNR routing stats (computed before inference) ──────────
            snr_full = estimate_local_snr_map(x, window_size=5)
            pct_low  = (snr_full < 0.5).float().mean().item() * 100.0

            # ── Timed inference (patch-based) ──────────────────────────
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # y_pred, y_low_snr, y_high_snr = infer_patches(
            #     model, x, patch_size=PATCH_SIZE,
            #     overlap=PATCH_OVERLAP, device=DEVICE
            # )
            y_pred, y_low_snr, y_high_snr = infer_full(model, x, device=DEVICE)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # ── Cast to float32 for metric computation ─────────────────
            y_pred_c  = y_pred.clamp(0, 1).float()
            y_c       = y.clamp(0, 1).float()
            x_c       = x.clamp(0, 1).float()
            y_low_c   = y_low_snr.clamp(0, 1).float()
            y_high_c  = y_high_snr.clamp(0, 1).float()

            # ── RAW Bayer domain metrics (mu=METRIC_MU everywhere) 
            psnr_raw_lin      = psnr(y_pred_c, y_c)
            psnr_raw_mu       = psnr(hdr_tonemap(y_pred_c, METRIC_MU),
                                     hdr_tonemap(y_c,      METRIC_MU))
            ssim_raw_lin      = ssim(y_pred_c, y_c)
            ssim_raw_mu       = ssim(hdr_tonemap(y_pred_c, METRIC_MU),
                                     hdr_tonemap(y_c,      METRIC_MU))
            psnr_noisy_raw_mu = psnr(hdr_tonemap(x_c,      METRIC_MU),
                                     hdr_tonemap(y_c,      METRIC_MU))

            # ── Per-expert PSNR (informative for dual mode; identical for single)
            psnr_exp_low  = psnr(hdr_tonemap(y_low_c,  METRIC_MU),
                                  hdr_tonemap(y_c,      METRIC_MU))
            psnr_exp_high = psnr(hdr_tonemap(y_high_c, METRIC_MU),
                                  hdr_tonemap(y_c,      METRIC_MU))

            # ── RGB domain metrics (demosaic then evaluate) ────────────
            mosaic_pred  = packed_bayer_to_mosaic(y_pred_c)
            mosaic_gt    = packed_bayer_to_mosaic(y_c)
            mosaic_noisy = packed_bayer_to_mosaic(x_c)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                rgb_pred  = gbtf(mosaic_pred).clamp(0, 1).float()
                rgb_gt    = gbtf(mosaic_gt).clamp(0, 1).float()
                rgb_noisy = gbtf(mosaic_noisy).clamp(0, 1).float()

            psnr_rgb_lin      = psnr(rgb_pred, rgb_gt)
            psnr_rgb_mu       = psnr(hdr_tonemap(rgb_pred,  METRIC_MU),
                                     hdr_tonemap(rgb_gt,    METRIC_MU))
            ssim_rgb_lin      = ssim(rgb_pred, rgb_gt)
            ssim_rgb_mu       = ssim(hdr_tonemap(rgb_pred,  METRIC_MU),
                                     hdr_tonemap(rgb_gt,    METRIC_MU))
            psnr_noisy_rgb_mu = psnr(hdr_tonemap(rgb_noisy, METRIC_MU),
                                     hdr_tonemap(rgb_gt,    METRIC_MU))

            # ── Accumulate ─────────────────────────────────────────────
            metrics["psnr_noisy_raw_mu"].append(psnr_noisy_raw_mu)
            metrics["psnr_noisy_rgb_mu"].append(psnr_noisy_rgb_mu)
            metrics["psnr_raw_linear"].append(psnr_raw_lin)
            metrics["psnr_raw_mu"].append(psnr_raw_mu)
            metrics["ssim_raw_linear"].append(ssim_raw_lin)
            metrics["ssim_raw_mu"].append(ssim_raw_mu)
            metrics["psnr_rgb_linear"].append(psnr_rgb_lin)
            metrics["psnr_rgb_mu"].append(psnr_rgb_mu)
            metrics["ssim_rgb_linear"].append(ssim_rgb_lin)
            metrics["ssim_rgb_mu"].append(ssim_rgb_mu)
            metrics["pct_low_snr_pixels"].append(pct_low)
            metrics["psnr_expert_low_mu"].append(psnr_exp_low)
            metrics["psnr_expert_high_mu"].append(psnr_exp_high)
            metrics["time_sec"].append(elapsed)

            csv_writer.writerow([
                i,
                f"{psnr_noisy_raw_mu:.4f}", f"{psnr_noisy_rgb_mu:.4f}",
                f"{psnr_raw_lin:.4f}",  f"{psnr_raw_mu:.4f}",
                f"{ssim_raw_lin:.4f}",  f"{ssim_raw_mu:.4f}",
                f"{psnr_rgb_lin:.4f}",  f"{psnr_rgb_mu:.4f}",
                f"{ssim_rgb_lin:.4f}",  f"{ssim_rgb_mu:.4f}",
                f"{pct_low:.1f}", f"{psnr_exp_low:.4f}", f"{psnr_exp_high:.4f}",
                f"{elapsed:.3f}",
            ])
            csv_file.flush()

            print(f"{i+1:>4}  {psnr_noisy_raw_mu:>18.2f}  {psnr_raw_mu:>10.2f}  "
                  f"{psnr_rgb_mu:>10.2f}  {pct_low:>7.1f}%  {elapsed:>8.3f}s")
            
            
            # ── Save RGB visuals ───────────────────────────────────────
            # if i % SAVE_EVERY == 0:
            #     vis_noisy = hdr_tonemap(rgb_noisy, METRIC_MU).clamp(0, 1)
            #     vis_pred  = hdr_tonemap(rgb_pred,  METRIC_MU).clamp(0, 1)
            #     vis_gt    = hdr_tonemap(rgb_gt,    METRIC_MU).clamp(0, 1)

            #     sep = torch.ones(1, 3, vis_gt.shape[2], 2, device=DEVICE)
            #     comparison = torch.cat([vis_noisy, sep, vis_pred, sep, vis_gt], dim=3)
            #     save_image(comparison[0],
            #                os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_compare.png"))
            #     save_image(vis_noisy[0], os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_noisy.png"))
            #     save_image(vis_pred[0],  os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_denoised.png"))
            #     save_image(vis_gt[0],    os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_gt.png"))
            if i % SAVE_EVERY == 0:
                vis_noisy = hdr_tonemap(rgb_noisy, METRIC_MU).clamp(0, 1)
                vis_pred  = hdr_tonemap(rgb_pred,  METRIC_MU).clamp(0, 1)
                vis_gt    = hdr_tonemap(rgb_gt,    METRIC_MU).clamp(0, 1)

                # Downsample to half resolution before saving (full-res is ~32 MB per file)
                vis_noisy = F.interpolate(vis_noisy, scale_factor=0.5, mode='bilinear', align_corners=False)
                vis_pred  = F.interpolate(vis_pred,  scale_factor=0.5, mode='bilinear', align_corners=False)
                vis_gt    = F.interpolate(vis_gt,    scale_factor=0.5, mode='bilinear', align_corners=False)

                sep = torch.ones(1, 3, vis_gt.shape[2], 2, device=DEVICE)
                comparison = torch.cat([vis_noisy, sep, vis_pred, sep, vis_gt], dim=3)
                save_jpg(comparison[0], os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_compare.jpg"))
                save_jpg(vis_pred[0],   os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_denoised.jpg"))
                save_jpg(vis_gt[0],     os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_gt.jpg"))
                save_jpg(vis_noisy[0],  os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_noisy.jpg"))

    csv_file.close()

    # ── Summary ────────────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / len(lst)

    mu_label = f"µ={METRIC_MU}"

    print("\n" + "=" * 65)
    print(f"     FINAL RESULTS  [{mode_tag.upper()} mode | mu={METRIC_MU}]")
    print("=" * 65)
    print(f"  Samples evaluated:      {len(test_dataset)}")
    print(f"  Total time:             {sum(metrics['time_sec']):.2f}s")
    print(f"  Avg time / image:       {avg(metrics['time_sec']):.3f}s")
    if gflops:
        print(f"  GFLOPs / patch:         {gflops:.2f}")
    print()
    print(f"  ── Noisy Baseline ────────────────────────────────────")
    print(f"  PSNR-{mu_label} (RAW):    {avg(metrics['psnr_noisy_raw_mu']):.4f} dB")
    print(f"  PSNR-{mu_label} (RGB):    {avg(metrics['psnr_noisy_rgb_mu']):.4f} dB")
    print()
    print(f"  ── RAW Bayer Domain ──────────────────────────────────")
    print(f"  PSNR-linear:            {avg(metrics['psnr_raw_linear']):.4f} dB")
    delta_raw = avg(metrics['psnr_raw_mu']) - avg(metrics['psnr_noisy_raw_mu'])
    print(f"  PSNR-{mu_label}:          {avg(metrics['psnr_raw_mu']):.4f} dB  (delta +{delta_raw:.2f} dB)")
    print(f"  SSIM-linear:            {avg(metrics['ssim_raw_linear']):.4f}")
    print(f"  SSIM-{mu_label}:          {avg(metrics['ssim_raw_mu']):.4f}")
    print()
    print(f"  ── RGB Domain (post GBTF demosaic) ───────────────────")
    print(f"  PSNR-linear:            {avg(metrics['psnr_rgb_linear']):.4f} dB")
    delta_rgb = avg(metrics['psnr_rgb_mu']) - avg(metrics['psnr_noisy_rgb_mu'])
    print(f"  PSNR-{mu_label}:          {avg(metrics['psnr_rgb_mu']):.4f} dB  (delta +{delta_rgb:.2f} dB)")
    print(f"  SSIM-linear:            {avg(metrics['ssim_rgb_linear']):.4f}")
    print(f"  SSIM-{mu_label}:          {avg(metrics['ssim_rgb_mu']):.4f}")
    print()
    print(f"  ── Expert Routing ────────────────────────────────────")
    pct_lo = avg(metrics['pct_low_snr_pixels'])
    print(f"  Avg % -> low-SNR expert:   {pct_lo:.1f}%")
    print(f"  Avg % -> high-SNR expert:  {100.0 - pct_lo:.1f}%")
    print(f"  PSNR-{mu_label} low expert:  {avg(metrics['psnr_expert_low_mu']):.4f} dB")
    print(f"  PSNR-{mu_label} high expert: {avg(metrics['psnr_expert_high_mu']):.4f} dB")
    print("=" * 65)
    print(f"\n  Full per-sample results: {csv_path}")
    print(f"  RGB visuals saved to:    {os.path.join(OUTPUT_DIR, 'rgb')}/")
    print(f"  Log saved to:            {log_path}")
    sys.stdout.close()   # flush and restore stdout