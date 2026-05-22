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
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from HDR_model_hybrid_Teacher import TransUNet_Teacher_HDR as HDR_model
from HDR_Mobile_dataset import MobileHDRDataset
from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR

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


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def mu_law_tonemap(x, mu=5000):
    return torch.log1p(mu * x) / torch.log1p(torch.tensor(mu, dtype=x.dtype, device=x.device))


def hdr_tonemap(x, nbits=20):
    mu = 2 ** nbits - 1
    return torch.log10(1.0 + mu * x) / torch.log10(torch.tensor(1.0 + mu, device=x.device))


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


# ─────────────────────────────────────────────────────────────────────────────
# Patch-based inference
# ─────────────────────────────────────────────────────────────────────────────
def infer_patches(model, noisy_bayer, patch_size=256, overlap=32, device='cuda'):
    """
    Runs the model on a single [1, 4, H, W] image using overlapping patches.
    Handles any image size by reflection-padding to a multiple of patch_size.

    overlap: pixel overlap between adjacent patches (reduces boundary artifacts).
    Uses a cosine blending window to feather patch seams.
    """
    _, C, H, W = noisy_bayer.shape
    stride = patch_size - overlap

    # Pad to ensure full patch coverage
    pad_h = (math.ceil((H - overlap) / stride) * stride + overlap) - H
    pad_w = (math.ceil((W - overlap) / stride) * stride + overlap) - W
    x_pad = F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = x_pad.shape

    output     = torch.zeros_like(x_pad)
    weight_sum = torch.zeros((1, 1, Hp, Wp), device=device)

    # Cosine blending window — smooth feathering at patch edges
    def cosine_window(size):
        ramp = torch.hann_window(size, device=device, periodic=False)
        return (ramp.unsqueeze(0) * ramp.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    win = cosine_window(patch_size)  # [1, 1, P, P]

    snr_full = estimate_local_snr_map(x_pad, window_size=5)  # computed once on full image

    for y in range(0, Hp - patch_size + 1, stride):
        for x in range(0, Wp - patch_size + 1, stride):
            patch    = x_pad[:, :, y:y+patch_size, x:x+patch_size]
            snr_crop = snr_full[:, :, y:y+patch_size, x:x+patch_size]

            with torch.amp.autocast('cuda'):
                pred, _, _ = model(patch, snr_crop)

            output[:, :, y:y+patch_size, x:x+patch_size] += pred * win
            weight_sum[:, :, y:y+patch_size, x:x+patch_size] += win

    output = output / (weight_sum + 1e-8)
    return output[:, :, :H, :W]   # crop padding back off


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
    CHECKPOINT     = "models_MoE_Teacher_MobileHDR_20260521_1923/preexpand_hdr_best.pth"
    OUTPUT_DIR     = "test_results/"
    PATCH_SIZE     = 256
    PATCH_OVERLAP  = 32          # overlap to reduce boundary seam artifacts
    SAVE_EVERY     = 1           # save visuals for every N-th test image (1 = all)
    NBITS          = 20
    DEVICE         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MODEL_KWARGS = {
        "dim": 32,
        "num_blocks": [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8],
    }
    # ───────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "rgb"), exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model = DualSNRDenoiser(**MODEL_KWARGS).to(DEVICE)

    # Load weights FIRST, then compile — order matters
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    # Handle torch.compile prefix (_orig_mod.) if checkpoint was saved compiled
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
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
        "psnr_raw_linear":  [],
        "psnr_raw_mu":      [],
        "ssim_raw_linear":  [],
        "psnr_rgb_linear":  [],
        "psnr_rgb_mu":      [],
        "ssim_rgb_linear":  [],
        "time_sec":         [],
    }

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "sample_idx",
        "psnr_raw_linear", "psnr_raw_mu", "ssim_raw_linear",
        "psnr_rgb_linear", "psnr_rgb_mu", "ssim_rgb_linear",
        "time_sec",
    ])

    print("=" * 65)
    print(f"{'#':>4}  {'PSNR-raw-lin':>12}  {'PSNR-raw-µ':>10}  "
          f"{'PSNR-rgb-lin':>12}  {'PSNR-rgb-µ':>10}  {'Time(s)':>8}")
    print("=" * 65)

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            x  = sample["x"].to(DEVICE, non_blocking=True)   # [1, 4, H, W] noisy
            y  = sample["y"].to(DEVICE, non_blocking=True)   # [1, 4, H, W] clean GT

            # ── Timed inference (patch-based) ──────────────────────────
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            y_pred = infer_patches(model, x, patch_size=PATCH_SIZE,
                                   overlap=PATCH_OVERLAP, device=DEVICE)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # ── RAW Bayer domain metrics ───────────────────────────────
            psnr_raw_lin = psnr(y_pred.clamp(0, 1), y.clamp(0, 1))
            psnr_raw_mu  = psnr(mu_law_tonemap(y_pred.clamp(0, 1)),
                                mu_law_tonemap(y.clamp(0, 1)))
            ssim_raw_lin = ssim(y_pred.clamp(0, 1), y.clamp(0, 1))

            # ── RGB domain metrics (demosaic then evaluate) ────────────
            mosaic_pred = packed_bayer_to_mosaic(y_pred.clamp(0, 1))
            mosaic_gt   = packed_bayer_to_mosaic(y.clamp(0, 1))
            mosaic_noisy = packed_bayer_to_mosaic(x.clamp(0, 1))

            with torch.amp.autocast('cuda'):
                rgb_pred  = gbtf(mosaic_pred).clamp(0, 1)    # [1, 3, 2H, 2W]
                rgb_gt    = gbtf(mosaic_gt).clamp(0, 1)
                rgb_noisy = gbtf(mosaic_noisy).clamp(0, 1)

            psnr_rgb_lin = psnr(rgb_pred, rgb_gt)
            psnr_rgb_mu  = psnr(mu_law_tonemap(rgb_pred), mu_law_tonemap(rgb_gt))
            ssim_rgb_lin = ssim(rgb_pred, rgb_gt)

            # ── Log ────────────────────────────────────────────────────
            metrics["psnr_raw_linear"].append(psnr_raw_lin)
            metrics["psnr_raw_mu"].append(psnr_raw_mu)
            metrics["ssim_raw_linear"].append(ssim_raw_lin)
            metrics["psnr_rgb_linear"].append(psnr_rgb_lin)
            metrics["psnr_rgb_mu"].append(psnr_rgb_mu)
            metrics["ssim_rgb_linear"].append(ssim_rgb_lin)
            metrics["time_sec"].append(elapsed)

            csv_writer.writerow([
                i,
                f"{psnr_raw_lin:.4f}", f"{psnr_raw_mu:.4f}", f"{ssim_raw_lin:.4f}",
                f"{psnr_rgb_lin:.4f}", f"{psnr_rgb_mu:.4f}", f"{ssim_rgb_lin:.4f}",
                f"{elapsed:.3f}",
            ])
            csv_file.flush()

            print(f"{i+1:>4}  {psnr_raw_lin:>12.2f}  {psnr_raw_mu:>10.2f}  "
                  f"{psnr_rgb_lin:>12.2f}  {psnr_rgb_mu:>10.2f}  {elapsed:>8.3f}s")

            # ── Save RGB visuals ───────────────────────────────────────
            if i % SAVE_EVERY == 0:
                # Tone-map for visualization (HDR → displayable range)
                vis_noisy = hdr_tonemap(rgb_noisy, NBITS).clamp(0, 1)
                vis_pred  = hdr_tonemap(rgb_pred,  NBITS).clamp(0, 1)
                vis_gt    = hdr_tonemap(rgb_gt,    NBITS).clamp(0, 1)

                # Side-by-side: [Noisy | Denoised | GT]
                # Add a 2px white separator between panels
                sep = torch.ones(1, 3, vis_gt.shape[2], 2, device=DEVICE)
                comparison = torch.cat([vis_noisy, sep, vis_pred, sep, vis_gt], dim=3)
                save_image(comparison[0],
                           os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_compare.png"))

                # Also save individual panels for paper figures
                save_image(vis_noisy[0], os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_noisy.png"))
                save_image(vis_pred[0],  os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_denoised.png"))
                save_image(vis_gt[0],    os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_gt.png"))

    csv_file.close()

    # ── Summary ────────────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / len(lst)

    print("\n" + "=" * 65)
    print("                    FINAL RESULTS")
    print("=" * 65)
    print(f"  Samples evaluated:     {len(test_dataset)}")
    print(f"  Total time:            {sum(metrics['time_sec']):.2f}s")
    print(f"  Avg time / image:      {avg(metrics['time_sec']):.3f}s")
    if gflops:
        print(f"  GFLOPs / patch:        {gflops:.2f}")
    print()
    print(f"  ── RAW Bayer Domain ──────────────────────────────────")
    print(f"  PSNR-linear:           {avg(metrics['psnr_raw_linear']):.4f} dB")
    print(f"  PSNR-µ (µ=5000):       {avg(metrics['psnr_raw_mu']):.4f} dB")
    print(f"  SSIM-linear:           {avg(metrics['ssim_raw_linear']):.4f}")
    print()
    print(f"  ── RGB Domain (post GBTF demosaic) ───────────────────")
    print(f"  PSNR-linear:           {avg(metrics['psnr_rgb_linear']):.4f} dB")
    print(f"  PSNR-µ (µ=5000):       {avg(metrics['psnr_rgb_mu']):.4f} dB")
    print(f"  SSIM-linear:           {avg(metrics['ssim_rgb_linear']):.4f}")
    print("=" * 65)
    print(f"\n  Full per-sample results: {csv_path}")
    print(f"  RGB visuals saved to:    {os.path.join(OUTPUT_DIR, 'rgb')}/")