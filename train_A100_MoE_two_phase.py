"""
train_two_phase.py — Two-phase HDR Teacher training on MobileHDR dataset
=========================================================================

Phase 1 — Patch training  (200 epochs, 512×512 crops, batch 8)
    The model learns all noise statistics quickly with diverse patch combinations.
    Larger effective batch size → stable gradients → can use higher LR.
    Bottleneck Restormer sees 64×64 = 4096 tokens — enough global context.

Phase 2 — Full-res fine-tune  (30 epochs, 2040×1528, batch 1)
    Closes the train/test distribution gap: patch statistics ≠ full-image
    statistics (boundary effects, global exposure, structured noise).
    Very small LR — adapt, don't relearn.

Usage
─────
  Phase 1:  set PHASE = 1, then submit job.
  Phase 2:  set PHASE = 2  AND  set PHASE1_CHECKPOINT to the best .pth from
            the Phase 1 run.  Phase 2 auto-loads it if no Phase 2 checkpoint
            exists in the current save folder.

D4 augmentation for BGGR packed Bayer
──────────────────────────────────────
Each 2×2 Bayer cell holds  B(0,0) G1(0,1) G2(1,0) R(1,1)  → packed channels [0,1,2,3].
Every spatial symmetry must be paired with a channel permutation that restores
the BGGR channel-spatial correspondence:

  H-flip     flip cols   → even col ↔ odd col  → B↔G1, G2↔R    permute [1,0,3,2]
  V-flip     flip rows   → even row ↔ odd row  → B↔G2, G1↔R    permute [2,3,0,1]
  180° rot   both flips  → compose above two   → B↔R,  G1↔G2   permute [3,2,1,0]
  Transpose  swap H,W    → (r,c)→(c,r)         → G1↔G2          permute [0,2,1,3]

Applying H-flip, V-flip, and Transpose independently at 50% each gives all 8
D4 symmetries with equal probability (Phase 1 — square patches only).
Full-res images are non-square so Transpose is skipped (Phase 2); H-flip and
V-flip independently already cover the 4 dimension-preserving symmetries.
"""

import os
import time
import lpips
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import wandb

from HDR_model_hybrid_Teacher import TransUNet_Teacher_HDR as HDR_model
from HDR_Mobile_dataset import MobileHDRDataset

os.environ["WANDB_CACHE_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
os.environ["WANDB_DATA_DIR"]  = "/scratch/gilbreth/chen4848/wandb_data"
os.environ["WANDB_DIR"]       = "/scratch/gilbreth/chen4848/projects/HDR-denoise"


##########################################################################
## Model wrappers
##########################################################################

class DualSNRDenoiser(nn.Module):
    """
    Two independent expert denoisers blended by a pixel-wise SNR map.
    Low-SNR expert specialises in high-noise (shadow) regions;
    high-SNR expert in low-noise (highlight) regions.
    The soft blend is: out = (1 - snr) * out_low + snr * out_high.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser_low_snr  = HDR_model(**kwargs)
        self.denoiser_high_snr = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        """
        x:       [B, 4, H, W] packed Bayer, values in [0, 1]
        snr_map: [B, 1, H, W] normalised SNR in [0, 1]
        Returns: (blended, out_low, out_high)
        """
        out_low  = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)
        blended  = (1.0 - snr_map) * out_low + snr_map * out_high
        return blended, out_low, out_high


class SingleDenoiser(nn.Module):
    """
    Baseline: one denoiser, no SNR routing.
    Returns the same 3-tuple as DualSNRDenoiser so the training loop
    requires zero changes.  aux_weight is set to 0 for this mode so
    the auxiliary losses have no effect.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        out = self.denoiser(x)
        return out, out, out


##########################################################################
## Logging utilities
##########################################################################

def print_running_loss(running_loss, running_psnr_mu, batch_sz, i, print_every=20):
    if i % print_every == (print_every - 1):
        psnr_str = "  PSNR-µ = %.2f" % (running_psnr_mu / (i + 1)) if running_psnr_mu else ""
        print("\r  step %5d  loss = %10.4f%s" % (
            i + 1, running_loss / (batch_sz * (i + 1)), psnr_str), end="")

def print_epoch_loss(epoch_num, avg_loss, avg_psnr_mu, avg_psnr, phase, time_spent=None):
    ts  = "?" if time_spent is None else "%.1fs" % time_spent
    msg = ("\r[P%d] Epoch %3d  loss=%10.4f  PSNR-µ=%6.2f  PSNR=%6.2f  (%s)"
           % (phase, epoch_num, avg_loss, avg_psnr_mu, avg_psnr, ts))
    print(msg)
    return msg

def write_log(filename, s, newfile=False, end="\n"):
    if newfile:
        open(filename, "w").close()
    with open(filename, "a") as f:
        f.write("%s%s" % (s, end))

def create_folder(directory):
    os.makedirs(directory, exist_ok=True)


##########################################################################
## GPU / tensor utilities
##########################################################################

def hdr_tonemap(x, mu=5000):
    """
    µ-law tonemapping: log1p(µ·x) / log1p(µ).
    Maps [0, 1] linear HDR → [0, 1] perceptually-uniform space.
    mu=5000 is the HDR imaging literature standard (Kalantari SIGGRAPH 2017).
    """
    mu_t = torch.tensor(mu, device=x.device, dtype=x.dtype)
    return torch.log1p(mu_t * x) / torch.log1p(mu_t)


def batch_psnr_gpu(img, gt, data_range=1.0):
    """Linear PSNR averaged over the batch, computed entirely on GPU."""
    with torch.no_grad():
        mse = torch.mean((img.detach() - gt.detach()) ** 2, dim=[1, 2, 3])
        p   = torch.where(mse == 0,
                          torch.tensor(100.0, device=img.device),
                          10.0 * torch.log10(data_range ** 2 / mse))
        return p.mean().item()


def estimate_local_snr_map(x, window_size=5, eps=1e-5):
    """
    Estimates a per-pixel SNR map from a noisy image batch.
    Returns [B, 1, H, W] normalised to [0, 1] per image.
    Used to route pixels between the two experts in DualSNRDenoiser.
    """
    unbatched = x.dim() == 3
    if unbatched:
        x = x.unsqueeze(0)
    pad           = window_size // 2
    local_mean    = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    local_sq_mean = F.avg_pool2d(x ** 2, window_size, stride=1, padding=pad)
    local_var     = torch.clamp(local_sq_mean - local_mean ** 2, min=0.0)
    local_std     = torch.sqrt(local_var + eps)
    spatial_snr   = (local_mean / local_std).mean(dim=1, keepdim=True)
    batch_max     = spatial_snr.amax(dim=(2, 3), keepdim=True)
    snr_norm      = spatial_snr / (batch_max + eps)
    if unbatched:
        snr_norm = snr_norm.squeeze(0)
    return snr_norm


def collate_pad_to_max(batch):
    """
    Pads variable-size full-resolution images to the largest H×W in the
    batch (rounded up to a multiple of 8) so PyTorch can stack them.
    Stores original sizes in orig_h / orig_w so padded pixels can be
    excluded from the loss via a binary valid_mask.
    Used only in Phase 2 (full-resolution) where images differ in size.
    """
    max_h = ((max(s["x"].shape[1] for s in batch) + 7) // 8) * 8
    max_w = ((max(s["x"].shape[2] for s in batch) + 7) // 8) * 8

    def pad(t):
        _, h, w = t.shape
        return F.pad(t, (0, max_w - w, 0, max_h - h), mode="reflect")

    return {
        "x":      torch.stack([pad(s["x"])  for s in batch]),
        "xm":     torch.stack([pad(s["xm"]) for s in batch]),
        "y":      torch.stack([pad(s["y"])  for s in batch]),
        "orig_h": torch.tensor([s["x"].shape[1] for s in batch]),
        "orig_w": torch.tensor([s["x"].shape[2] for s in batch]),
    }


def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """
    Linear warmup for warmup_epochs, then cosine annealing to eta_min.
    If warmup_epochs == 0 returns a plain CosineAnnealingLR.
    """
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=eta_min)


##########################################################################
## Augmentation transforms
##########################################################################

def make_patch_transform(patch_size: int):
    """
    Phase 1 transform: random 512×512 crop + full D4 augmentation.

    D4 group (8 symmetries) via three independent 50/50 coin flips:
      H-flip × V-flip × Transpose  →  2³ = 8 equally-likely outcomes.

    Channel permutations maintain BGGR packed-Bayer correspondence:
      H-flip    → [1, 0, 3, 2]   (B↔G1, G2↔R)
      V-flip    → [2, 3, 0, 1]   (B↔G2, G1↔R)
      Transpose → [0, 2, 1, 3]   (G1↔G2, B and R unchanged)
    Transpose is valid here because patches are square (patch_size × patch_size).
    """
    def transform(noisy, clean):
        _, h, w = noisy.shape
        # ── Random crop ──────────────────────────────────────────────
        top  = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        noisy = noisy[:, top:top+patch_size, left:left+patch_size]
        clean = clean[:, top:top+patch_size, left:left+patch_size]

        # ── H-flip: even col ↔ odd col  →  B↔G1, G2↔R ───────────────
        if random.random() > 0.5:
            noisy = TF.hflip(noisy)[[1, 0, 3, 2]]
            clean = TF.hflip(clean)[[1, 0, 3, 2]]

        # ── V-flip: even row ↔ odd row  →  B↔G2, G1↔R ───────────────
        if random.random() > 0.5:
            noisy = TF.vflip(noisy)[[2, 3, 0, 1]]
            clean = TF.vflip(clean)[[2, 3, 0, 1]]

        # ── Transpose: (r,c)→(c,r)  →  G1↔G2 (valid for square patches)
        if random.random() > 0.5:
            noisy = noisy.permute(0, 2, 1)[[0, 2, 1, 3]].contiguous()
            clean = clean.permute(0, 2, 1)[[0, 2, 1, 3]].contiguous()

        return noisy, clean
    return transform


def make_fullres_transform():
    """
    Phase 2 transform: no crop, dimension-preserving D4 subset only.

    Full-res images are 2040×1528 (non-square), so Transpose is skipped.
    H-flip and V-flip applied independently give all 4 dimension-preserving
    symmetries with equal probability (25% each):
      identity | H-flip | V-flip | 180° (= H+V)
    """
    def transform(noisy, clean):
        # ── H-flip ────────────────────────────────────────────────────
        if random.random() > 0.5:
            noisy = TF.hflip(noisy)[[1, 0, 3, 2]]
            clean = TF.hflip(clean)[[1, 0, 3, 2]]

        # ── V-flip ────────────────────────────────────────────────────
        if random.random() > 0.5:
            noisy = TF.vflip(noisy)[[2, 3, 0, 1]]
            clean = TF.vflip(clean)[[2, 3, 0, 1]]

        return noisy, clean
    return transform


##########################################################################
## Main
##########################################################################

if __name__ == "__main__":

    # ── A100 matmul / TF32 optimisations ──────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Reproducibility ───────────────────────────────────────────────
    seed = 21
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    def seed_worker(worker_id):
        """Distinct but deterministic seed per DataLoader worker."""
        random.seed(torch.initial_seed() % 2**32)
    worker_gen = torch.Generator()
    worker_gen.manual_seed(seed)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TOP-LEVEL FLAGS  — the only lines you change between runs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PHASE    = 1        # 1 = patch training  |  2 = full-res fine-tune
    USE_DUAL = True     # True = DualSNRDenoiser  |  False = SingleDenoiser (ablation)

    # Required for Phase 2: path to the best Phase 1 checkpoint.
    # Phase 2 loads this if no Phase 2 checkpoint exists yet.
    PHASE1_CHECKPOINT = (
        "models_p1_dual_Teacher_MobileHDR_YYYYMMDD_HHMM/phase1_best.pth"
    )
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Paths ─────────────────────────────────────────────────────────
    timestamp_str    = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag         = "dual" if USE_DUAL else "single"
    save_folder      = (f"models_p{PHASE}_{mode_tag}_Teacher_"
                        f"MobileHDR_{timestamp_str}/")
    dataset_dir      = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    create_folder(save_folder)

    # Checkpoint paths inside this run's save folder
    save_path      = save_folder + "latest.pth"
    best_save_path = save_folder + (f"phase{PHASE}_best.pth")
    last_path      = None          # tracks the most recently written checkpoint

    # ── Phase-specific hyperparameters ────────────────────────────────
    if PHASE == 1:
        # ── Patch training ────────────────────────────────────────────
        # 512×512 packed patches (= 1024×1024 sensor crop).
        # Bottleneck Restormer sees 64×64 = 4096 tokens — sufficient for
        # global noise statistics without processing the entire 49K-token
        # full-resolution feature map.
        # batch_sz=8 keeps peak VRAM ≈ 12 GB on A100-80 GB.
        PATCH_SIZE     = 512
        batch_sz       = 8
        num_patch      = 16    # 16 virtual repeats per image per epoch
        lr             = 1e-4  # higher LR safe with batch 8
        warmup_epochs  = 10    # linear ramp from 1% lr prevents early instability
        num_epochs     = 200
        eta_min        = 1e-6
        gamma          = 0.1   # moderate perceptual weight
        aux_weight     = 0.5 if USE_DUAL else 0.0
        grad_clip      = 1.0
        rollback_mult  = 3.0   # tighter than Phase 2; batch 8 has low variance
        LPIPS_CROP     = 256   # sub-crop from 512×512 patch for LPIPS

    else:
        # ── Full-resolution fine-tune ─────────────────────────────────
        # batch_sz=1 is the limit for 4080×3056 on A100-80 GB.
        # Very small LR: the model is already trained — just adapt the
        # boundary/global-context statistics to full-resolution inputs.
        PATCH_SIZE     = None  # full resolution
        batch_sz       = 1
        num_patch      = 1
        lr             = 5e-6
        warmup_epochs  = 0
        num_epochs     = 30
        eta_min        = 1e-7
        gamma          = 0.05  # near-zero perceptual; focus on pixel fidelity
        aux_weight     = 0.5 if USE_DUAL else 0.0
        grad_clip      = 0.5   # tighter clip for fine-tune stability
        rollback_mult  = 5.0
        LPIPS_CROP     = 256

    # Shared
    mu          = 5000   # µ-law tonemapping constant (HDR literature standard)
    start_epoch = 0
    best_psnr_mu = 0.0
    last_epoch_psnr_mu, last_epoch_loss = 0.0, 1e6

    model_kwargs = {
        "dim":                  32,
        "num_blocks":           [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads":                [1, 2, 4, 8],
        "se_reduction":         8,
    }

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project="hdr-dual-moe",
        name=f"run_p{PHASE}_{timestamp_str}_{mode_tag}",
        config={
            "phase":            PHASE,
            "mode":             mode_tag,
            "use_dual":         USE_DUAL,
            "patch_size":       PATCH_SIZE,
            "batch_size":       batch_sz,
            "num_patch":        num_patch,
            "lr":               lr,
            "warmup_epochs":    warmup_epochs,
            "num_epochs":       num_epochs,
            "eta_min":          eta_min,
            "mu":               mu,
            "gamma":            gamma,
            "aux_weight":       aux_weight,
            "grad_clip":        grad_clip,
            "rollback_mult":    rollback_mult,
            "lpips_crop":       LPIPS_CROP,
            "seed":             seed,
            "save_folder":      save_folder,
            **model_kwargs,
        },
    )

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\nPreparing model  [phase={PHASE}  mode={mode_tag}]")
    model = (DualSNRDenoiser(**model_kwargs) if USE_DUAL
             else SingleDenoiser(**model_kwargs)).to(device)

    loss_lpips = lpips.LPIPS(net="vgg").to(device)
    loss_lpips.eval()
    for p in loss_lpips.parameters():
        p.requires_grad_(False)

    scaler = (torch.amp.GradScaler("cuda")
              if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler())

    # ── Logging ───────────────────────────────────────────────────────
    logfile = save_folder + f"log_p{PHASE}_{timestamp_str}.txt"
    write_log(logfile, (
        f"{'='*60}\n"
        f"  HDR Teacher — Phase {PHASE}  [{mode_tag.upper()}]\n"
        f"{'='*60}\n"
        f"Timestamp  : {timestamp_str}\n"
        f"Dataset    : {dataset_dir}\n"
        f"Save folder: {save_folder}\n"
        f"Patch size : {PATCH_SIZE}  (None = full resolution)\n"
        f"Batch size : {batch_sz}   num_patch={num_patch}\n"
        f"LR         : {lr}  warmup={warmup_epochs}ep  "
        f"cosine→{eta_min}  total={num_epochs}ep\n"
        f"mu (tonemap): {mu}\n"
        f"Loss weights: gamma={gamma}  aux={aux_weight}\n"
        f"Grad clip  : {grad_clip}   rollback×{rollback_mult}\n"
        f"Model      : {model_kwargs}\n"
        f"{'='*60}\n\n"
    ), newfile=True)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, warmup_epochs, num_epochs, eta_min)

    # ── Checkpoint resume ─────────────────────────────────────────────
    def _load_checkpoint(path, load_scheduler=True):
        """Load model + optimiser (+ scheduler) from a checkpoint file."""
        global start_epoch, last_epoch_loss, best_psnr_mu
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Strip torch.compile prefix if the checkpoint was saved compiled
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if load_scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch     = ckpt.get("epoch", 0)
        last_epoch_loss = ckpt.get("loss",  1e6)
        best_psnr_mu    = ckpt.get("best_psnr_mu", 0.0)
        print(f"  Loaded: epoch={start_epoch}  loss={last_epoch_loss:.4f}"
              f"  best_psnr_mu={best_psnr_mu:.2f}")

    if os.path.exists(save_path):
        # Resume this phase's own interrupted run
        print(f"Resuming Phase {PHASE} from {save_path}")
        _load_checkpoint(save_path)

    elif PHASE == 2 and os.path.exists(PHASE1_CHECKPOINT):
        # Start Phase 2 from the best Phase 1 weights.
        # Do NOT load the Phase 1 scheduler state — Phase 2 uses a
        # different schedule and the state would be meaningless.
        print(f"Phase 2 init: loading Phase 1 weights from {PHASE1_CHECKPOINT}")
        _load_checkpoint(PHASE1_CHECKPOINT, load_scheduler=False)
        start_epoch  = 0     # Phase 2 epoch counter resets
        last_epoch_loss = 1e6
        best_psnr_mu    = 0.0

    elif PHASE == 2:
        raise FileNotFoundError(
            f"Phase 2 requires a Phase 1 checkpoint.\n"
            f"Set PHASE1_CHECKPOINT to a valid path.\n"
            f"Currently: {PHASE1_CHECKPOINT}"
        )
    else:
        print("Starting Phase 1 from scratch.")

    model.train()

    # ── Dataset ───────────────────────────────────────────────────────
    print("Creating dataset...")

    if PHASE == 1:
        transform  = make_patch_transform(PATCH_SIZE)
        collate_fn = None          # all patches are PATCH_SIZE×PATCH_SIZE → default collate
    else:
        transform  = make_fullres_transform()
        collate_fn = collate_pad_to_max

    dataset = MobileHDRDataset(
        base_dir=dataset_dir,
        split="train",
        transform=transform,
        num_patch=num_patch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=8 if PHASE == 1 else 4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=worker_gen,
    )

    n_batches = len(dataloader)
    print(f"  {len(dataset)} virtual samples  →  {n_batches} batches/epoch")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nPhase {PHASE} training started  ({num_epochs} epochs)\n")
    training_t0 = time.time()

    for epoch in range(start_epoch, num_epochs):

        # ── Inner loop with rollback ───────────────────────────────────
        # Re-runs the epoch if a loss spike is detected (e.g. outlier image).
        improved = False
        while not improved:
            running_loss    = 0.0
            running_psnr    = 0.0
            running_psnr_mu = 0.0
            t1 = time.time()

            for i, sample in enumerate(dataloader):
                x  = sample["x"].to(device, non_blocking=True)   # noisy
                y  = sample["y"].to(device, non_blocking=True)   # clean GT
                xm = sample["xm"].to(device, non_blocking=True)  # noisy (same as x)
                B, _, H, W = xm.shape

                # Build valid_mask: 1 on real pixels, 0 on reflect-padding.
                # Phase 1 patches are never padded → mask is all ones.
                # Phase 2 full-res images may be padded by collate_pad_to_max.
                if "orig_h" in sample:
                    orig_h = sample["orig_h"]
                    orig_w = sample["orig_w"]
                    valid_mask = torch.zeros(B, 1, H, W, device=device)
                    for b in range(B):
                        valid_mask[b, 0, :orig_h[b], :orig_w[b]] = 1.0
                else:
                    # Phase 1: uniform patch size, no padding
                    orig_h     = torch.full((B,), H)
                    orig_w     = torch.full((B,), W)
                    valid_mask = torch.ones(B, 1, H, W, device=device)

                with torch.no_grad():
                    snr_map = estimate_local_snr_map(xm, window_size=5)

                optimizer.zero_grad(set_to_none=True)

                # ── Forward + losses ─────────────────────────────────
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                    # 1. Forward pass
                    y_pred, y_low_snr, y_high_snr = model(xm, snr_map)

                    # 2. Main L1-µ loss on valid pixels
                    tm_pred = hdr_tonemap(y_pred, mu=mu)
                    tm_gt   = hdr_tonemap(y,      mu=mu)
                    l1_map      = torch.abs(tm_pred - tm_gt)
                    loss_l1_mu  = ((l1_map * valid_mask).sum()
                                   / (valid_mask.sum() * y_pred.shape[1] + 1e-6))

                    # 3. Auxiliary expert losses (dual mode only; zero-weighted for single)
                    # Encourages low-SNR expert to specialise on noisy pixels and
                    # high-SNR expert on clean pixels, rather than both converging
                    # to the same solution as the blended output.
                    snr_low   = (snr_map < 0.5).float()  * valid_mask
                    snr_high  = (snr_map >= 0.5).float() * valid_mask
                    loss_aux_low  = ((snr_low.expand_as(y_pred) *
                                     torch.abs(hdr_tonemap(y_low_snr,  mu=mu) - tm_gt)).sum()
                                    / (snr_low.sum()  * y_pred.shape[1] + 1e-6))
                    loss_aux_high = ((snr_high.expand_as(y_pred) *
                                     torch.abs(hdr_tonemap(y_high_snr, mu=mu) - tm_gt)).sum()
                                    / (snr_high.sum() * y_pred.shape[1] + 1e-6))

                # 4. Perceptual loss on a 256×256 random crop.
                # Always run outside autocast so LPIPS VGG stays in float32.
                # Channel order: R=ch3, G=ch1, B=ch0  (corrected BGGR→RGB mapping).
                min_h  = int(orig_h.min().item())
                min_w  = int(orig_w.min().item())
                crop_h = min(LPIPS_CROP, min_h)
                crop_w = min(LPIPS_CROP, min_w)
                top    = random.randint(0, min_h - crop_h)
                left   = random.randint(0, min_w - crop_w)

                def _lpips_crop(t):
                    """Tone-map, normalise to [-1, 1], crop to LPIPS size."""
                    rgb = torch.stack([t[:, 3, top:top+crop_h, left:left+crop_w],   # R
                                       t[:, 1, top:top+crop_h, left:left+crop_w],   # G
                                       t[:, 0, top:top+crop_h, left:left+crop_w]],  # B
                                      dim=1).float()
                    return hdr_tonemap(rgb.clamp(0, 1), mu=mu) * 2.0 - 1.0

                loss_perceptual = loss_lpips(_lpips_crop(y_pred),
                                             _lpips_crop(y)).mean()

                # 5. Total weighted loss
                ttl_loss = (loss_l1_mu
                            + gamma      * loss_perceptual
                            + aux_weight * (loss_aux_low + loss_aux_high))

                # ── Backward ─────────────────────────────────────────
                scaler.scale(ttl_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                # ── Running stats ────────────────────────────────────
                running_loss += ttl_loss.item()
                with torch.no_grad():
                    running_psnr    += batch_psnr_gpu(y_pred, y)
                    running_psnr_mu += batch_psnr_gpu(
                        hdr_tonemap(y_pred.float(), mu=mu),
                        hdr_tonemap(y.float(),      mu=mu))

                print_running_loss(running_loss, running_psnr_mu, batch_sz, i)

            # ── Epoch stats ───────────────────────────────────────────
            t2             = time.time()
            epoch_loss     = running_loss    / n_batches
            epoch_psnr     = running_psnr    / n_batches
            epoch_psnr_mu  = running_psnr_mu / n_batches
            epoch_msg = print_epoch_loss(epoch + 1, epoch_loss,
                                         epoch_psnr_mu, epoch_psnr,
                                         PHASE, t2 - t1)
            write_log(logfile, epoch_msg, end="")

            wandb.log({
                "epoch":               epoch + 1,
                "phase":               PHASE,
                "train/loss":          epoch_loss,
                "train/psnr_linear":   epoch_psnr,
                "train/psnr_mu":       epoch_psnr_mu,
                "train/lr":            optimizer.param_groups[0]["lr"],
                "train/loss_l1_mu":    loss_l1_mu.item(),
                "train/loss_percep":   loss_perceptual.item(),
                "train/loss_aux_low":  loss_aux_low.item(),
                "train/loss_aux_high": loss_aux_high.item(),
            })

            # ── Rollback check ────────────────────────────────────────
            # If the loss spikes by more than rollback_mult × last epoch's
            # loss, reload the previous checkpoint and retry the epoch.
            # This guards against outlier images or transient instability.
            improved = True
            if (last_epoch_psnr_mu > 10.0
                    and epoch_loss > rollback_mult * last_epoch_loss):
                print(f"\n  ⚠ Loss spike ({epoch_loss:.4f} > "
                      f"{rollback_mult}× {last_epoch_loss:.4f}) — reverting")
                improved = False
                if last_path and os.path.exists(last_path):
                    ckpt = torch.load(last_path, map_location=device,
                                      weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                last_epoch_psnr_mu = epoch_psnr_mu
                last_epoch_loss = epoch_loss

        # ── LR schedule step (accepted epochs only) ───────────────────
        # Advancing the scheduler on a rolled-back epoch would decay LR
        # even though the model weights were reverted.
        scheduler.step()

        # ── Save checkpoints ──────────────────────────────────────────
        save_dict = {
            "epoch":                epoch + 1,
            "phase":                PHASE,
            "mode":                 mode_tag,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss":                 epoch_loss,
            "best_psnr_mu":         best_psnr_mu,
        }
        torch.save(save_dict, save_path)
        last_path = save_path

        if epoch_psnr_mu > best_psnr_mu:
            best_psnr_mu = epoch_psnr_mu
            torch.save(save_dict, best_save_path)
            print(f"  ★ New best PSNR-µ: {best_psnr_mu:.2f} dB  → {best_save_path}")

    # ── Done ──────────────────────────────────────────────────────────
    total = time.time() - training_t0
    
    
    
    msg   = f"\nPhase {PHASE} finished in {total/3600:.2f} h  |  best PSNR-µ = {best_psnr_mu:.2f} dB"
    write_log(logfile, msg)
    print(msg)
    
    artifact = wandb.Artifact(f"hdr_p{PHASE}_{mode_tag}_final", type="model")
    artifact.add_file(best_save_path)
    wandb.log_artifact(artifact, aliases=["best"])

    wandb.finish()

    if PHASE == 1:
        print(f"\nNext step → set  PHASE = 2  and  PHASE1_CHECKPOINT = \"{best_save_path}\"")

    wandb.finish()