import os
import time
import lpips
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import wandb

from HDR_model_hybrid_Teacher import TransUNet_Teacher_HDR as HDR_model
from HDR_Mobile_dataset import MobileHDRDataset
# from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
# Reroute W&B to use scratch space instead of the home directory
os.environ["WANDB_CACHE_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
os.environ["WANDB_DATA_DIR"] = "/scratch/gilbreth/chen4848/wandb_data"
os.environ["WANDB_DIR"] = "/scratch/gilbreth/chen4848/projects/HDR-denoise"
##########################################################################
## Model wrappers
##########################################################################

class DualSNRDenoiser(nn.Module):
    """Two expert denoisers blended by a pixel-wise SNR map."""
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser_low_snr  = HDR_model(**kwargs)
        self.denoiser_high_snr = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        """
        x:       [B, 4, H, W] packed Bayer
        snr_map: [B, 1, H, W] normalized [0, 1]
        Returns: blended, out_low, out_high
        """
        out_low  = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)
        blended  = (1.0 - snr_map) * out_low + snr_map * out_high
        return blended, out_low, out_high


class SingleDenoiser(nn.Module):
    """Baseline: one denoiser, no SNR routing.
    Returns the same 3-tuple interface as DualSNRDenoiser so the
    training loop requires zero changes."""
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        """snr_map is accepted but ignored."""
        out = self.denoiser(x)
        return out, out, out   # same tensor for all three slots


##########################################################################
## Logging utilities
##########################################################################
def print_running_loss(running_loss, running_psnr, batch_sz, i, print_every=10):
    psnr_str = "\tPSNR-µ = %.2f" % (running_psnr / (i + 1)) if running_psnr else ""
    if i % print_every == (print_every - 1):
        print("\r\tLoss = %15.4f\t%s" % (running_loss / (batch_sz * (i + 1)), psnr_str), end="")

def print_epoch_loss(epoch_num, avg_loss, avg_psnr, time_spent=None):
    time_spent = "Unknown" if time_spent is None else "%f" % time_spent
    psnr_str   = "\tPSNR-µ = %.2f" % avg_psnr if avg_psnr else ""
    msg = "\rEpoch #%d: Loss = %12.4f \t%s\t(Time: %s)" % (epoch_num, avg_loss, psnr_str, time_spent)
    print(msg)
    return msg

def write_log(filename, s, newfile=False, end="\n"):
    if newfile:
        open(filename, "w").close()
    with open(filename, "a") as f:
        f.write("%s%s" % (s, end))

def create_folder(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        print("Error: Creating directory. " + directory)

##########################################################################
## GPU utilities
##########################################################################
def hdr_tonemap(hdr_image, nbits=14):
    mu = 2 ** nbits - 1
    return torch.log1p(mu * hdr_image) / torch.log1p(torch.tensor(mu, device=hdr_image.device, dtype=hdr_image.dtype))
    
def batch_psnr_gpu(img, gt, data_range=1.0):
    with torch.no_grad():
        mse   = torch.mean((img.detach() - gt.detach()) ** 2, dim=[1, 2, 3])
        psnrs = torch.where(
            mse == 0,
            torch.tensor(100.0, device=img.device),
            10.0 * torch.log10((data_range ** 2) / mse),
        )
        return torch.mean(psnrs).item()

def estimate_local_snr_map(x, window_size=5, eps=1e-5):
    """Returns [B, 1, H, W] normalized SNR map."""
    unbatched = x.dim() == 3
    if unbatched:
        x = x.unsqueeze(0)
    pad           = window_size // 2
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



def unpack_packed_bayer_gpu(packed):
    """[B, 4, H, W] -> [B, 1, H*2, W*2]  (BGGR layout)"""
    B, _, h, w = packed.shape
    mosaic = torch.zeros((B, 1, h * 2, w * 2), device=packed.device, dtype=packed.dtype)
    mosaic[:, 0, 0::2, 0::2] = packed[:, 0]   # B
    mosaic[:, 0, 0::2, 1::2] = packed[:, 1]   # G1
    mosaic[:, 0, 1::2, 0::2] = packed[:, 2]   # G2
    mosaic[:, 0, 1::2, 1::2] = packed[:, 3]   # R
    return mosaic

def collate_pad_to_max(batch):
    """Pads variable-size full-resolution images to the largest H×W in the
    batch (rounded up to multiple of 4) so PyTorch can stack them.
    Also stores original sizes so padded regions can be masked in the loss."""
    max_h = max(s["x"].shape[1] for s in batch)
    max_w = max(s["x"].shape[2] for s in batch)
    max_h = ((max_h + 7) // 8) * 8   # round up to multiple of 8
    max_w = ((max_w + 7) // 8) * 8

    def pad(t):
        _, h, w = t.shape
        return F.pad(t, (0, max_w - w, 0, max_h - h), mode='reflect')

    return {
        "x":      torch.stack([pad(s["x"])  for s in batch]),
        "xm":     torch.stack([pad(s["xm"]) for s in batch]),
        "y":      torch.stack([pad(s["y"])  for s in batch]),
        "orig_h": torch.tensor([s["x"].shape[1] for s in batch]),
        "orig_w": torch.tensor([s["x"].shape[2] for s in batch]),
    }

##########################################################################
## Main
##########################################################################
if __name__ == "__main__":
    # ── A100 matmul optimizations ─────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 21
    torch.manual_seed(seed)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ABLATION FLAG — only line you need to change between the two runs
    #   True  → DualSNRDenoiser  (two experts + SNR blending)
    #   False → SingleDenoiser   (one model, identical params, baseline)
    USE_DUAL = True
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Paths ─────────────────────────────────────────────────────────
    timestamp_str     = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag          = "dual" if USE_DUAL else "single"
    save_folder       = f"models_MoE_{mode_tag}_Teacher_MobileHDR_{timestamp_str}/"
    pretrained_path   = "models_MoE_dual_Teacher_MobileHDR_20260522_1033/preexpand_hdr.pth"  
    dataset_directory = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    create_folder(save_folder)

    save_path_preexpand      = save_folder + "preexpand_hdr.pth"
    best_save_path_preexpand = save_folder + "preexpand_hdr_best.pth"
    save_path                = save_folder + "hdr.pth"
    best_save_path           = save_folder + "hdr_best.pth"
    last_path                = None

    # ── Hyperparameters ───────────────────────────────────────────────
    nbits                    = 20
    nbits_data               = 10
    batch_sz                 = 1    # full 4080x3056 images; 2 is safe on A100 80GB
    patch_sz                 = None  # unused — no crop in full-res mode
    num_patch                = 1     # each image used once per epoch; no virtual repeats needed
    read_noise               = 0
    regen_crops_every_epoch  = 10
    regen_noise_every_epoch  = 5

    # LR scaled down for batch_sz=1 vs the original batch_sz=32:
    # rule-of-thumb linear scaling -> 1e-4 / 32 * 1 ≈ 3e-6 (aggressive lower bound)
    # sqrt scaling (more conservative) -> 1e-4 / sqrt(32) ≈ 1.8e-5  <-- used here
    lr                = 1.8e-5
    optimizer_choice  = "Adam"
    num_epochs        = 100
    start_epoch       = 0
    epoch_expand_mode = 999

    gamma      = 0.1   # raised from 0.05 to compensate for patch-crop perceptual signal
    # aux_weight is only meaningful for dual mode; zero it out for single
    # so the loss functions are directly comparable (both just l1_mu + perceptual)
    aux_weight = 0.5 if USE_DUAL else 0.0

    best_running_psnr            = 10.0
    last_epoch_psnr, last_epoch_loss = 0.0, 1e6

    # LPIPS crop size: VGG was trained on small patches; running it on a
    # multi-megapixel image produces unreliable / oscillating gradients.
    # A 256×256 random crop gives a stable signal without GBTF overhead.
    LPIPS_CROP = 256

    model_kwargs = {
        "dim": 32,
        "num_blocks": [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8],
    }

    # ── W&B ───────────────────────────────────────────────────────────
    # Both runs log to the same project so W&B overlays their curves automatically
    wandb.init(
        project="hdr-dual-moe",
        name=f"run_{timestamp_str}_{mode_tag}_Teacher",
        config={
            "mode":                mode_tag,
            "use_dual":            USE_DUAL,
            "learning_rate":       lr,
            "epochs":              num_epochs,
            "batch_size":          batch_sz,
            "patch_size":          patch_sz,
            "num_patches_per_img": num_patch,
            "optimizer":           optimizer_choice,
            "expand_mode_epoch":   epoch_expand_mode,
            "nbits":               nbits,
            "nbits_data":          nbits_data,
            "read_noise":          read_noise,
            "seed":                seed,
            "gamma":               gamma,
            "aux_weight":          aux_weight,
            "lpips_crop_size":     LPIPS_CROP,
            "grad_clip_norm":      1.0,
            "rollback_threshold":  5.0,
            "dataset_path":        dataset_directory,
            "pretrained_path":     pretrained_path,
            "save_folder":         save_folder,
            **model_kwargs,
        },
    )

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Preparing model  [mode = {mode_tag}]")
    model = DualSNRDenoiser(**model_kwargs).to(device) if USE_DUAL else SingleDenoiser(**model_kwargs).to(device)

    # if hasattr(torch, "compile"):
    #     try:
    #         print("Compiling model via torch.compile ...")
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print(f"torch.compile skipped: {e}")

    loss_lpips = lpips.LPIPS(net="vgg").to(device)

    scaler = (
        torch.amp.GradScaler("cuda")
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler()
    )

    # ── Logging ───────────────────────────────────────────────────────
    logfile = save_folder + "log_%s.txt" % timestamp_str
    log_header = (
        f"{'='*58}\n"
        f"  HDR TEACHER ABLATION — mode={mode_tag.upper()}\n"
        f"{'='*58}\n"
        f"Timestamp:    {timestamp_str}\n"
        f"Dataset:      {dataset_directory}\n"
        f"Save folder:  {save_folder}\n"
        f"Batch size:   {batch_sz}\n"
        f"Patch size:   {patch_sz}  (x{num_patch} per image)\n"
        f"LR:           {lr}  (CosineAnnealingLR -> eta_min=1e-6)\n"
        f"Epochs:       {num_epochs}\n"
        f"Loss weights: gamma_percep={gamma}, aux={aux_weight}\n"
        f"Model:        {model_kwargs}\n"
        f"{'='*58}\n\n"
    )
    write_log(logfile, log_header, newfile=True)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = (
        torch.optim.RMSprop(model.parameters(), lr=lr)
        if optimizer_choice == "RMSprop"
        else torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # ── Checkpoint resume ─────────────────────────────────────────────
    def _load_checkpoint(path):
        global start_epoch, last_epoch_loss
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Old checkpoint with no scheduler state: fast-forward to match epoch
            for _ in range(ckpt.get("epoch", 0)):
                scheduler.step()
        start_epoch     = ckpt.get("epoch", 0)
        last_epoch_loss = ckpt.get("loss", 1e6)

    if os.path.exists(save_path) or os.path.exists(best_save_path):
        target = save_path if os.path.exists(save_path) else best_save_path
        _load_checkpoint(target)
        print(f"Resumed (expanded) from epoch {start_epoch}")

    elif os.path.exists(save_path_preexpand) or os.path.exists(best_save_path_preexpand):
        target = save_path_preexpand if os.path.exists(save_path_preexpand) else best_save_path_preexpand
        _load_checkpoint(target)
        print(f"Resumed (pre-expand) from epoch {start_epoch}")

    elif pretrained_path and os.path.exists(pretrained_path):
        ckpt               = torch.load(pretrained_path)
        state_dict_to_load = ckpt.get("student_state_dict", ckpt.get("model_state_dict"))
        if USE_DUAL:
            model.denoiser_low_snr.load_state_dict(state_dict_to_load,  strict=False)
            model.denoiser_high_snr.load_state_dict(state_dict_to_load, strict=False)
        else:
            model.denoiser.load_state_dict(state_dict_to_load, strict=False)
        print("Initialized from pretrained weights.")
    else:
        print("Starting from scratch.")

    model.train()

    # ── Dataset ───────────────────────────────────────────────────────
    print("Creating dataset")

    def train_transforms(noisy, clean):
        # No crop — pass the full image so Restormer sees global context.
        # collate_pad_to_max handles variable sizes at batch assembly time.
        if random.random() > 0.5:          # H-flip: B<->G1, G2<->R
            noisy = TF.hflip(noisy)[[1, 0, 3, 2]]
            clean = TF.hflip(clean)[[1, 0, 3, 2]]
        if random.random() > 0.5:          # V-flip: B<->G2, G1<->R
            noisy = TF.vflip(noisy)[[2, 3, 0, 1]]
            clean = TF.vflip(clean)[[2, 3, 0, 1]]
        return noisy, clean

    dataset = MobileHDRDataset(
        base_dir=dataset_directory,
        split="train",
        transform=train_transforms,
        num_patch=num_patch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=4,               # reduced: full-res images are heavy to transfer
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_pad_to_max,   # handles variable spatial sizes
    )

    # gbtf_differentiable = DifferentiableGBTF_BGGR().to(device)

    # ── Training loop ─────────────────────────────────────────────────
    print("Training started")
    training_t0 = time.time()

    for epoch in range(start_epoch, num_epochs):
        if hasattr(dataset, "do_expand") and epoch >= epoch_expand_mode:
            dataset.do_expand = True
        if regen_crops_every_epoch and epoch % regen_crops_every_epoch == 0 and epoch:
            if hasattr(dataset, "regen_crops"): dataset.regen_crops()
        if regen_noise_every_epoch and epoch % regen_noise_every_epoch == 0 and epoch:
            if hasattr(dataset, "regen_noise"): dataset.regen_noise()

        improved = False
        while not improved:
            running_loss = 0.0
            running_psnr = 0.0
            t1 = time.time()

            for i, sample in enumerate(dataloader):
                x      = sample["x"].to(device, non_blocking=True)
                y      = sample["y"].to(device, non_blocking=True)
                xm     = sample["xm"].to(device, non_blocking=True)
                orig_h = sample["orig_h"]   # [B] CPU tensors, used for mask only
                orig_w = sample["orig_w"]

                # Build [B, 1, H, W] binary mask: 1 inside valid pixels, 0 on pad.
                # Ensures padded reflect-borders never contribute to any loss.
                B, _, H, W = xm.shape
                valid_mask = torch.zeros(B, 1, H, W, device=device)
                for b in range(B):
                    valid_mask[b, 0, :orig_h[b], :orig_w[b]] = 1.0

                with torch.no_grad():
                    snr_map = estimate_local_snr_map(xm, window_size=5)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 1. Forward — identical call for both modes
                    y_pred, y_low_snr, y_high_snr = model(xm, snr_map)
                    # gt_tm = hdr_tonemap(y, nbits)

                    # 2. Main reconstruction loss — valid pixels only
                    l1_map     = torch.abs(hdr_tonemap(y_pred) - hdr_tonemap(y))
                    loss_l1_mu = (l1_map * valid_mask).sum() / (valid_mask.sum() * y_pred.shape[1] + 1e-6)

                    # 3. Auxiliary expert losses
                    # Unified to hdr_tonemap space (same as main loss) so both
                    # objectives pull weights in a consistent direction.
                    # Single mode: y_low_snr == y_high_snr == y_pred AND aux_weight==0.
                    snr_low   = (snr_map < 0.5).float()  * valid_mask
                    snr_high  = (snr_map >= 0.5).float() * valid_mask
                    mask_low  = snr_low.expand_as(y_pred)
                    mask_high = snr_high.expand_as(y_pred)
                    se_low    = torch.abs(hdr_tonemap(y_low_snr)  - hdr_tonemap(y))
                    se_high   = torch.abs(hdr_tonemap(y_high_snr) - hdr_tonemap(y))
                    loss_aux_low  = (mask_low  * se_low).sum()  / (mask_low.sum()  + 1e-6)
                    loss_aux_high = (mask_high * se_high).sum() / (mask_high.sum() + 1e-6)

                # 4. Perceptual loss — random 256×256 crop on Bayer channels B/G1/R
                # Avoids feeding VGG a multi-megapixel tensor (out of its training
                # distribution → wild gradient oscillation) and keeps VRAM usage flat.
                min_h = orig_h.min().item()
                min_w = orig_w.min().item()
                crop_h = min(LPIPS_CROP, min_h)
                crop_w = min(LPIPS_CROP, min_w)
                top  = random.randint(0, min_h - crop_h)
                left = random.randint(0, min_w - crop_w)

                pred_crop = torch.stack([
                    y_pred[:, 0, top:top+crop_h, left:left+crop_w],
                    y_pred[:, 1, top:top+crop_h, left:left+crop_w],
                    y_pred[:, 3, top:top+crop_h, left:left+crop_w],
                ], dim=1).float()
                gt_crop = torch.stack([
                    y[:, 0, top:top+crop_h, left:left+crop_w],
                    y[:, 1, top:top+crop_h, left:left+crop_w],
                    y[:, 3, top:top+crop_h, left:left+crop_w],
                ], dim=1).float()
                pred_crop = hdr_tonemap(pred_crop.clamp(0, 1)) * 2.0 - 1.0
                gt_crop   = hdr_tonemap(gt_crop.clamp(0, 1))   * 2.0 - 1.0
                loss_perceptual = loss_lpips(pred_crop, gt_crop).mean()

                # 5. Total loss
                ttl_loss = (
                    loss_l1_mu
                    + gamma      * loss_perceptual
                    + aux_weight * (loss_aux_low + loss_aux_high)
                )

                scaler.scale(ttl_loss).backward()
                # Gradient clipping: full-res single-image batches can produce
                # large gradient norms from outlier images; clip before stepping.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += ttl_loss.item()
                running_psnr += batch_psnr_gpu(y_pred, y)
                print_running_loss(running_loss, running_psnr, batch_sz, i)

            t2          = time.time()
            num_batches = len(dataloader)
            this_epoch_loss = running_loss / num_batches
            this_epoch_psnr = running_psnr / num_batches
            epoch_msg = print_epoch_loss(epoch + 1, this_epoch_loss, this_epoch_psnr, t2 - t1)
            write_log(logfile, epoch_msg, end="")

            wandb.log({
                "epoch":               epoch + 1,
                "train/loss":          this_epoch_loss,
                "train/psnr-mu":          this_epoch_psnr,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/loss_l1_mu":    loss_l1_mu.item(),
                "train/loss_percep":   loss_perceptual.item(),
                "train/loss_aux_low":  loss_aux_low.item(),
                "train/loss_aux_high": loss_aux_high.item(),
            })

            improved = True

            bad_loss_spike = (
                (epoch < epoch_expand_mode or epoch >= epoch_expand_mode + 10)
                and last_epoch_psnr > 10
                and this_epoch_loss > 5 * last_epoch_loss  # loosened from 2x: batch_sz=1 has high variance
            ) or (
                epoch_expand_mode <= epoch < epoch_expand_mode + 10
                and this_epoch_loss > 1000
            )

            if bad_loss_spike:
                print(last_epoch_psnr, this_epoch_psnr, "loss spike — reverting")
                improved = False
                if hasattr(dataset, "regen_crops"): dataset.regen_crops()
                if hasattr(dataset, "regen_noise"):  dataset.regen_noise()
                if last_path and os.path.exists(last_path):
                    ckpt = torch.load(last_path)
                    model.load_state_dict(ckpt["model_state_dict"])
            else:
                last_epoch_psnr = this_epoch_psnr
                last_epoch_loss = this_epoch_loss

        scheduler.step()

        if improved:
            sp  = save_path      if epoch >= epoch_expand_mode else save_path_preexpand
            bsp = best_save_path if epoch >= epoch_expand_mode else best_save_path_preexpand
            last_path = sp

            save_dict = {
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss":                 this_epoch_loss,
                "mode":                 mode_tag,
            }
            torch.save(save_dict, sp)

            if this_epoch_psnr > best_running_psnr:
                best_running_psnr = this_epoch_psnr
                torch.save(save_dict, bsp)

                artifact_name = "model_expanded" if epoch >= epoch_expand_mode else "model_preexpand"
                artifact = wandb.Artifact(f"hdr_{mode_tag}_{artifact_name}", type="model")
                artifact.add_file(bsp)
                wandb.log_artifact(artifact, aliases=["best"])

    training_t1 = time.time()
    write_log(logfile, "\nTotal training time = %.2f s" % (training_t1 - training_t0))
    print("Training finished")
    wandb.finish()