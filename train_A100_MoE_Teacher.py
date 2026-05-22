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

##########################################################################
## Dual SNR Denoiser Wrapper
##########################################################################
class DualSNRDenoiser(nn.Module):
    def __init__(self, **kwargs):
        super(DualSNRDenoiser, self).__init__()
        self.denoiser_low_snr  = HDR_model(**kwargs)
        self.denoiser_high_snr = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        """
        x:       [B, 4, H, W] packed Bayer noisy input
        snr_map: [B, 1, H, W] normalized [0, 1]
        """
        out_low  = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)
        blended_out = (1.0 - snr_map) * out_low + snr_map * out_high
        return blended_out, out_low, out_high

##########################################################################
## Logging utilities
##########################################################################
def print_running_loss(running_loss, running_psnr, batch_sz, i, print_every=10):
    psnr_str = "\tPSNR-µ = %.2f" % (running_psnr / (i + 1)) if running_psnr else ""
    if i % print_every == (print_every - 1):
        print("\r\tLoss = %15.4f\t%s" % (running_loss / (i + 1), psnr_str), end="")

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
def hdr_tonemap(hdr_image, nbits=20):
    mu = 2 ** nbits - 1
    return torch.log10(1.0 + mu * hdr_image) / torch.log10(
        torch.tensor(1.0 + mu, device=hdr_image.device, dtype=hdr_image.dtype)
    )

def batch_psnr_gpu(img, gt, data_range=1.0):
    with torch.no_grad():
        mse   = torch.mean((img.detach() - gt.detach()) ** 2, dim=[1, 2, 3])
        psnrs = torch.where(
            mse == 0,
            torch.tensor(100.0, device=img.device),
            10.0 * torch.log10((data_range ** 2) / mse)
        )
        return torch.mean(psnrs).item()
    
def batch_psnr_mu_gpu(img, gt, mu=5000):
    """
    PSNR computed in µ-law tone-mapped domain (µ=5000).
    More meaningful than linear PSNR for HDR — weights shadows equally to highlights.
    Higher is better; a well-trained HDR denoiser should reach 35+ dB.
    """
    with torch.no_grad():
        img_mu = mu_law_tonemap(img.detach().clamp(0, 1), mu=mu)
        gt_mu  = mu_law_tonemap(gt.detach().clamp(0, 1),  mu=mu)
        mse    = torch.mean((img_mu - gt_mu) ** 2, dim=[1, 2, 3])
        psnrs  = torch.where(
            mse == 0,
            torch.tensor(100.0, device=img.device),
            10.0 * torch.log10(1.0 / mse)   # data_range=1.0 after µ-law
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

def mu_law_tonemap(x, mu=5000):
    return torch.log1p(mu * x) / torch.log1p(
        torch.tensor(mu, dtype=x.dtype, device=x.device)
    )

# def unpack_packed_bayer_gpu(packed):
#     """[B, 4, H, W] -> [B, 1, H*2, W*2]  (BGGR layout)"""
#     B, _, h, w = packed.shape
#     mosaic = torch.zeros((B, 1, h * 2, w * 2), device=packed.device, dtype=packed.dtype)
#     mosaic[:, 0, 0::2, 0::2] = packed[:, 0]   # B
#     mosaic[:, 0, 0::2, 1::2] = packed[:, 1]   # G1
#     mosaic[:, 0, 1::2, 0::2] = packed[:, 2]   # G2
#     mosaic[:, 0, 1::2, 1::2] = packed[:, 3]   # R
#     return mosaic

##########################################################################
## Main
##########################################################################
if __name__ == "__main__":
    # ── A100 matmul optimizations ──────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 21
    torch.manual_seed(seed)

    # ── Paths ──────────────────────────────────────────────────────────
    timestamp_str      = datetime.now().strftime("%Y%m%d_%H%M")
    save_folder        = "models_MoE_Teacher_MobileHDR_%s/" % timestamp_str
    pretrained_path    = None
    dataset_directory  = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    create_folder(save_folder)

    save_path_preexpand      = save_folder + "preexpand_hdr.pth"
    best_save_path_preexpand = save_folder + "preexpand_hdr_best.pth"
    save_path                = save_folder + "hdr.pth"
    best_save_path           = save_folder + "hdr_best.pth"
    last_path                = None

    # ── Hyperparameters ────────────────────────────────────────────────
    nbits                  = 20
    nbits_data             = 10
    batch_sz               = 32
    patch_sz               = 256    
    num_patch              = 32     # virtual repeats per image per epoch
    read_noise             = 0
    regen_crops_every_epoch = 10
    regen_noise_every_epoch = 5

    lr                  = 2e-4     # FIX: raised from 1e-4; cosine decay handles the rest
    optimizer_choice    = "Adam"
    num_epochs          = 100
    start_epoch         = 0
    epoch_expand_mode   = 999

    # Loss weights
    gamma      = 0.05   # perceptual loss weight  (was 0.01 — raised slightly for Teacher)
    aux_weight = 0.5    # FIX: was 1.0, dominating the main reconstruction loss

    best_running_psnr          = 10.0
    last_epoch_psnr, last_epoch_loss = 0.0, 1e6

    model_kwargs = {
        "dim": 32,
        "num_blocks": [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8],
    }

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project="hdr-dual-moe",
        name=f"run_{timestamp_str}_MobileHDR_Optimized_MoE_100_epochs",
        config={
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
            "model_architecture":  "DualSNR_Teacher_HDR",
            "dataset_path":        dataset_directory,
            "pretrained_path":     pretrained_path,
            "save_folder":         save_folder,
            **model_kwargs,
        },
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("Preparing dual model")
    model = DualSNRDenoiser(**model_kwargs).to(device)

    if hasattr(torch, "compile"):
        try:
            print("Compiling model via torch.compile …")
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    # FIX: instantiate LPIPS only once
    loss_lpips = lpips.LPIPS(net="vgg").to(device)

    scaler = (
        torch.amp.GradScaler("cuda")
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler()
    )

    # ── Logging ───────────────────────────────────────────────────────
    logfile    = save_folder + "log_%s.txt" % timestamp_str
    log_header = (
        f"{'='*58}\n"
        f"  DUAL MOE TEACHER — A100 TRAINING\n"
        f"{'='*58}\n"
        f"Timestamp:       {timestamp_str}\n"
        f"Dataset:         {dataset_directory}\n"
        f"Save folder:     {save_folder}\n"
        f"Batch size:      {batch_sz}\n"
        f"Patch size:      {patch_sz}  (×{num_patch} per image)\n"
        f"LR:              {lr}  (CosineAnnealingLR → η_min=1e-6)\n"
        f"Epochs:          {num_epochs}\n"
        f"Loss weights:    γ_percep={gamma}, aux={aux_weight}\n"
        f"nbits:           {nbits} / {nbits_data}\n"
        f"Model:           {model_kwargs}\n"
        f"{'='*58}\n\n"
    )
    write_log(logfile, log_header, newfile=True)

    # ── Optimizer + LR scheduler ──────────────────────────────────────
    if optimizer_choice == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # FIX: replace the broken manual step-table with CosineAnnealingLR.
    # Smoothly decays lr → eta_min over num_epochs; no sudden jumps or 
    # accidental mid-training increases.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # ── Checkpoint resume ─────────────────────────────────────────────
    if os.path.exists(save_path) or os.path.exists(best_save_path):
        load_target = save_path if os.path.exists(save_path) else best_save_path
        checkpoint  = torch.load(load_target)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch     = checkpoint.get("epoch", 0)
        last_epoch_loss = checkpoint.get("loss", 1e6)
        print(f"Resumed (expanded mode) from epoch {start_epoch}")

    elif os.path.exists(save_path_preexpand) or os.path.exists(best_save_path_preexpand):
        load_target = save_path_preexpand if os.path.exists(save_path_preexpand) else best_save_path_preexpand
        checkpoint  = torch.load(load_target)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch     = checkpoint.get("epoch", 0)
        last_epoch_loss = checkpoint.get("loss", 1e6)
        print(f"Resumed (pre-expand mode) from epoch {start_epoch}")

    elif pretrained_path and os.path.exists(pretrained_path):
        checkpoint         = torch.load(pretrained_path)
        state_dict_to_load = checkpoint.get("student_state_dict", checkpoint.get("model_state_dict"))
        model.denoiser_low_snr.load_state_dict(state_dict_to_load,  strict=False)
        model.denoiser_high_snr.load_state_dict(state_dict_to_load, strict=False)
        print("Initialized both experts from pretrained weights.")
    else:
        print("Starting from scratch.")

    model.train()

    # ── Dataset ───────────────────────────────────────────────────────
    print("Creating dataset")

    def train_transforms(noisy, clean):
        # FIX: use the top-level patch_sz instead of a hardcoded local literal
        _, h, w = noisy.shape
        if h > patch_sz and w > patch_sz:
            top  = random.randint(0, h - patch_sz)
            left = random.randint(0, w - patch_sz)
            noisy = TF.crop(noisy, top, left, patch_sz, patch_sz)
            clean = TF.crop(clean, top, left, patch_sz, patch_sz)

        # H-flip: B↔G1 (0↔1), G2↔R (2↔3)
        if random.random() > 0.5:
            noisy = TF.hflip(noisy)[[1, 0, 3, 2]]
            clean = TF.hflip(clean)[[1, 0, 3, 2]]

        # V-flip: B↔G2 (0↔2), G1↔R (1↔3)
        if random.random() > 0.5:
            noisy = TF.vflip(noisy)[[2, 3, 0, 1]]
            clean = TF.vflip(clean)[[2, 3, 0, 1]]

        return noisy, clean

    # FIX: pass num_patch so dataset length matches the config value
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
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,   # avoids worker respawn overhead each epoch
    )

    # gbtf_differentiable = DifferentiableGBTF_BGGR().to(device)

    # ── Training loop ─────────────────────────────────────────────────
    print("Training started")
    training_t0 = time.time()

    for epoch in range(start_epoch, num_epochs):
        if hasattr(dataset, "do_expand") and epoch >= epoch_expand_mode:
            dataset.do_expand = True
        if regen_crops_every_epoch and epoch % regen_crops_every_epoch == 0 and epoch:
            if hasattr(dataset, "regen_crops"):
                dataset.regen_crops()
        if regen_noise_every_epoch and epoch % regen_noise_every_epoch == 0 and epoch:
            if hasattr(dataset, "regen_noise"):
                dataset.regen_noise()

        improved     = False
        while not improved:
            running_loss = 0.0
            running_psnr = 0.0
            t1 = time.time()

            for i, sample in enumerate(dataloader):
                # x   = sample["x"].to(device, non_blocking=True)
                xm  = sample["xm"].to(device, non_blocking=True)
                y   = sample["y"].to(device, non_blocking=True)

                with torch.no_grad():
                    snr_map = estimate_local_snr_map(xm, window_size=5)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 1. Forward
                    y_pred, y_low_snr, y_high_snr = model(xm, snr_map)
                    # gt_tm = hdr_tonemap(y, nbits)

                    # 2. Main reconstruction loss (mu-law domain)
                    noise_weight = (1.0 - snr_map)  # [B, 1, H, W], high where noisy
                    loss_l1_mu = torch.mean(noise_weight * torch.abs(mu_law_tonemap(y_pred) - mu_law_tonemap(y)))

                    # 3. Auxiliary MoE expert losses
                    # Expand snr_map to [B, 4, H, W] so masking is explicit
                    mask_low  = (snr_map < 0.5).expand_as(y_pred).float()
                    mask_high = (snr_map >= 0.5).expand_as(y_pred).float()

                    # Fix — use mu_law for both so all losses operate in the same domain
                    ae_low  = torch.abs(mu_law_tonemap(y_low_snr)  - mu_law_tonemap(y))
                    ae_high = torch.abs(mu_law_tonemap(y_high_snr) - mu_law_tonemap(y))

                    loss_aux_low  = torch.sum(mask_low  * ae_low)  / (torch.sum(mask_low)  + 1e-6)
                    loss_aux_high = torch.sum(mask_high * ae_high) / (torch.sum(mask_high) + 1e-6)

                # 4. Perceptual loss via differentiable GBTF demosaicing
                pred_tm  = torch.stack([y_pred[:,0], y_pred[:,1], y_pred[:,3]], dim=1).detach().float()
                gt_tm_p  = torch.stack([y[:,0],      y[:,1],      y[:,3]     ], dim=1).float()
                pred_tm  = mu_law_tonemap(pred_tm.clamp(0,1)) * 2.0 - 1.0
                gt_tm_p  = mu_law_tonemap(gt_tm_p.clamp(0,1)) * 2.0 - 1.0
                loss_perceptual = loss_lpips(pred_tm, gt_tm_p).mean()

                # 5. Total loss
                ttl_loss = (
                    loss_l1_mu
                    + gamma      * loss_perceptual
                    + aux_weight * (loss_aux_low + loss_aux_high)
                )

                scaler.scale(ttl_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += ttl_loss.item()
                running_psnr += batch_psnr_mu_gpu(y_pred, y)
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
                "train/psnr_mu":       this_epoch_psnr,
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
                and this_epoch_loss > 2 * last_epoch_loss
            ) or (
                epoch_expand_mode <= epoch < epoch_expand_mode + 10
                and this_epoch_loss > 1000
            )

            if bad_loss_spike:
                print(last_epoch_psnr, this_epoch_psnr, "loss spike — reverting")
                improved = False
                if hasattr(dataset, "regen_crops"): dataset.regen_crops()
                if hasattr(dataset, "regen_noise"):  dataset.regen_noise()
                # Fix — restore all three, not just the model
                if last_path and os.path.exists(last_path):
                    checkpoint = torch.load(last_path)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])   # ADD
                    if "scheduler_state_dict" in checkpoint:                        # ADD
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                last_epoch_psnr = this_epoch_psnr
                last_epoch_loss = this_epoch_loss

        # FIX: step the scheduler every epoch (was never called before)
        scheduler.step()

        if improved:
            sp   = save_path           if epoch >= epoch_expand_mode else save_path_preexpand
            bsp  = best_save_path      if epoch >= epoch_expand_mode else best_save_path_preexpand
            last_path = sp

            save_dict = {
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),   # FIX: persist scheduler
                "loss":                 this_epoch_loss,
            }
            torch.save(save_dict, sp)

            if this_epoch_psnr > best_running_psnr:
                best_running_psnr = this_epoch_psnr
                torch.save(save_dict, bsp)

                artifact_name = "model_expanded" if epoch >= epoch_expand_mode else "model_preexpand"
                artifact = wandb.Artifact(f"dual_hdr_{artifact_name}", type="model")
                artifact.add_file(bsp)
                wandb.log_artifact(artifact, aliases=[f"epoch_{epoch + 1}", "best"])

    training_t1 = time.time()
    write_log(logfile, "\nTotal training time = %.2f s" % (training_t1 - training_t0))
    print("Training finished")
    wandb.finish()