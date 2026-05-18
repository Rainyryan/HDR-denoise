import os
import glob
import numpy as np
import time
import cv2
import lpips
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import save_image

import wandb  # <-- Added W&B Import

# Import the base model and the new SNR-aware dataset
from HDR_model_hybrid_student import Hybrid_Student_HDR
from HDR_dataset_snr_map import HDRDataset

##########################################################################
## Dual SNR Denoiser Wrapper
##########################################################################
class DualSNRDenoiser(nn.Module):
    def __init__(self, **kwargs):
        super(DualSNRDenoiser, self).__init__()
        # Expert 1: Specializes in Low SNR (High Noise) regions
        self.denoiser_low_snr = Hybrid_Student_HDR(**kwargs)
        
        # Expert 2: Specializes in High SNR (Low Noise) regions
        self.denoiser_high_snr = Hybrid_Student_HDR(**kwargs)

    def forward(self, x, snr_map):
        """
        x: [B, C, H, W] noisy input
        snr_map: [B, 1, H, W] normalized between 0 and 1.
                 0 = extremely noisy (low SNR), 1 = clean (high SNR)
        """
        out_low = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)

        # Soft Blend: The SNR map dictates which model's output to trust for each pixel.
        blended_out = (1.0 - snr_map) * out_low + snr_map * out_high
        
        return blended_out, out_low, out_high

##########################################################################
## Main function util
##########################################################################
def print_running_loss(running_loss, running_psnr, batch_sz, i, print_every_epoch=10):
    psnr_str = "\tPSNR = %.2f" % (running_psnr/(i+1)) if running_psnr else ""
    if i % print_every_epoch == (print_every_epoch-1):
        print("\r\tLoss = %15.4f\t%s" % (running_loss/(batch_sz*(i+1)), psnr_str) , end="")

def print_epoch_loss(epoch_num, avg_loss, avg_psnr, time_spent=None):
    time_spent = "Unknown" if time_spent is None else "%f"%time_spent
    psnr_str = "\tPSNR = %.2f" % avg_psnr if avg_psnr else ""
    epoch_loss = "\rEpoch #%d: Loss = %12.4f \t%s\t(Time: %s)" % (epoch_num, avg_loss, psnr_str, time_spent)
    print(epoch_loss)
    return epoch_loss

def write_log(filename, s, newfile=False, end="\n"):
    if newfile:
        with open(filename, "w") as ofile:
            pass
    with open(filename, "a") as ofile:
        ofile.write("%s%s" % (s, end))

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

##########################################################################
## Training util
##########################################################################
def hdr_tonemap(hdr_image, nbits=20):
    mu = 2**nbits-1
    return torch.log10(1.0 + mu * hdr_image) / torch.log10(torch.tensor(1.0 + mu))

def hdr_tonemap_np(hdr_image, nbits=20):
    mu = 2**nbits-1
    return np.log10(1.0 + mu * hdr_image) / np.log10(1.0 + mu)

def batch_psnr(img, gt, data_range=1):
    def compare_psnr(img1, img2, data_range=1):
        mse = np.mean((img1.astype(float) - img2.astype(float))**2)
        if mse == 0:
            return 100
        return 10 * np.log10((data_range**2) / mse)
    img = img.data.cpu().numpy().astype(np.float32)
    gt = gt.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += compare_psnr(gt[i,:,:,:], img[i,:,:,:], data_range=1)
    return (PSNR/img.shape[0])

def learning_rate(lr, epoch):
    factor =  [1, 1, 2, 2, 5, 5, 10, 20, 30, 50, \
               2, 2, 5, 5, 10, 20, 30, 50, 60, 70, \
               5, 5, 10, 20, 30, 50, 60, 70, 80, 100, \
               10, 20, 50, 70, 100, 120, 150, 170, 200, 300]
    return lr / factor[epoch//10] if epoch//10 < len(factor) else lr / 400

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 21 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hyperparameters
    phase = "train"
    # timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    timestamp_str = "20260518_0442"
    save_folder = "models_MoE_hybrid_student_%s/" % timestamp_str
    pretrained_path = "./distill_models_20260518_0442/preexpand_hdr_best.pth"
    create_folder(save_folder)
    
    save_path_preexpand = save_folder + "preexpand_hdr.pth"
    best_save_path_preexpand = save_folder + "preexpand_hdr_best.pth"
    save_path = save_folder + "hdr.pth"
    best_save_path = save_folder + "hdr_best.pth"
    last_path = None
    
    # Data loading
    nbits = 20
    nbits_data = 10
    batch_sz = 8
    read_noise = 0
    regen_crops_every_epoch = 10
    regen_noise_every_epoch = 5
    
    # Optimizing
    lr = 2e-4
    optimizer_choice = "Adam" 
    num_epochs = 500
    start_epoch = 0 
    epoch_expand_mode = 999
    
    best_running_psnr = 10.0
    last_epoch_psnr, last_epoch_loss = 0.0, 1e6

    # ------------------ WANDB INITIALIZATION ------------------
    wandb.init(
        project="hdr-dual-moe",
        name=f"run_{timestamp_str}",
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_sz,
            "optimizer": optimizer_choice,
            "expand_mode_epoch": epoch_expand_mode,
            "nbits": nbits,
            "nbits_data": nbits_data,
            "read_noise": read_noise,
            "seed": seed,
            "model_architecture": "Dual_Hybrid_Student_HDR"
        }
    )
    # ------------------------------------------------------------

    ##########################################################################
    ## Training Setup
    ##########################################################################
    print("Preparing dual model")
    # Initialize the Dual Model wrapper
    model = DualSNRDenoiser(dim=8, num_blocks=[2, 2, 2, 1], num_refinement_blocks=2, heads=[1, 2, 4, 8]).to(device)
    
    loss_lpips = lpips.LPIPS(net='vgg').to(device)
    scaler = torch.cuda.amp.GradScaler()

    logfile = save_folder + "log_%s.txt" % timestamp_str
    
    # ------------------ DETAILED LOGGING SETUP ------------------
    log_header = (
        f"==========================================================\n"
        f"               DUAL MOE TRAINING SESSION                  \n"
        f"==========================================================\n"
        f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"--- PATHS ---\n"
        f"Save Folder:     {save_folder}\n"
        f"Pretrained Path: {pretrained_path}\n\n"
        f"--- HYPERPARAMETERS ---\n"
        f"Model Base:      Hybrid_Student_HDR (dim=8, blocks=[2,2,2,1], refine=2, heads=[1,2,4,8])\n"
        f"Total Epochs:    {num_epochs}\n"
        f"Batch Size:      {batch_sz}\n"
        f"Base LR:         {lr}\n"
        f"Optimizer:       {optimizer_choice}\n"
        f"Expand Mode @:   {epoch_expand_mode}\n"
        f"Loss Weights:    L2 Main (1.0), Perceptual (0.01), Aux (0.5)\n"
    )
    
    write_log(logfile, log_header, newfile=True)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(logfile, f"Total params = {pytorch_total_params}\n")
    write_log(logfile, "==========================================================\n")
    # ------------------------------------------------------------
    training_t0 = time.time()

    if optimizer_choice == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate(lr, start_epoch))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(lr, start_epoch))
        
    # ------------------------------------------------------------
    # Model Loading Logic: Check reverse chronological order
    # ------------------------------------------------------------
    if os.path.exists(save_path) or os.path.exists(best_save_path):
        print("Loading expanded mode dual model (resuming later training stage)")
        load_target = save_path if os.path.exists(save_path) else best_save_path
        checkpoint = torch.load(load_target)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        last_epoch_loss = checkpoint.get('loss', 1e6)
        print(f"Resumed expanded mode from epoch {start_epoch}")

    elif os.path.exists(save_path_preexpand) or os.path.exists(best_save_path_preexpand):
        print("Loading pre-expand dual model (resuming early training stage)")
        load_target = save_path_preexpand if os.path.exists(save_path_preexpand) else best_save_path_preexpand
        checkpoint = torch.load(load_target)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        last_epoch_loss = checkpoint.get('loss', 1e6)
        print(f"Resumed pre-expand mode from epoch {start_epoch}")

    elif os.path.exists(pretrained_path):
        print("Loading pretrained single student model into both experts")
        checkpoint = torch.load(pretrained_path)
        
        state_dict_to_load = checkpoint.get('student_state_dict', checkpoint.get('model_state_dict'))
        
        model.denoiser_low_snr.load_state_dict(state_dict_to_load, strict=False)
        model.denoiser_high_snr.load_state_dict(state_dict_to_load, strict=False)
        print("Initialized both experts with pretrained distilled weights. Starting from epoch 0.")
        
    else:
        print("No prior checkpoints or pretrained weights found. Starting from scratch.")
        
    model.train()

    print("Creating dataset")
    directory = "/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/train"
    dataset = HDRDataset(directory, patch_sz=128, num_patch=64, batch_sz=batch_sz, J=1, 
                        nbits=nbits, nbits_data=nbits_data, read_noise=read_noise, do_expand=False)

    print("Training started")
    for epoch in range(start_epoch, num_epochs):
        if epoch >= epoch_expand_mode:
            dataset.do_expand = True
        if regen_crops_every_epoch != 0 and epoch % regen_crops_every_epoch == 0 and epoch != 0:
            dataset.regen_crops()
        if regen_noise_every_epoch != 0 and epoch % regen_noise_every_epoch == 0 and epoch != 0:
            dataset.regen_noise()

        improved = False
        while not improved:
            running_loss = 0.0
            running_psnr = 0.0
            t1 = time.time()
            
            for i in range(len(dataset)):
                sample = dataset[i]
                
                x = sample["x"].to(device, non_blocking=True)
                y = sample["y"].to(device, non_blocking=True)
                xm = sample["xm"].to(device, non_blocking=True)
                
                snr_map = sample["snr_map"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    y_pred, y_low_snr, y_high_snr = model(xm, snr_map)
                    
                    gt_tm = hdr_tonemap(y, nbits)
                    
                    loss_l2_main = torch.mean((hdr_tonemap(y_pred, nbits) - gt_tm) ** 2)
                    
                    mask_low = (snr_map < 0.5).float()
                    mask_high = (snr_map >= 0.5).float()
                    
                    se_low = (hdr_tonemap(y_low_snr, nbits) - gt_tm) ** 2
                    se_high = (hdr_tonemap(y_high_snr, nbits) - gt_tm) ** 2
                    
                    loss_aux_low = torch.sum(mask_low * se_low) / (3.0 * torch.sum(mask_low) + 1e-6)
                    loss_aux_high = torch.sum(mask_high * se_high) / (3.0 * torch.sum(mask_high) + 1e-6)
                    
                    pred_tm_clamped = hdr_tonemap(y_pred, nbits).clamp(0, 1) * 2.0 - 1.0
                    gt_tm_clamped = gt_tm.clamp(0, 1) * 2.0 - 1.0
                    loss_perceptual = loss_lpips(pred_tm_clamped, gt_tm_clamped).mean()
                    
                    gamma = 0.01      
                    aux_weight = 0.1  
                    ttl_loss = loss_l2_main + (gamma * loss_perceptual) + aux_weight * (loss_aux_low + loss_aux_high)

                scaler.scale(ttl_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += ttl_loss.item()
                running_psnr += batch_psnr(y_pred, y)
                print_running_loss(running_loss, running_psnr, batch_sz, i)

            # Save tmp image & Prepare WandB Images
            output_dir = "tmp_dual/"
            create_folder(output_dir)
            
            wandb_images = [] # Collect images for W&B
            
            for i in range(min(5, batch_sz)):
                # 1. Ground Truth
                image_gt = hdr_tonemap_np(y[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_gt.png", np.transpose(image_gt, [1,2,0])[:,:,::-1]*255)
                # W&B expects RGB arrays (0-1 float or 0-255 uint8)
                img_gt_rgb = np.clip(np.transpose(image_gt, [1, 2, 0]), 0, 1)
                
                # 2. Prediction
                image_pred = hdr_tonemap_np(y_pred[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_pred.png", np.transpose(image_pred, [1,2,0])[:,:,::-1]*255)
                img_pred_rgb = np.clip(np.transpose(image_pred, [1, 2, 0]), 0, 1)
                
                # 3. Noisy Input
                image_noisy = hdr_tonemap_np(x[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_noisy.png", np.transpose(image_noisy, [1,2,0])[:,:,::-1]*255)
                img_noisy_rgb = np.clip(np.transpose(image_noisy, [1, 2, 0]), 0, 1)
                
                # 4. SNR Map
                image_snr = snr_map[i].detach().cpu().numpy()
                image_snr_scaled = np.transpose(image_snr, [1,2,0]) * 255.0
                cv2.imwrite(f"{output_dir}{i}np_snr_map.png", image_snr_scaled.astype(np.uint8))
                img_snr_gray = np.clip(np.transpose(image_snr, [1, 2, 0]), 0, 1)

                # Add to W&B list (limit to first 3 to save dashboard bandwidth)
                if i < 3:
                    wandb_images.append(wandb.Image(img_gt_rgb, caption=f"GT {i}"))
                    wandb_images.append(wandb.Image(img_pred_rgb, caption=f"Pred {i}"))
                    wandb_images.append(wandb.Image(img_noisy_rgb, caption=f"Noisy {i}"))
                    wandb_images.append(wandb.Image(img_snr_gray, caption=f"SNR {i}"))

            t2 = time.time()
            this_epoch_loss = running_loss/(batch_sz*len(dataset))
            this_epoch_psnr = running_psnr/len(dataset)
            epoch_loss = print_epoch_loss(epoch+1, this_epoch_loss, this_epoch_psnr, t2-t1)
            write_log(logfile, epoch_loss, end="")

            # ------------------ WANDB METRIC LOGGING ------------------
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": this_epoch_loss,
                "train/psnr": this_epoch_psnr,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "visualizations": wandb_images
            })
            # ----------------------------------------------------------

            improved = True
            
            # Rollback logic
            if ((epoch < epoch_expand_mode or epoch >= epoch_expand_mode + 10) and last_epoch_psnr > 10 and this_epoch_loss > 2 * last_epoch_loss) or \
                (epoch >= epoch_expand_mode and epoch < epoch_expand_mode + 10 and this_epoch_loss > 1000):
                print(last_epoch_psnr, this_epoch_psnr, "do not save")
                improved = False
                dataset.regen_crops()
                dataset.regen_noise()
                if last_path and os.path.exists(last_path):
                    checkpoint = torch.load(last_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
            else:
                last_epoch_psnr = this_epoch_psnr
                last_epoch_loss = this_epoch_loss

            # Save model
            if improved:
                sp = save_path if epoch >= epoch_expand_mode else save_path_preexpand
                bsp = best_save_path if epoch >= epoch_expand_mode else best_save_path_preexpand
                last_path = sp
                
                save_dict = {
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': this_epoch_loss
                }
                
                torch.save(save_dict, sp)
                
                # Update best model and log artifact to W&B
                if running_psnr > best_running_psnr:
                    best_running_psnr = running_psnr
                    torch.save(save_dict, bsp)
                    
                    # ------------------ WANDB ARTIFACT LOGGING ------------------
                    artifact_name = "model_expanded" if epoch >= epoch_expand_mode else "model_preexpand"
                    artifact = wandb.Artifact(f"dual_hdr_{artifact_name}", type="model")
                    artifact.add_file(bsp)
                    wandb.log_artifact(artifact, aliases=[f"epoch_{epoch+1}", "best"])
                    # ------------------------------------------------------------

    training_t1 = time.time()
    write_log(logfile, "\nTotal training time = %.2f" % (training_t1 - training_t0))
    print("Training finished")
    
    # Close the W&B run
    wandb.finish()