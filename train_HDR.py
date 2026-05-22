import os
import time
import cv2
import glob
import numpy as np
import lpips
from datetime import datetime

# Reroute W&B cache to scratch BEFORE importing wandb
os.environ["WANDB_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
os.environ["WANDB_CACHE_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
os.environ["WANDB_CONFIG_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
os.environ["WANDB_DATA_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
import wandb

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from HDR_model_switch import HDR_model
from HDR_dataset import *

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
    # 🛡️ SAFETY FIX: Clamp negatives to 0.0 before log math to prevent NaN crashes
    safe_image = torch.clamp(hdr_image, min=0.0)
    return torch.log10(1.0 + mu * safe_image) / torch.log10(torch.tensor(1.0 + mu, device=safe_image.device))

def hdr_tonemap_np(hdr_image, nbits=20):
    mu = 2**nbits-1
    safe_image = np.clip(hdr_image, 0.0, None)
    return np.log10(1.0 + mu * safe_image) / np.log10(1.0 + mu)

def batch_psnr_gpu(img, gt, data_range=1.0):
    with torch.no_grad():
        mse = torch.mean((img.detach() - gt.detach()) ** 2, dim=[1, 2, 3])
        psnrs = torch.where(mse == 0, torch.tensor(100.0, device=img.device), 10.0 * torch.log10((data_range ** 2) / mse))
        return torch.mean(psnrs).item()

def learning_rate(lr, epoch):
    factor =  [1, 1, 2, 2, 5, 5, 10, 20, 30, 50, \
               2, 2, 5, 5, 10, 20, 30, 50, 60, 70, \
               5, 5, 10, 20, 30, 50, 60, 70, 80, 100, \
               10, 20, 50, 70, 100, 120, 150, 170, 200, 300]
    return lr / factor[epoch//10] if epoch//10 < len(factor) else lr / 400

if __name__ == "__main__":
    device = torch.device("cuda:0")

    seed = 21 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hyperparameters
    phase = "train"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    save_folder = f"models_rest_300_999_hi_noise_dim32_4444_{timestamp_str}/"
    pretrained_path = "./models_rest_200_999_hi_noise_dim32_4444/preexpand_hdr_best.pth"
    create_folder(save_folder)
    
    save_path_preexpand = save_folder + "preexpand_hdr.pth"
    best_save_path_preexpand = save_folder + "preexpand_hdr_best.pth"
    save_path = save_folder + "hdr.pth"
    best_save_path = save_folder + "hdr_best.pth"
    last_path = None
    
    # Data loading
    nbits = 20
    nbits_data = 10
    batch_sz = 128
    read_noise = 0
    regen_crops_every_epoch = 10
    regen_noise_every_epoch = 5
    
    # Optimizing
    lr = 4e-4
    optimizer_choice = "Adam" 
    num_epochs = 300
    start_epoch = 0 
    epoch_expand_mode = 999 
    
    best_running_psnr = 10.0
    last_epoch_psnr, last_epoch_loss = 0.0, 1e6

    ##########################################################################
    ## W&B Integration
    ##########################################################################
    wandb.init(
        project="hdr-kalantari",
        name=f"run_{timestamp_str}_kalantari",
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_sz,
            "optimizer": optimizer_choice,
            "nbits": nbits,
            "nbits_data": nbits_data,
            "model_architecture": "HDR_model_switch_dim32_4444",
            "seed": seed
        }
    )

    ##########################################################################
    ## Training Setup
    ##########################################################################
    print("Preparing model")
    model = HDR_model(dim=32, num_blocks=[4,4,4,4]).to(device)
    loss_lpips = lpips.LPIPS(net='vgg').to(device)

    logfile = save_folder + "log.txt"
    write_log(logfile, "", newfile=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(logfile, "Total params = %d" % pytorch_total_params)
    training_t0 = time.time()

    if optimizer_choice == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate(lr, start_epoch))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(lr, start_epoch))
        
    # Load/Init logic
    if os.path.exists(pretrained_path):
        print("Loading pretrained model")
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming from epoch {start_epoch}")
        
    if os.path.exists(save_path):
        print("Loading continued training model")
        checkpoint = torch.load(best_save_path if os.path.exists(best_save_path) else save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming from epoch {start_epoch}")
        
    model.train()

    print("Creating dataset")
    directory = "/scratch/gilbreth/chen4848/datasets/kalantari2017/train"
    dataset = HDRDataset(directory, patch_sz=128, num_patch=64, batch_sz=batch_sz, J=1, 
                         nbits=nbits, nbits_data=nbits_data, read_noise=read_noise, do_expand=False)

    # 🚀 OPTIMIZATION: Hardware-accelerated DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=True,          
        num_workers=8,         # 8 background CPU threads feeding the A100
        pin_memory=True,       # Hardware-level page-locking for ultra-fast GPU transfer
        drop_last=True         # Prevents weird batch shape crashes on the final epoch step
    )

    print("Training started")
    for epoch in range(start_epoch, num_epochs):
        current_lr = learning_rate(lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

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
            
            # 🚀 OPTIMIZATION: Loop the dataloader!
            for i, sample in enumerate(dataloader):
                x = sample["x"].to(device, non_blocking=True)
                y = sample["y"].to(device, non_blocking=True)
                xm = sample["xm"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y_pred = model(xm)
                    
                    loss_l2 = torch.mean((hdr_tonemap(y_pred, nbits) - hdr_tonemap(y, nbits)) ** 2)
                    
                    pred_tm = hdr_tonemap(y_pred, nbits).clamp(0, 1) * 2.0 - 1.0
                    gt_tm = hdr_tonemap(y, nbits).clamp(0, 1) * 2.0 - 1.0
                    loss_perceptual = loss_lpips(pred_tm, gt_tm).mean()
                    
                    gamma = 0.01 
                    ttl_loss = loss_l2 + (gamma * loss_perceptual)

                ttl_loss.backward()
                optimizer.step()
                
                running_loss += ttl_loss.item()
                running_psnr += batch_psnr_gpu(y_pred, y)
                print_running_loss(running_loss, running_psnr, batch_sz, i)

            # W&B: Log an image preview at the end of the epoch
            output_dir = "tmp/"
            create_folder(output_dir)
            vis_idx = 0
            img_gt = hdr_tonemap_np(y[vis_idx].detach().cpu().numpy())
            img_pred = hdr_tonemap_np(y_pred[vis_idx].detach().cpu().numpy())
            img_noisy = hdr_tonemap_np(x[vis_idx].detach().cpu().numpy())
            
            wandb.log({
                "Images/Prediction": wandb.Image(np.clip(np.transpose(img_pred, [1, 2, 0]), 0, 1), caption="Model Output"),
                "Images/Ground Truth": wandb.Image(np.clip(np.transpose(img_gt, [1, 2, 0]), 0, 1), caption="Target HDR"),
                "Images/Noisy Input": wandb.Image(np.clip(np.transpose(img_noisy, [1, 2, 0]), 0, 1), caption="Input")
            }, step=epoch)

            t2 = time.time()
            num_batches = len(dataloader)
            this_epoch_loss = running_loss / num_batches
            this_epoch_psnr = running_psnr / num_batches
            epoch_loss = print_epoch_loss(epoch+1, this_epoch_loss, this_epoch_psnr, t2-t1)
            write_log(logfile, epoch_loss, end="")

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": this_epoch_loss,
                "train/psnr": this_epoch_psnr,
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            improved = True
            
            if ((epoch < epoch_expand_mode or epoch >= epoch_expand_mode + 10) and last_epoch_psnr > 10 and this_epoch_loss > 2 * last_epoch_loss) or \
                (epoch >= epoch_expand_mode and epoch < epoch_expand_mode + 10 and this_epoch_loss > 1000):
                print(last_epoch_psnr, this_epoch_psnr, "do not save")
                improved = False
                dataset.regen_crops()
                dataset.regen_noise()
                checkpoint = torch.load(last_path)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                last_epoch_psnr = this_epoch_psnr
                last_epoch_loss = this_epoch_loss

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
                
                if this_epoch_psnr > best_running_psnr:
                    best_running_psnr = this_epoch_psnr
                    torch.save(save_dict, bsp)
                    
                    artifact = wandb.Artifact(f"kalantari_model", type="model")
                    artifact.add_file(bsp)
                    wandb.log_artifact(artifact, aliases=[f"epoch_{epoch+1}", "best"])

    training_t1 = time.time()
    write_log(logfile, "\nTotal training time = %.2f" % (training_t1 - training_t0))
    print("Training finished")
    wandb.finish()