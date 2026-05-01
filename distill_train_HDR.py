import os
import glob
import numpy as np
import time
import cv2
import lpips
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Assuming these are available in your directory
from HDR_model_switch import HDR_model
from HDR_dataset import *
from HDR_model_hybrid_student import Hybrid_Student_HDR
from distiller import RestormerDistiller

##########################################################################
## Main function util
##########################################################################
def print_running_loss(running_loss, running_psnr, batch_sz, i, print_every_epoch=10):
    psnr_str = "\tPSNR = %.2f" % (running_psnr/(i+1)) if running_psnr else ""
    if i % print_every_epoch == (print_every_epoch-1):
        print("\r\tLoss = %15.4f\t%s" % (running_loss/(i+1), psnr_str) , end="")

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

##########################################################################
## Training util
##########################################################################
def hdr_tonemap(hdr_image, nbits=10):
    mu = 2**nbits-1
    return torch.log10(1.0 + mu * hdr_image) / torch.log10(torch.tensor(1.0 + mu))

def hdr_tonemap_np(hdr_image, nbits=10):
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

##########################################################################
## Training
##########################################################################
if __name__ == "__main__":
    # Setup device and seed
    device = torch.device("cuda:0")
    seed = 21 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configuration
    phase = "train"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = "distill_models_%s/" % timestamp_str
    create_folder(save_folder)
    
    save_path_preexpand = save_folder + "preexpand_hdr.pth"
    best_save_path_preexpand = save_folder + "preexpand_hdr_best.pth"
    save_path = save_folder + "hdr.pth"
    best_save_path = save_folder + "hdr_best.pth"
    last_path = None
    # pretrained_path = ""
    pretrained_path = "./distill_models_20260420_173301/preexpand_hdr_best.pth" 
 
    # Data parameters
    nbits = 20 
    nbits_data = 10
    batch_sz = 8
    read_noise = 0
    regen_crops_every_epoch = 10 
    regen_noise_every_epoch = 5  
    
    # Training hyperparameters
    lr = 2e-4 
    optimizer_choice = "Adam" 
    num_epochs = 1000
    start_epoch = 0 
    epoch_expand_mode = 9999 
    
    # Loss Weights
    alpha = 1   # Output distillation weight
    gamma = 0.001  # LPIPS perceptual weight
    feat_weights = {
        "x_level1": 0.01,
        "x_level2": 0.02,
        "x_latent": 0.10,
        "w_level2": 0.05,
        "w_level1": 0.02,
        "w_level0_refined": 0.01
    }
    
    best_running_psnr = 10.0
    last_epoch_psnr, last_epoch_loss = 0.0, 1e6

    # =================================================================
    # Model Initialization
    # =================================================================
    print("Preparing models for Distillation")
    
    # Define model configurations
    teacher_dim = 32
    teacher_blocks = [4, 4, 4, 4]

    student_dim = 8
    student_blocks = [2, 2, 2, 1]
    student_refinement = 2
    student_heads = [1, 2, 4, 8]

    # Initialize models
    teacher_model = HDR_model(dim=teacher_dim, num_blocks=teacher_blocks) 
    student_model = Hybrid_Student_HDR(dim=student_dim, num_blocks=student_blocks, num_refinement_blocks=student_refinement, heads=student_heads)

    teacher_checkpoint_path = "./models_rest_800_999_hi_noise_dim32_4444/preexpand_hdr_best.pth"
    if os.path.exists(teacher_checkpoint_path):
        print(f"Loading Teacher weights from {teacher_checkpoint_path}")
        checkpoint = torch.load(teacher_checkpoint_path)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: Teacher weights not found. Using random weights.")
        exit(1)

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    distiller = RestormerDistiller(
        teacher_model=teacher_model, 
        student_model=student_model, 
        teacher_dim=teacher_dim, 
        student_dim=student_dim
    ).to(device)

    if optimizer_choice == "RMSprop":
        optimizer = torch.optim.RMSprop([p for p in distiller.parameters() if p.requires_grad], lr=learning_rate(lr, start_epoch))
    else:
        optimizer = torch.optim.Adam([p for p in distiller.parameters() if p.requires_grad], lr=learning_rate(lr, start_epoch))

    # Initialize Perceptual Loss Network
    loss_lpips = lpips.LPIPS(net='vgg').to(device)
    
    # Initialize Gradient Scaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    logfile = save_folder + "log_%s.txt" % timestamp_str
    write_log(logfile, "", newfile=True)
    pytorch_total_params = sum(p.numel() for p in distiller.student.parameters() if p.requires_grad)
    write_log(logfile, f"Total Student params = {pytorch_total_params}")


    # Checkpoint loading
    if os.path.exists(pretrained_path):
        print("Loading continued training model")
        checkpoint = torch.load(pretrained_path)
        if 'distiller_state_dict' in checkpoint:
            distiller.load_state_dict(checkpoint['distiller_state_dict'])
        else:
            distiller.student.load_state_dict(checkpoint['student_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming from epoch {start_epoch}")
        
    # Format the log string to include the new variables
    log_string = (
        f"Configuration:\n"
        f"nbits={nbits}, nbits_data={nbits_data}, batch_sz={batch_sz}, read_noise={read_noise}, num_epochs={num_epochs}\n"
        f"Teacher Model: {teacher_model.__class__.__name__} (dim={teacher_dim}, num_blocks={teacher_blocks})\n"
        f"Student Model: {student_model.__class__.__name__} (dim={student_dim}, num_blocks={student_blocks}, "
        f"num_refinement={student_refinement}, heads={student_heads})\n"
        f"Loss Weights: alpha={alpha}, gamma={gamma}\n\n" # <--- Added this line
    )
    
    write_log(logfile, log_string)
    
    
    distiller.train()

    print("Creating dataset")
    directory = "/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/train"
    dataset = HDRDataset(directory, patch_sz=128, num_patch=64, batch_sz=batch_sz, J=1, 
                        nbits=nbits, nbits_data=nbits_data, read_noise=read_noise, do_expand=False)

    print("Training started")
    training_t0 = time.time()

    # =================================================================
    # Training Loop
    # =================================================================
    for epoch in range(start_epoch, num_epochs):
        if epoch >= epoch_expand_mode:
            dataset.do_expand = True
        if regen_crops_every_epoch != 0 and epoch % regen_crops_every_epoch == 0 and epoch != 0:
            dataset.regen_crops()
        if regen_noise_every_epoch != 0 and epoch % regen_noise_every_epoch == 0 and epoch != 0:
            dataset.regen_noise()
        
        improved = False
        while not improved:
            current_lr = learning_rate(lr, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
            running_loss = 0.0
            running_psnr = 0.0
            t1 = time.time()
            
            for i in range(len(dataset)):
                sample = dataset[i]
                x, y, xm = sample["x"].to(device), sample["y"].to(device), sample["xm"].to(device)

                optimizer.zero_grad()
                
                # Use Automatic Mixed Precision for faster compute and lower VRAM
                with torch.cuda.amp.autocast():
                    # Forward pass
                    s_out, t_out, s_maps_aligned, t_maps = distiller(xm)
                    
                    # 1. Base Task Loss (Student vs GT)
                    loss_task = torch.mean((hdr_tonemap(s_out, nbits) - hdr_tonemap(y, nbits)) ** 2)
                    
                    # 2. Output Distillation (Student vs Teacher)
                    loss_out_distill = F.l1_loss(hdr_tonemap(s_out, nbits), hdr_tonemap(t_out, nbits))
                    
                    # 3. LPIPS Perceptual Loss (Student vs GT)
                    # Scale tonemapped images to [-1, 1] for the VGG network
                    s_out_tm = hdr_tonemap(s_out, nbits).clamp(0, 1) * 2.0 - 1.0
                    gt_tm = hdr_tonemap(y, nbits).clamp(0, 1) * 2.0 - 1.0
                    loss_perceptual = loss_lpips(s_out_tm, gt_tm).mean()
                    
                    # 4. Layer-Wise Feature Distillation
                    loss_feat = 0
                    for key in feat_weights:
                        loss_feat += feat_weights[key] * F.mse_loss(s_maps_aligned[key], t_maps[key])
                        
                    # 5. Total Loss Calculation
                    ttl_loss = loss_task + (alpha * loss_out_distill) + loss_feat + (gamma * loss_perceptual)
                
                # Scaled Backward Pass
                scaler.scale(ttl_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += ttl_loss.item()
                running_psnr += batch_psnr(s_out, y)
                print_running_loss(running_loss, running_psnr, batch_sz, i)

            # Logging & Saving images
            output_dir = "tmp/"
            create_folder(output_dir)
            for i in range(min(5, batch_sz)):
                image = hdr_tonemap_np(y[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_gt.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)
                image = hdr_tonemap_np(s_out[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_pred.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)
                image = hdr_tonemap_np(x[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_noisy.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)
                image = hdr_tonemap_np(xm[i].detach().cpu().numpy())
                cv2.imwrite(f"{output_dir}{i}np_dm.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)

            t2 = time.time()
            this_epoch_loss = running_loss/len(dataset)
            this_epoch_psnr = running_psnr/len(dataset)
            epoch_loss_str = print_epoch_loss(epoch+1, this_epoch_loss, this_epoch_psnr, t2-t1)
            write_log(logfile, epoch_loss_str, end="")

            improved = True
            
            # Rollback logic
            if ((epoch < epoch_expand_mode or epoch >= epoch_expand_mode + 10) and last_epoch_psnr > 10 and this_epoch_loss > 2 * last_epoch_loss) or \
                (epoch >= epoch_expand_mode and epoch < epoch_expand_mode + 10 and this_epoch_loss > 1000):
                print(last_epoch_psnr, this_epoch_psnr, "do not save")
                improved = False
                dataset.regen_crops()
                dataset.regen_noise()
                
                checkpoint = torch.load(last_path)
                if 'distiller_state_dict' in checkpoint:
                    distiller.load_state_dict(checkpoint['distiller_state_dict'])
                else:
                    distiller.student.load_state_dict(checkpoint['student_state_dict'])
            else:
                last_epoch_psnr = this_epoch_psnr
                last_epoch_loss = this_epoch_loss

            # Save checkpoint
            if improved:
                sp = save_path if epoch >= epoch_expand_mode else save_path_preexpand
                bsp = best_save_path if epoch >= epoch_expand_mode else best_save_path_preexpand
                last_path = sp
                
                save_dict = {
                    'epoch': epoch+1,
                    'student_state_dict': distiller.student.state_dict(),
                    'distiller_state_dict': distiller.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': this_epoch_loss
                }
                
                torch.save(save_dict, sp)
                if this_epoch_psnr > best_running_psnr:  # Use the average!
                    best_running_psnr = this_epoch_psnr
                    torch.save(save_dict, bsp)

    training_t1 = time.time()
    write_log(logfile, "\nTotal training time = %.2f" % (training_t1 - training_t0))
    print("Training finished")