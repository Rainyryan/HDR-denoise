import os
import glob
import numpy as np
import time
import cv2
import lpips  # Added for perceptual metric
# Prevent trained model overwrite
from datetime import datetime
# PyTorch
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image

from HDR_model_switch import HDR_model
from HDR_dataset import *
from HDR_model_hybrid_student import Hybrid_Student_HDR

##########################################################################
## Utility Functions (Unchanged)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##########################################################################
## Core Evaluation Function
##########################################################################
def evaluate_model(model, model_name, samples, device, loss_fn_vgg, nbits, patch_sz=512, output_dir=None):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} | Parameters: {count_parameters(model):,}")
    print(f"{'='*50}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    running_psnr = 0.0
    running_lpips = 0.0
    total_time = 0.0
    
    for i, sample in enumerate(samples):
        x, y = sample["xm"], sample["y"]
        x = modcrop(x, 64, channel_first=True)
        y = modcrop(y, 64, channel_first=True)
        C, H, W = x.shape
        x = torch.tensor(x[np.newaxis,...], dtype=torch.float).to(device)
        y_pred = np.zeros_like(y)

        pd1 = 8; pd2 = pd1*2
        ii_idx = 0
        
        # Start timing the entire image reconstruction
        torch.cuda.synchronize()
        start_time = time.time()
        
        for ii in range(0, H, patch_sz-pd2):
            jj_idx = 0
            for jj in range(0, W, patch_sz-pd2):
                x_patch = torch.tensor(np.zeros((1,C,patch_sz,patch_sz)), dtype=torch.float).to(device)
                ii_end, jj_end = min(ii+patch_sz,H), min(jj+patch_sz,W)
                x_patch[:, :, :(ii_end-ii), :(jj_end-jj)] = x[:, :, ii:ii_end, jj:jj_end]
                
                with torch.no_grad():
                    y_patch = model(x_patch)
                    
                y_patch = y_patch.detach().cpu().numpy()
                y_patch = y_patch[:, :, :(ii_end-ii), :(jj_end-jj)]

                # Overlap Blending logic
                y_pred[:, ii_idx*(patch_sz-pd2)+pd1:(ii_idx+1)*(patch_sz-pd2)+pd1, jj_idx*(patch_sz-pd2)+pd1:(jj_idx+1)*(patch_sz-pd2)+pd1] = y_patch[0, :, pd1:patch_sz-pd1, pd1:patch_sz-pd1]
                if ii_idx == 0: y_pred[:, :pd1, jj_idx*(patch_sz-pd2)+pd1:(jj_idx+1)*(patch_sz-pd2)+pd1] = y_patch[0, :, :pd1, pd1:patch_sz-pd1]
                if jj_idx == 0: y_pred[:, ii_idx*(patch_sz-pd2)+pd1:(ii_idx+1)*(patch_sz-pd2)+pd1, :pd1] = y_patch[0, :, pd1:patch_sz-pd1, :pd1]
                if ii_idx == 0 and jj_idx == 0: y_pred[:, :pd1, :pd1] = y_patch[0, :, :pd1, :pd1]
                jj_idx += 1
            ii_idx += 1

        # End timing
        torch.cuda.synchronize()
        total_time += (time.time() - start_time)
        
        y_pred = torch.tensor(y_pred[np.newaxis,...], dtype=torch.float).to(device)
        y = torch.tensor(y[np.newaxis,...], dtype=torch.float).to(device)

        # Calculate Metrics
        running_psnr += batch_psnr(y_pred, y)
        
        # --- Image Saving Block ---
        if output_dir:
            # Save Ground Truth
            gt_img = np.transpose(y[0].detach().cpu().numpy(), [1, 2, 0])[:, :, ::-1] # C,H,W to H,W,C and RGB to BGR
            gt_img = np.clip(hdr_tonemap_np(gt_img)*255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{i}_tonemap_gt.png"), gt_img)
            
            # Save Noisy Input
            # x is [1, C, H, W], we take x[0] and convert to H,W,C BGR
            noisy_img = np.transpose(x[0].detach().cpu().numpy(), [1, 2, 0])[:, :, ::-1]
            # Since input is typically already in log-domain or normalized for HDR, 
            # we scale to 255. Adjust if your 'xm' scale differs.
            noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{i}_noisy_input.png"), noisy_img)

            # Save Prediction
            pred_img = np.transpose(y_pred[0].detach().cpu().numpy(), [1, 2, 0])[:, :, ::-1]
            pred_img = np.clip(hdr_tonemap_np(pred_img)*255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{i}_{model_name}_tonemap_pred.png"), pred_img)
        
        # Calculate LPIPS (Requires values scaled to [-1, 1] using tonemapped outputs)
        pred_tm = hdr_tonemap(y_pred, nbits).clamp(0, 1)
        gt_tm = hdr_tonemap(y, nbits).clamp(0, 1)
        pred_lpips_input = pred_tm * 2.0 - 1.0
        gt_lpips_input = gt_tm * 2.0 - 1.0
        with torch.no_grad():
            running_lpips += loss_fn_vgg(pred_lpips_input, gt_lpips_input).item()

    num_samples = len(samples)
    avg_psnr = running_psnr / num_samples
    avg_lpips = running_lpips / num_samples
    avg_time = total_time / num_samples
    fps = 1.0 / avg_time

    print(f"Results for {model_name}:")
    print(f"Avg PSNR:  {avg_psnr:.4f} dB")
    print(f"Avg LPIPS: {avg_lpips:.4f}")
    print(f"Speed:     {fps:.2f} FPS ({avg_time*1000:.1f} ms / image)")
    return avg_psnr, avg_lpips, fps


if __name__ == "__main__":
    device = torch.device("cuda:0")
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup
    patch_sz = 512
    nbits = 10
    read_noise = 0 
    
    teacher_dim = 32
    student_dim = 8
    
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    

    # 1. Initialize & Load Teacher Model
    print("Loading Teacher...")
    teacher_model = HDR_model(dim=teacher_dim, num_blocks=[4,4,4,4]) 
    teacher_ckpt = "./models_rest_200_999_hi_noise_dim32_4444/preexpand_hdr_best.pth" # Update this path
    if os.path.exists(teacher_ckpt):
        teacher_model.load_state_dict(torch.load(teacher_ckpt)['model_state_dict'])
    else:
        raise FileNotFoundError(f"Could not find teacher checkpoint at {teacher_ckpt}")
    teacher_model.eval().to(device)

    # 2. Initialize & Load Student Model
    print("Loading Student...")
    # student_model = HDR_model(dim=student_dim, num_blocks=[2,2,2,1]) 
    student_model = Hybrid_Student_HDR(dim=student_dim, num_blocks=[2,2,2,1], num_refinement_blocks=2, heads=[1,2,4,8])
    student_ckpt = "./distill_models_20260420_173301/preexpand_hdr_best.pth"
    if os.path.exists(student_ckpt):
        student_model.load_state_dict(torch.load(student_ckpt)['student_state_dict'])
    else:
        raise FileNotFoundError(f"Could not find student checkpoint at {student_ckpt}")
    student_model.eval().to(device)

    directory = "/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/test"
    HDR_name = "ref_hdr_aligned.hdr"

    # Evaluate across different alpha levels
    for alpha in [1, 3, "full"]:
        print(f"\n>>> Running Evaluation for Alpha = {alpha} <<<")
        samples = load_test_samples(directory, HDR_name, nbits, read_noise, alpha=alpha)
        
        alpha_folder = f"eval_results_alpha_{alpha}"
        evaluate_model(teacher_model, f"Teacher (dim={teacher_dim})", samples, device, loss_fn_vgg, nbits, patch_sz, output_dir=alpha_folder)
        evaluate_model(student_model, f"Student (dim={student_dim})", samples, device, loss_fn_vgg, nbits, patch_sz, output_dir=alpha_folder)
        
