import os
import glob
import numpy as np
import time
import cv2
# Prevent trained model overwrite
from datetime import datetime
# PyTorch
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image

#from model import *
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


##########################################################################
## Training util
##########################################################################
def hdr_tonemap(hdr_image, nbits=10):
    mu = 2**nbits-1
    return torch.log10(1.0 + mu * hdr_image) / torch.log10(torch.tensor(1.0 + mu))

def hdr_tonemap_np(hdr_image, nbits=10):
    mu = 2**nbits-1
    return np.log10(1.0 + mu * hdr_image) / np.log10(1.0 + mu)

# Metrics
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

# Learning rate
def learning_rate(lr, epoch):
    #factor =  [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 80, 120, 120, 160, 160, 200, 200]
    factor =  [1, 1, 2, 2, 5, 5, 10, 20, 30, 50, \
               2, 2, 5, 5, 10, 20, 30, 50, 60, 70, \
               5, 5, 10, 20, 30, 50, 60, 70, 80, 100, \
               10, 20, 50, 70, 100, 120, 150, 170, 200, 300]
    return lr / factor[epoch//10] if epoch//10 < len(factor) else lr / 400

# Given numpy image (in channel-first format), return the tonemap image
def to_tonemap(img):
    assert len(img.shape) == 3
    img = np.transpose(img, [1,2,0])[:,:,::-1]
    img = np.clip(img, 0, None)
    img = np.float32(img)
    # tonemap = cv2.createTonemapDurand(2.2)
    tonemap = cv2.createTonemap(gamma=2.2)
    tonemapped_image = tonemap.process(img)
    return tonemapped_image


# # get attention map from intermediate activation tensor
# def get_attention_map(activation_tensor, p=1):
#     """
#     Computes a spatial attention map from an intermediate activation tensor 
#     by summing absolute values raised to power p across the channel dimension.
#     """
#     # Formula from Section 3.1: sum of absolute values raised to power p 
#     attn = torch.pow(torch.abs(activation_tensor), p).sum(dim=1, keepdim=True)
    
#     # Normalize for visualization as suggested in paper [cite: 161]
#     attn_min = attn.min()
#     attn_max = attn.max()
#     attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)
    
#     return attn

def get_attention_map(activation_tensor, p=1):
    """
    Restormer-specific attention mapping.
    Uses L2 normalization to remove brightness bias and 
    p=1 to capture a broader structural response.
    """
    # Step 1: Normalize across channels (dim 1) B x C x H x W -> B x 1 x H x W
    A_norm = torch.nn.functional.normalize(activation_tensor, p=2, dim=1)
    
    # Step 2: Compute the Mean Absolute Deviation across channels
    # In Restormer, areas with high feature variance are the 'attended' areas
    # We use p=1 here to be less sensitive to extreme outliers
    attn = torch.abs(A_norm - A_norm.mean(dim=1, keepdim=True)).sum(dim=1, keepdim=True)
    
    # Step 3: Global min-max scaling for the patch
    attn_min, attn_max = attn.min(), attn.max()
    return (attn - attn_min) / (attn_max - attn_min + 1e-8)

    
# Add this helper to your test script
def save_transposed_map(attn_tensor, save_path):
    """
    attn_tensor: [Heads, C/head, C/head]
    """
    # Average across heads or pick the first head
    avg_attn = torch.mean(attn_tensor, dim=0).cpu().numpy()
    
    # Normalize for visualization
    avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)
    
    # Upscale for visibility (channel maps are often small, e.g., 16x16 or 32x32)
    avg_attn = cv2.resize(avg_attn, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * avg_attn), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(save_path, heatmap)


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    # Set up gpu
    device = torch.device("cuda:0")

    # Set seed
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Hyperparameters
    # Phase
    phase = "test"
    # Filenames
    save_folder = "models_rest_300_999_hi_noise_dim16_4442/"
    create_folder(save_folder)
    save_path = save_folder + "preexpand_hdr_best.pth"
    output_setting = "rest_300_999_hi_noise_dim16_4442_transposed_attention"

    # Data loading
    alpha = 1
    nbits = 10
    read_noise = 0 

    ####################################################################################################################################################
    ## Testing
    ####################################################################################################################################################

    if phase == "test":
        patch_sz = 512

        # Load model
        model = HDR_model()
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)

        for alpha in ["full"]:
            # Output folder
            if alpha == "full":
                output_name = "full"
            elif type(alpha) == type(0):
                output_name = "a%d" % int(alpha)
            else:
                output_name = "a%.2f" % alpha
            output_dir = f"results_preexpand_{nbits}bit_{output_setting}/{output_name}/"
            create_folder(output_dir)
            logfile = output_dir + "psnr.txt"

            # Load test samples
            directory = "/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/test"
            HDR_name = "ref_hdr_aligned.hdr"
            samples = load_test_samples(directory, HDR_name, nbits, read_noise, alpha=alpha)
            running_psnr = 0.0
            
            # Modify samples to only include the first one for quick testing
            samples = samples[:1]
            for i, sample in enumerate(samples):
                x, y = sample["xm"], sample["y"]
                print(x.shape, x.min(), x.max())
                print(y.shape, y.min(), y.max())
                x = modcrop(x, 64, channel_first=True)
                y = modcrop(y, 64, channel_first=True)
                C, H, W = x.shape
                x = torch.tensor(x[np.newaxis,...], dtype=torch.float).to(device)
                y_pred = np.zeros_like(y)
                
                # # We initialize a dictionary to hold full stitched maps for each layer
                # stitched_maps = {
                #     "x_level1": np.zeros((H, W), dtype=np.float32),
                #     "x_level2": np.zeros((H, W), dtype=np.float32),
                #     "x_latent": np.zeros((H, W), dtype=np.float32),
                #     "w_level2": np.zeros((H, W), dtype=np.float32),
                #     "w_level1": np.zeros((H, W), dtype=np.float32),
                #     "w_level0_refined": np.zeros((H, W), dtype=np.float32)
                # }
                
                # --- Initialize accumulators for Transposed Attention ---
                # We will figure out the exact shape dynamically on the first patch
                accumulated_t_maps = {}
                patch_count = 0

                pd1 = 8; pd2 = pd1*2
                ii_idx = 0
                for ii in range(0, H, patch_sz-pd2):
                    jj_idx = 0
                    for jj in range(0, W, patch_sz-pd2):
                        x_patch = torch.tensor(np.zeros((1,C,patch_sz,patch_sz)), dtype=torch.float).to(device)
                        ii_end, jj_end = min(ii+patch_sz,H), min(jj+patch_sz,W)
                        x_patch[:, :, :(ii_end-ii), :(jj_end-jj)] = x[:, :, ii:ii_end, jj:jj_end]
                        
                        
                        # with torch.no_grad():
                        #     y_patch = model(x_patch)
                        # Inside the jj loop (where you process y_patch)
                        
                        # --- NEW: Call the model requesting transposed attention ---
                        with torch.no_grad():
                            # Assumes you updated the model as shown in Step 2
                            y_patch, t_attn_maps_dict = model(x_patch, return_transposed_attn=True)

                        # --- NEW: Accumulate the transposed maps for this patch ---
                        for map_name, t_map_tensor in t_attn_maps_dict.items():
                            # t_map_tensor shape is [1, Heads, C/head, C/head]
                            # Move to CPU and remove batch dim
                            t_map_np = t_map_tensor[0].detach().cpu().numpy() 
                            
                            if map_name not in accumulated_t_maps:
                                accumulated_t_maps[map_name] = np.zeros_like(t_map_np)
                            accumulated_t_maps[map_name] += t_map_np
                        
                        patch_count += 1
                        
                        # ## This part is to output the feature map of each layer
                        # with torch.no_grad():
                        #     y_patch, maps = model(x_patch, return_maps=True)
                        
                        # # Inside the jj loop
                        # for map_name, tensor in maps.items():
                        #     attn_map = get_attention_map(tensor, p=2)
                        #     attn_np = attn_map[0, 0].cpu().numpy()
                            
                        #     # Resize the attention map to the patch size for stitching so that I don't need to deal with different sizes
                        #     attn_np = cv2.resize(attn_np, (patch_sz, patch_sz))
                            
                        #     # Calculate the actual boundaries for this patch
                        #     # This prevents the shape mismatch at the edges (the 472 vs 496 error)
                        #     h_start = ii_idx * (patch_sz - pd2) + pd1
                        #     h_end = min(h_start + (patch_sz - pd2), H) 
                            
                        #     w_start = jj_idx * (patch_sz - pd2) + pd1
                        #     w_end = min(w_start + (patch_sz - pd2), W)
                            
                        #     # Calculate how much of the patch we actually need
                        #     patch_h_end = pd1 + (h_end - h_start)
                        #     patch_w_end = pd1 + (w_end - w_start)
                            
                        #     # Stitch valid center
                        #     stitched_maps[map_name][h_start:h_end, w_start:w_end] = attn_np[pd1:patch_h_end, pd1:patch_w_end]
                            
                        #     # Handle top/left edges only on the first row/column
                        #     if ii_idx == 0:stitched_maps[map_name][:pd1, w_start:w_end] = attn_np[:pd1, pd1:patch_w_end]
                        #     if jj_idx == 0:stitched_maps[map_name][h_start:h_end, :pd1] = attn_np[pd1:patch_h_end, :pd1]
                        #     if ii_idx == 0 and jj_idx == 0:stitched_maps[map_name][:pd1, :pd1] = attn_np[:pd1, :pd1]
                            
                            
                        y_patch = y_patch.detach().cpu().numpy()
                        y_patch = y_patch[:, :, :(ii_end-ii), :(jj_end-jj)]

                        y_pred[:, ii_idx*(patch_sz-pd2)+pd1:(ii_idx+1)*(patch_sz-pd2)+pd1, jj_idx*(patch_sz-pd2)+pd1:(jj_idx+1)*(patch_sz-pd2)+pd1] = y_patch[0, :, pd1:patch_sz-pd1, pd1:patch_sz-pd1]
                        if ii_idx == 0: y_pred[:, :pd1, jj_idx*(patch_sz-pd2)+pd1:(jj_idx+1)*(patch_sz-pd2)+pd1] = y_patch[0, :, :pd1, pd1:patch_sz-pd1]
                        if jj_idx == 0: y_pred[:, ii_idx*(patch_sz-pd2)+pd1:(ii_idx+1)*(patch_sz-pd2)+pd1, :pd1] = y_patch[0, :, pd1:patch_sz-pd1, :pd1]
                        if ii_idx == 0 and jj_idx == 0: y_pred[:, :pd1, :pd1] = y_patch[0, :, :pd1, :pd1]
                        jj_idx += 1
                    ii_idx += 1
                
                # # After the loops finish, we have the full stitched attention maps for each layer in stitched_maps
                # for map_name, full_map in stitched_maps.items():
                #     # Normalize the full image map [cite: 161]
                #     full_map = (full_map - full_map.min()) / (full_map.max() - full_map.min() + 1e-8)
                    
                #     # Convert to heatmap
                #     full_map_uint8 = np.uint8(255 * full_map)
                #     heatmap = cv2.applyColorMap(full_map_uint8, cv2.COLORMAP_JET)
                    
                #     # Save the full image attention map
                #     cv2.imwrite(os.path.join(output_dir, f"{i}_FULL_attn_{map_name}.png"), heatmap)



                # --- NEW: Process and Save the Transposed Attention Maps after the loops ---
                for map_name, t_map_sum in accumulated_t_maps.items():
                    # 1. Average across all patches
                    avg_t_map = t_map_sum / patch_count
                    
                    # 2. Average across all Heads to get a single 2D matrix: [C/head, C/head]
                    head_avg_map = np.mean(avg_t_map, axis=0)
                    
                    # 3. Extract the actual dimensions of this specific attention map
                    map_h, map_w = head_avg_map.shape
                    
                    # 4. Normalize for visualization (0 to 1)
                    head_avg_map = (head_avg_map - head_avg_map.min()) / (head_avg_map.max() - head_avg_map.min() + 1e-8)
                    
                    # 5. Upscale for visibility
                    vis_map = cv2.resize(head_avg_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                    
                    # 6. Apply colormap and save with the dimensions in the filename
                    heatmap = cv2.applyColorMap(np.uint8(255 * vis_map), cv2.COLORMAP_VIRIDIS)
                    
                    # Construct the new filename with the size appended
                    filename = f"{i}_avg_transposed_attn_{map_name}_{map_h}x{map_w}.png"
                    cv2.imwrite(os.path.join(output_dir, filename), heatmap)
                    
                    
                # # --- NEW: Process and Save the Transposed Attention Maps after the loops ---
                # for map_name, t_map_sum in accumulated_t_maps.items():
                #     # 1. Average across all patches
                #     avg_t_map = t_map_sum / patch_count
                    
                #     # 2. Average across all Heads to get a single 2D matrix: [C/head, C/head]
                #     # (Alternatively, you could loop through avg_t_map to save each head separately)
                #     head_avg_map = np.mean(avg_t_map, axis=0)
                    
                #     # 3. Normalize for visualization (0 to 1)
                #     head_avg_map = (head_avg_map - head_avg_map.min()) / (head_avg_map.max() - head_avg_map.min() + 1e-8)
                    
                #     # 4. Upscale for visibility. A 16x16 pixel image is too small to see.
                #     # Using INTER_NEAREST preserves the hard grid squares of the matrix.
                #     vis_map = cv2.resize(head_avg_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                    
                #     # 5. Apply colormap and save
                #     # VIRIDIS or INFERNO are usually better than JET for correlation matrices
                #     heatmap = cv2.applyColorMap(np.uint8(255 * vis_map), cv2.COLORMAP_VIRIDIS)
                #     cv2.imwrite(os.path.join(output_dir, f"{i}_avg_transposed_attn_{map_name}.png"), heatmap)
                
                
                

                y_pred = torch.tensor(y_pred[np.newaxis,...], dtype=torch.float).to(device)
                y = torch.tensor(y[np.newaxis,...], dtype=torch.float).to(device)
                running_psnr += batch_psnr(y_pred, y)

                # Save images (noisy, prediction, ground truth)
                image = np.transpose(y[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
                image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_gt.png"%i), image)

                image = np.transpose(y[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
                cv2.imwrite(os.path.join(output_dir, "%d_gt.png"%i), image*255)
                image = to_tonemap(y[0].detach().cpu().numpy())
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_cv_gt.png"%i), image*255)

                image = np.transpose(y_pred[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
                # cv2.imwrite(os.path.join(output_dir, "%d_pred.png"%i), image*255)
                image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_pred.png"%i), image)

                image = np.transpose(x[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
                cv2.imwrite(os.path.join(output_dir, "%d_noisy.png"%i), image*255)
                image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_noisy.png"%i), image)

                image = np.transpose(sample["x"], [1,2,0])
                image = GBRG2disp(image)
                cv2.imwrite(os.path.join(output_dir, "%d_noisy_bayer.png"%i), image[:,:,::-1]*255)
                image = np.clip(hdr_tonemap_np(image[:,:,::-1])*255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_noisy_bayer.png"%i), image)

                image = to_tonemap(y_pred[0].detach().cpu().numpy())
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_cv_pred.png"%i), image*255)

                image = np.transpose(sample["xt"], [1,2,0])
                cv2.imwrite(os.path.join(output_dir, "%d_noisy_color.png"%i), image[:,:,::-1]*255)
                image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "%d_tonemap_noisy_color.png"%i), image[:,:,::-1])


            print(alpha)
            print("Avg PSNR = %.4f dB"%(running_psnr/i))
            write_log(logfile, "\nTest Avg PSNR = %.4f dB"%(running_psnr/i))
            


