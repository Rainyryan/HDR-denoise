import numpy as np
import time
import cv2

# Define Prb 7x7 matrix
Prb_matrix = np.array([
    [ 0, 0, -0.0313, 0, -0.0313, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [-0.0313, 0, 0.3125, 0, 0.3125, 0, -0.0313],
    [ 0, 0, 0, 0, 0, 0, 0],
    [-0.0313, 0, 0.3125, 0, 0.3125, 0, -0.0313],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, -0.0313, 0, -0.0313, 0, 0]
], dtype=np.float64)

def fast_copyMakeBorder(src, top, bottom, left, right, borderType, value=0):
    if borderType == 'BORDER_REFLECT_101': # Matches BORDER_DEFAULT
        return np.pad(src, ((top, bottom), (left, right)), mode='reflect')
    elif borderType == 'BORDER_CONSTANT':
        return np.pad(src, ((top, bottom), (left, right)), mode='constant', constant_values=value)
    else:
        raise NotImplementedError(f"Mode {borderType} not implemented")


def fast_filter2D(src, kernel):
    # Use OpenCV's ultra-optimized C++ backend filtering engine
    # -1 keeps the output depth the same as the input
    return cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
# def fast_filter2D(src, kernel):
#     src = src.astype(np.float64, copy=False)
#     kernel = kernel.astype(np.float64, copy=False)
    
#     k_h, k_w = kernel.shape
#     pad_h = k_h // 2
#     pad_w = k_w // 2
    
#     # Pad image once
#     padded = np.pad(src, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
#     output = np.zeros_like(src)
#     h, w = src.shape
#     for i in range(k_h):
#         for j in range(k_w):
#             weight = kernel[i, j]
#             if weight != 0:
#                 output += weight * padded[i:i+h, j:j+w]
                
#     return output

def to_3Channel_vectorized(bayer, height, width):
    Dst = np.zeros((height, width, 3), dtype=np.float64)
    
    # GRBG Pattern Layout
    # Red: Row 0, 2... | Col 1, 3...
    Dst[0::2, 1::2, 2] = bayer[0::2, 1::2]
    # Blue: Row 1, 3... | Col 0, 2...
    Dst[1::2, 0::2, 0] = bayer[1::2, 0::2]
    # Green: Checkerboard
    Dst[0::2, 0::2, 1] = bayer[0::2, 0::2]
    Dst[1::2, 1::2, 1] = bayer[1::2, 1::2]
    
    return Dst

def GBTF_CFAInterpolation(Bayer, BayerPatternFlag=0):
    Bayer = Bayer.astype(np.float64, copy=False)
    
    # 0. Align Bayer Pattern (Reflect Padding)
    pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0
    if BayerPatternFlag == 1: # RGGB
        pad_top, pad_right = 0, 1
        pad_bot, pad_left = 1, 1
    elif BayerPatternFlag == 2: # GBRG
        pad_top, pad_bot, pad_left, pad_right = 1, 1, 1, 1
    elif BayerPatternFlag == 3: # BGGR
        pad_top, pad_bot = 1, 1
        pad_left, pad_right = 0, 0
    
    if any((pad_top, pad_bot, pad_left, pad_right)):
        # Use edge padding or reflect to align phase
        Src = np.pad(Bayer, ((pad_top, pad_bot), (pad_left, pad_right)), mode='edge')
    else:
        Src = Bayer

    height, width = Src.shape
    Dst = to_3Channel_vectorized(Src, height, width)

    # H & V Color Difference Maps, Kernel: [-0.25, 0.5, 0.5, 0.5, -0.25]
    k_1d = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=np.float64)
    HK = k_1d.reshape(1, 5)
    VK = k_1d.reshape(5, 1)

    HCDMap = fast_filter2D(Src, HK)
    VCDMap = fast_filter2D(Src, VK)

    checker = np.ones((height, width), dtype=np.float64)
    checker[1::2, ::2] = -1
    checker[::2, 1::2] = -1

    HCDMap = checker * (Src - HCDMap)
    VCDMap = checker * (Src - VCDMap)

    # Gradients
    HCD_pad = fast_copyMakeBorder(HCDMap, 5, 5, 5, 5, 'BORDER_CONSTANT', value=0)
    VCD_pad = fast_copyMakeBorder(VCDMap, 5, 5, 5, 5, 'BORDER_CONSTANT', value=0)

    HGradientMap = np.abs(HCD_pad[5:5+height, 6:6+width] - HCD_pad[5:5+height, 4:4+width])
    VGradientMap = np.abs(VCD_pad[6:6+height, 5:5+width] - VCD_pad[4:4+height, 5:5+width])

    # Weighting Table
    H_grad_pad = np.pad(HGradientMap, ((0,0), (2,2)), mode='constant', constant_values=0)
    # Sum sliding window of 5
    H_sum_h = (H_grad_pad[:, 0:width] + H_grad_pad[:, 1:width+1] + H_grad_pad[:, 2:width+2] + 
                H_grad_pad[:, 3:width+3] + H_grad_pad[:, 4:width+4])
    
    # Vertical Sum of the result
    H_sum_pad = np.pad(H_sum_h, ((2,2), (0,0)), mode='constant', constant_values=0)
    H_sum_total = (H_sum_pad[0:height, :] + H_sum_pad[1:height+1, :] + H_sum_pad[2:height+2, :] + 
                    H_sum_pad[3:height+3, :] + H_sum_pad[4:height+4, :])
    
    # Same for VGradientMap
    V_grad_pad = np.pad(VGradientMap, ((0,0), (2,2)), mode='constant', constant_values=0)
    V_sum_h = (V_grad_pad[:, 0:width] + V_grad_pad[:, 1:width+1] + V_grad_pad[:, 2:width+2] + 
                V_grad_pad[:, 3:width+3] + V_grad_pad[:, 4:width+4])
    V_sum_pad = np.pad(V_sum_h, ((2,2), (0,0)), mode='constant', constant_values=0)
    V_sum_total = (V_sum_pad[0:height, :] + V_sum_pad[1:height+1, :] + V_sum_pad[2:height+2, :] + 
                    V_sum_pad[3:height+3, :] + V_sum_pad[4:height+4, :])

    epsilon = 1e-9
    
    # Pad sums to allow shifting
    V_sum_total_pad = np.pad(V_sum_total, ((2,2), (0,0)), mode='edge')
    H_sum_total_pad = np.pad(H_sum_total, ((0,0), (2,2)), mode='edge')
    
    # N: Shift Down by 2
    N_val = V_sum_total_pad[0:height, :] 
    # S: Shift Up by 2
    S_val = V_sum_total_pad[4:4+height, :]
    # E (Left): Shift Right by 2
    E_val = H_sum_total_pad[:, 0:width]
    # W (Right): Shift Left by 2
    W_val = H_sum_total_pad[:, 4:4+width]
    
    N = 1.0 / (N_val**2 + epsilon)
    S = 1.0 / (S_val**2 + epsilon)
    E = 1.0 / (E_val**2 + epsilon)
    W = 1.0 / (W_val**2 + epsilon)
    
    TotalWeight = N + S + E + W

    # Green Interpolation
    HCD_pad9 = np.pad(HCDMap, ((0,0), (4,4)), mode='reflect')
    VCD_pad9 = np.pad(VCDMap, ((4,4), (0,0)), mode='reflect')
    
    # Sum Left 4 (x-4 to x-1)
    H_sum_L = (HCD_pad9[:, 0:width] + HCD_pad9[:, 1:width+1] + 
                HCD_pad9[:, 2:width+2] + HCD_pad9[:, 3:width+3])
    # Sum Right 4 (x+1 to x+4)
    H_sum_R = (HCD_pad9[:, 5:width+5] + HCD_pad9[:, 6:width+6] + 
                HCD_pad9[:, 7:width+7] + HCD_pad9[:, 8:width+8])
    # Sum Up 4 (y-4 to y-1)
    V_sum_U = (VCD_pad9[0:height, :] + VCD_pad9[1:height+1, :] + 
                VCD_pad9[2:height+2, :] + VCD_pad9[3:height+3, :])
    # Sum Down 4 (y+1 to y+4)
    V_sum_D = (VCD_pad9[5:height+5, :] + VCD_pad9[6:height+6, :] + 
                VCD_pad9[7:height+7, :] + VCD_pad9[8:height+8, :])

    AccumH = (E * 0.2 * H_sum_L) + ((W + E) * 0.2 * HCDMap) + (W * 0.2 * H_sum_R)
    AccumV = (N * 0.2 * V_sum_U) + ((N + S) * 0.2 * VCDMap) + (S * 0.2 * V_sum_D)
    
    TPdiff = (AccumH + AccumV) / (TotalWeight + epsilon)
    
    # Apply to holes in Green (Red/Blue pixels)
    y_idx, x_idx = np.indices((height, width))
    mask_missing_G = (y_idx + x_idx) % 2 == 1 # Checkboard pattern hole
    
    G_new = np.clip(Src + TPdiff, 0.0, 1.0)
    Dst[..., 1] = np.where(mask_missing_G, G_new, Dst[..., 1])

    # R/B on R/B pixels (using Prb)
    # TPdiff * Prb_matrix convolution
    Correction = fast_filter2D(TPdiff, Prb_matrix)
    
    RB_val = np.clip(Dst[..., 1] - Correction, 0.0, 1.0)
    
    # Masks
    # Red Locations: Row even, Col odd (Need Blue)
    mask_R = (y_idx % 2 == 0) & (x_idx % 2 == 1)
    # Blue Locations: Row odd, Col even (Need Red)
    mask_B = (y_idx % 2 == 1) & (x_idx % 2 == 0)
    
    Dst[..., 0] = np.where(mask_R, RB_val, Dst[..., 0])
    Dst[..., 2] = np.where(mask_B, RB_val, Dst[..., 2])

    # R/B on Green pixels
    # Average of 4 neighbors for difference
    GmB = Dst[..., 1] - Dst[..., 0]
    GmR = Dst[..., 1] - Dst[..., 2]
    
    # Kernel for 4 neighbors average: [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]
    k_cross = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=np.float64)
    
    GmB_smooth = fast_filter2D(GmB, k_cross)
    GmR_smooth = fast_filter2D(GmR, k_cross)
    
    B_rec = np.clip(Dst[..., 1] - GmB_smooth, 0.0, 1.0)
    R_rec = np.clip(Dst[..., 1] - GmR_smooth, 0.0, 1.0)
    
    mask_G = (y_idx + x_idx) % 2 == 0
    Dst[..., 0] = np.where(mask_G, B_rec, Dst[..., 0])
    Dst[..., 2] = np.where(mask_G, R_rec, Dst[..., 2])

    # Crop Padding
    if BayerPatternFlag == 1:
        Dst = Dst[:, 1:width-1, :]
    elif BayerPatternFlag == 2:
        Dst = Dst[1:height-1, 1:width-1, :]
    elif BayerPatternFlag == 3:
        Dst = Dst[1:height-1, :, :]

    # BGR to RGB
    Dst = Dst[..., ::-1]

    return Dst.copy()

if __name__ == "__main__":
    input_image_path = "../imgs/real_bayer_img.npy"
    # output_image_path = "./my_gbtf_out_python_fast.bmp"

    # load real bayer image
    bayer_img_norm = np.load(input_image_path).astype(np.float64)  # HxW np.float64
    print("bayer_img_norm shape:", bayer_img_norm.shape)
    print("bayer_img_norm dtype:", bayer_img_norm.dtype)
    bayer_img_norm = bayer_img_norm.astype(np.float64)

    # run GBTF CFA interpolation
    start_time = time.time()
    gbtf_img = GBTF_CFAInterpolation_Fast(bayer_img_norm, BayerPatternFlag=2)
    end_time = time.time()
    
    print(f"GBTF CFA Interpolation Time: {end_time - start_time:.4f} seconds")
    print("gbtf_img shape:", gbtf_img.shape)
    
    gbtf_img = gbtf_img[..., ::-1]  # Convert RGB to BGR for OpenCV
    cv2.imwrite("./my_gbtf_out_python_fast.bmp", (gbtf_img * 255).astype(np.uint8))