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
    if borderType == 'BORDER_REFLECT_101': 
        return np.pad(src, ((top, bottom), (left, right)), mode='reflect')
    elif borderType == 'BORDER_CONSTANT':
        return np.pad(src, ((top, bottom), (left, right)), mode='constant', constant_values=value)
    else:
        raise NotImplementedError(f"Mode {borderType} not implemented")

def fast_filter2D(src, kernel):
    return cv2.filter2D(src.astype(np.float64, copy=False), -1, kernel, borderType=cv2.BORDER_REFLECT_101)

def to_3Channel_vectorized(bayer, height, width, BayerPatternFlag=3):
    Dst = np.zeros((height, width, 3), dtype=np.float64)
    
    if BayerPatternFlag == 1:    # RGGB Layout
        Dst[0::2, 0::2, 2] = bayer[0::2, 0::2]
        Dst[0::2, 1::2, 1] = bayer[0::2, 1::2]
        Dst[1::2, 0::2, 1] = bayer[1::2, 0::2]
        Dst[1::2, 1::2, 0] = bayer[1::2, 1::2]

    elif BayerPatternFlag == 2:  # GBRG Layout
        Dst[0::2, 0::2, 1] = bayer[0::2, 0::2]
        Dst[0::2, 1::2, 0] = bayer[0::2, 1::2]
        Dst[1::2, 0::2, 2] = bayer[1::2, 0::2]
        Dst[1::2, 1::2, 1] = bayer[1::2, 1::2]

    elif BayerPatternFlag == 3:  # BGGR Layout
        Dst[0::2, 0::2, 0] = bayer[0::2, 0::2]
        Dst[0::2, 1::2, 1] = bayer[0::2, 1::2]
        Dst[1::2, 0::2, 1] = bayer[1::2, 0::2]
        Dst[1::2, 1::2, 2] = bayer[1::2, 1::2]
        
    return Dst

def GBTF_CFAInterpolation(Bayer, BayerPatternFlag=3):
    Src = Bayer.astype(np.float64, copy=False)
    height, width = Src.shape
    
    Dst = to_3Channel_vectorized(Src, height, width, BayerPatternFlag=BayerPatternFlag)

    # 1. Horizontal and Vertical Color Difference Maps
    k_1d = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=np.float64)
    HK = k_1d.reshape(1, 5)
    VK = k_1d.reshape(5, 1)

    HCDMap = fast_filter2D(Src, HK)
    VCDMap = fast_filter2D(Src, VK)

    # FIX 1: Dynamically assign the checkerboard phase depending on Bayer layout
    checker = np.ones((height, width), dtype=np.float64)
    if BayerPatternFlag in [1, 3]:  # RGGB and BGGR
        checker[0::2, 1::2] = -1.0
        checker[1::2, 0::2] = -1.0
    elif BayerPatternFlag == 2:     # GBRG
        checker[0::2, 0::2] = -1.0
        checker[1::2, 1::2] = -1.0

    # FIX 2: Correct the color difference vector direction (G - Color, NOT Color - G)
    HCDMap = checker * (HCDMap - Src) 
    VCDMap = checker * (VCDMap - Src)

    # 2. Gradient Maps Processing
    HCD_pad = fast_copyMakeBorder(HCDMap, 5, 5, 5, 5, 'BORDER_CONSTANT', value=0)
    VCD_pad = fast_copyMakeBorder(VCDMap, 5, 5, 5, 5, 'BORDER_CONSTANT', value=0)

    HGradientMap = np.abs(HCD_pad[5:5+height, 6:6+width] - HCD_pad[5:5+height, 4:4+width])
    VGradientMap = np.abs(VCD_pad[6:6+height, 5:5+width] - VCD_pad[4:4+height, 5:5+width])

    # 3. Sum Sliding Windows optimized
    H_sum_total = cv2.boxFilter(HGradientMap, -1, (5, 5), normalize=False, borderType=cv2.BORDER_CONSTANT)
    V_sum_total = cv2.boxFilter(VGradientMap, -1, (5, 5), normalize=False, borderType=cv2.BORDER_CONSTANT)

    epsilon = 1e-9
    N_val = np.vstack([V_sum_total[0:2, :], V_sum_total[:-2, :]])
    S_val = np.vstack([V_sum_total[2:, :], V_sum_total[-2:, :]])
    E_val = np.hstack([H_sum_total[:, 0:2], H_sum_total[:, :-2]])
    W_val = np.hstack([H_sum_total[:, 2:], H_sum_total[:, -2:]])
    
    N = 1.0 / (N_val**2 + epsilon)
    S = 1.0 / (S_val**2 + epsilon)
    E = 1.0 / (E_val**2 + epsilon)
    W = 1.0 / (W_val**2 + epsilon)
    
    TotalWeight = N + S + E + W

    # 4. Green Channel Interpolation
    HCD_pad9 = np.pad(HCDMap, ((0, 0), (4, 4)), mode='reflect')
    VCD_pad9 = np.pad(VCDMap, ((4, 4), (0, 0)), mode='reflect')
    
    H_sum_L = (HCD_pad9[:, 0:width] + HCD_pad9[:, 1:width+1] + HCD_pad9[:, 2:width+2] + HCD_pad9[:, 3:width+3])
    H_sum_R = (HCD_pad9[:, 5:width+5] + HCD_pad9[:, 6:width+6] + HCD_pad9[:, 7:width+7] + HCD_pad9[:, 8:width+8])
    V_sum_U = (VCD_pad9[0:height, :] + VCD_pad9[1:height+1, :] + VCD_pad9[2:height+2, :] + VCD_pad9[3:height+3, :])
    V_sum_D = (VCD_pad9[5:height+5, :] + VCD_pad9[6:height+6, :] + VCD_pad9[7:height+7, :] + VCD_pad9[8:height+8, :])

    AccumH = (E * 0.2 * H_sum_L) + ((W + E) * 0.2 * HCDMap) + (W * 0.2 * H_sum_R)
    AccumV = (N * 0.2 * V_sum_U) + ((N + S) * 0.2 * VCDMap) + (S * 0.2 * V_sum_D)
    
    TPdiff = (AccumH + AccumV) / (TotalWeight + epsilon)
    
    # Target only the holes where Green is genuinely missing
    mask_missing_G = np.zeros((height, width), dtype=bool)
    if BayerPatternFlag in [1, 3]:  
        mask_missing_G[0::2, 0::2] = True
        mask_missing_G[1::2, 1::2] = True
    elif BayerPatternFlag == 2:     
        mask_missing_G[0::2, 1::2] = True
        mask_missing_G[1::2, 0::2] = True
    
    G_new = np.clip(Src + TPdiff, 0.0, 1.0)
    
    # Populate the entire Green plane (native + interpolated)
    Dst[..., 1] = np.where(mask_missing_G, G_new, Src)

    # 5. Red and Blue Interpolation on opposing native positions
    Correction = fast_filter2D(TPdiff, Prb_matrix)
    RB_val = np.clip(Dst[..., 1] - Correction, 0.0, 1.0)
    
    if BayerPatternFlag == 3:    # BGGR layout 
        Dst[1::2, 1::2, 0] = RB_val[1::2, 1::2] # Blue interp onto Red locations (1,1)
        Dst[0::2, 0::2, 2] = RB_val[0::2, 0::2] # Red interp onto Blue locations (0,0)
    elif BayerPatternFlag == 1:  # RGGB layout 
        Dst[0::2, 0::2, 0] = RB_val[0::2, 0::2] # Blue interp onto Red locations (0,0)
        Dst[1::2, 1::2, 2] = RB_val[1::2, 1::2] # Red interp onto Blue locations (1,1)
    elif BayerPatternFlag == 2:  # GBRG layout 
        Dst[1::2, 0::2, 0] = RB_val[1::2, 0::2] # Blue interp onto Red locations (1,0)
        Dst[0::2, 1::2, 2] = RB_val[0::2, 1::2] # Red interp onto Blue locations (0,1)

    # 6. Red and Blue Interpolation on native Green grid sites
    GmB = Dst[..., 1] - Dst[..., 0]
    GmR = Dst[..., 1] - Dst[..., 2]
    
    k_cross = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=np.float64)
    
    GmB_smooth = fast_filter2D(GmB, k_cross)
    GmR_smooth = fast_filter2D(GmR, k_cross)
    
    B_rec = np.clip(Dst[..., 1] - GmB_smooth, 0.0, 1.0)
    R_rec = np.clip(Dst[..., 1] - GmR_smooth, 0.0, 1.0)
    
    Dst[..., 0] = np.where(~mask_missing_G, B_rec, Dst[..., 0])
    Dst[..., 2] = np.where(~mask_missing_G, R_rec, Dst[..., 2])

    return Dst[..., ::-1].copy()