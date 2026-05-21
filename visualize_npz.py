import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from my_GBTF import GBTF_CFAInterpolation as GBTF


# 1. Find the first available training file
file_paths = glob.glob('/scratch/gilbreth/chen4848/datasets/Mobile-HDR/train/tensors/**/*.npz', recursive=True)
if not file_paths:
    raise FileNotFoundError("Could not locate any .npz training files in the specified path.")
file_path = file_paths[120] # 121 files in static, 97 files in dynamic
print(f"📁 Processing File: {file_path}")

# 2. Load the packed arrays
with np.load(file_path) as data:
    sht = data['sht'].astype(np.float32)
    mid = data['mid'].astype(np.float32)
    lng = data['lng'].astype(np.float32)
    hdr = data['hdr'].astype(np.float32)

def unpack_raw_to_mosaic(packed_4ch):
    """
    Takes a (4, 1728, 2304) packed tensor in sequential B-G1-G2-R order
    and reconstructs the full resolution (3456, 4608) 2D Bayer mosaic image.
    Outputs a native, unshifted BGGR layout grid.
    """
    _, h, w = packed_4ch.shape
    mosaic = np.zeros((h * 2, w * 2), dtype=packed_4ch.dtype)
    
    # Map back according to the CVPR2023 sequential layer specifications
    # Top-left block mosaic map pattern:
    # [B,  G1]
    # [G2, R ]
    mosaic[0::2, 0::2] = packed_4ch[0]  # Channel 0: Blue (B)
    mosaic[0::2, 1::2] = packed_4ch[1]  # Channel 1: Green 1 (G1)
    mosaic[1::2, 0::2] = packed_4ch[2]  # Channel 2: Green 2 (G2)
    mosaic[1::2, 1::2] = packed_4ch[3]  # Channel 3: Red (R)
    
    return mosaic

def process_safe_channel(img):
    """
    Applies a clean logarithmic dynamic range compression curve.
    Bypasses buggy OpenCV mappers to restore true, vibrant contrast.
    """
    # 1. Ensure it's in channels-last format (H, W, 3) for processing
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        
    img = img.astype(np.float32)
    
    # 2. Use a robust, non-linear logarithmic curve to pull highlights down
    # and push deep shadow midtones up naturally
    # (Adjust the multiplier 100.0 up if the image is too dark, down if too bright)
    tonemapped = np.log1p(img * 100.0) / np.log1p(100.0)
    
    # 3. Apply standard sRGB Display Gamma correction (1/2.2)
    gamma_corrected = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)
    
    return gamma_corrected

def process_and_demosaic(packed_4ch):
    """
    Reconstructs full Bayer arrays and applies optimized GBTF demosaicing.
    """
    # Step A: Reconstruct full resolution Bayer 2D matrix
    mosaic_2d = unpack_raw_to_mosaic(packed_4ch)
    
    # Step B: Run optimized GBTF script passing Flag 3 (Native BGGR Layout)
    rgb_img = GBTF(mosaic_2d, BayerPatternFlag=3)
    
    return rgb_img

if __name__ == "__main__":
    # 3. Process all required image segments
    print("⚙️ Unpacking and demosaicing exposure stacks...")

    # Group 1: Original LDR Data (Channels 0 to 3)
    img_sht_ldr = process_and_demosaic(sht[0:4])
    img_mid_ldr = process_and_demosaic(mid[0:4])
    img_lng_ldr = process_and_demosaic(lng[0:4])

    # Group 2: Luminance-Aligned Data (Channels 4 to 7)
    img_sht_align = process_and_demosaic(sht[4:8])
    img_mid_align = process_and_demosaic(mid[4:8])
    img_lng_align = process_and_demosaic(lng[4:8])

    # Target reference HDR
    img_hdr_gt = process_and_demosaic(hdr)

    # 4. Create the final 7-image grid layout
    print("📊 Generating 7-image comparison canvas...")
    fig = plt.figure(figsize=(20, 11))

    # Setup grid configurations
    # Row 0: Original Native LDR Frames
    # Row 1: Brightness Equalized/Aligned Frames + Final Target HDR
    axes_layout = [
        (2, 4, 1, img_sht_ldr, "1. Short Exposure\n(Original Native LDR)"),
        (2, 4, 2, img_mid_ldr, "2. Medium Exposure\n(Original Native LDR Baseline)"),
        (2, 4, 3, img_lng_ldr, "3. Long Exposure\n(Original Native LDR Saturated)"),
        
        (2, 4, 5, img_sht_align, "4. Short Exposure\n(Luminance-Aligned)"),
        (2, 4, 6, img_mid_align, "5. Medium Exposure\n(Luminance-Aligned)"),
        (2, 4, 7, img_lng_align, "6. Long Exposure\n(Luminance-Aligned)"),
        
        (2, 4, 8, img_hdr_gt, "7. Ground-Truth HDR\n(Target Frame Result)")
    ]

    for rows, cols, index, img_data, title_text in axes_layout:
        ax = fig.add_subplot(rows, cols, index)
        
        # Step C: Run high-fidelity local tone-mapping to lift shadows without clipping highlights
        vis_img = process_safe_channel(img_data)
        
        ax.imshow(vis_img)
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    # Leave spot 4 empty to create a clean separation layout
    fig.add_subplot(2, 4, 4).axis('off')

    plt.tight_layout()
    output_name = "mobile_hdr_7way_visualization.png"
    plt.savefig(output_name, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 Success! All 7 full-color images have been written to: {output_name}")