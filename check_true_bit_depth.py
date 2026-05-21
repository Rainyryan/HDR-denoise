import numpy as np
import cv2
import os

def analyze_array_depth(arr, channel_name):
    """
    Computes unique values, math step size (LSB), and physical dynamic range
    to reverse-engineer original integer bit depths from normalized float representations.
    """
    # Flatten and clip absolute zero boundaries to measure cleanly
    valid_pixels = arr[arr > 1e-8]
    if len(valid_pixels) == 0:
        print(f"  [{channel_name}] Array contains no non-zero data.")
        return

    # 1. Measure Continuous vs. Discrete Resolution
    unique_vals = len(np.unique(valid_pixels))
    
    # 2. Extract smallest non-zero gap between adjacent values (LSB step)
    sorted_unique = np.sort(np.unique(valid_pixels))
    diffs = np.diff(sorted_unique)
    positive_diffs = diffs[diffs > 1e-9]
    smallest_step = np.min(positive_diffs) if len(positive_diffs) > 0 else 0
    calculated_multiplier = np.round(1.0 / smallest_step) if smallest_step > 0 else 0

    # 3. Calculate Effective Bit Depth via Dynamic Range
    max_val = np.max(valid_pixels)
    min_val = np.min(valid_pixels)
    dyn_range = max_val / min_val
    effective_bits = np.log2(dyn_range)

    print(f"\n📊 Analysis for [{channel_name}]:")
    print(f"  -> Total Unique Scalar Values: {unique_vals}")
    print(f"  -> Smallest Quantization Step: {smallest_step:.8e}")
    print(f"  -> Extracted Sensor Multiplier: {calculated_multiplier}")
    print(f"  -> Effective Math Dynamic Range: {effective_bits:.2f} bits")

    # Match structural integer targets
    if unique_vals <= 1025 and abs(calculated_multiplier - 1023) < 5:
        print("  🎯 Result: Native 10-bit integer sensor data.")
    elif unique_vals <= 4097 and abs(calculated_multiplier - 4095) < 5:
        print("  🎯 Result: Native 12-bit integer sensor data.")
    elif unique_vals <= 16385 and abs(calculated_multiplier - 16383) < 5:
        print("  🎯 Result: Native 14-bit integer sensor data.")
    else:
        print("  🎯 Result: Fused continuous float maps (Highly continuous floating-point HDR workspace).")


# ==========================================
# PART 1: KALANTARI 2017 AUDIT (.hdr file)
# ==========================================
path_kalantari = '/scratch/gilbreth/chen4848/datasets/kalantari2017/train/16-09-28-01/ref_hdr_aligned.hdr'

print("=" * 60)
print(f"📁 Loading Kalantari 2017: {os.path.basename(path_kalantari)}")
if os.path.exists(path_kalantari):
    # Must use cv2.imread with IMREAD_UNCHANGED to prevent truncating floating-point precision
    img_kalantari = cv2.imread(path_kalantari, cv2.IMREAD_UNCHANGED)
    # Run the calculator on a single color channel (Green Channel)
    analyze_array_depth(img_kalantari[..., 1], "Kalantari Ground-Truth HDR")
else:
    print(f"❌ File not found at: {path_kalantari}")


# ==========================================
# PART 2: MOBILE-HDR AUDIT (.npz file)
# ==========================================
path_mobile = '/scratch/gilbreth/chen4848/datasets/Mobile-HDR/train/tensors/static/raw-2022-0208-1117-1076.npz'

print("\n" + "=" * 60)
print(f"📁 Loading Mobile-HDR: {os.path.basename(path_mobile)}")
if os.path.exists(path_mobile):
    with np.load(path_mobile) as data:
        # Check an un-aligned native short exposure channel to find the sensor hardware specs
        sht_native_channel = data['sht'][0].astype(np.float64)
        analyze_array_depth(sht_native_channel, "Mobile-HDR Native Short Exposure (SHT[0])")
        
        # Check the final merged ground truth frame to observe the fused dynamic range
        hdr_gt_channel = data['hdr'][0].astype(np.float64)
        analyze_array_depth(hdr_gt_channel, "Mobile-HDR Ground-Truth Target (HDR[0])")
else:
    print(f"❌ File not found at: {path_mobile}")
print("=" * 60)