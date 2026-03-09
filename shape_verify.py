import torch
# Assuming your files are named blocks_SwinIR.py and blocks_Restormer.py
from blocks_SwinIR import SwinTransformerBlock
from blocks_Restormer import RestormerBlock
from blocks_RepUNet import RepUNetBlock
from blocks_EfficienT_HDR import EfficienT_HDRBlock
from blocks_CoDe import CODEBlock

def test_shape_consistency():
    # Parameters from your SwinIR implementation
    dim = 64
    num_heads = 8
    window_size = 8
    
    # Test cases: (Batch, Channels, Height, Width)
    # We test multiple resolutions to ensure Restormer's linear scaling works
    test_resolutions = [
        (1, dim, 64, 64),
        (2, dim, 128, 128),
        (1, dim, 256, 256)
    ]
    
    # Initialize both blocks
    # Note: RestormerBlock ignores window_size/shift_size but accepts them via **kwargs
    swin_block = SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size)
    restormer_block = RestormerBlock(dim=dim, num_heads=num_heads, window_size=window_size)
    repunet_block = RepUNetBlock(dim=dim, num_heads=num_heads)
    code_block = CODEBlock(dim=dim, num_heads=num_heads)
    efficient_hdr_block = EfficienT_HDRBlock(dim=dim, num_heads=num_heads)
    
    print(f"{'Resolution':<20} | {'SwinIR Output':<20} | {'Restormer Output':<20} | {'Match'}")
    print("-" * 85)

    for shape in test_resolutions:
        x = torch.randn(shape)
        
        # Get outputs
        with torch.no_grad():
            out_swin = swin_block(x)
            out_test = code_block(x)
        
        shape_str = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"
        swin_shape = list(out_swin.shape)
        rest_shape = list(out_test.shape)
        match = swin_shape == rest_shape
        
        print(f"{shape_str:<20} | {str(swin_shape):<20} | {str(rest_shape):<20} | {match}")
        
        if not match:
            raise ValueError(f"Shape mismatch at resolution {shape}!")

    print("\n[SUCCESS] TestBlock maintains exact shape consistency with SwinIR.")

if __name__ == "__main__":
    test_shape_consistency()