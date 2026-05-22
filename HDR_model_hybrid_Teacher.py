import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the heavy attention blocks from your existing repository
from blocks_Restormer import RestormerBlock

# ---------------------------------------------------------
# 1. Heavy CNN Blocks for Encoder/Decoder
# ---------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """
    A robust, standard Residual Block used in the Teacher's CNN paths.
    Provides strong local feature extraction before the Transformer bottleneck.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.GELU()

    def forward(self, x):
        res = x
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        return self.act2(res + x)

class HeavyExposhare(nn.Module):
    """
    The Teacher's version of Exposhare. Uses multiple dense convolutions 
    to heavily align features across the channel dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        z = self.act(self.conv1(x))
        z = self.act(self.conv2(z))
        z = self.conv3(z)
        return res + z

# ---------------------------------------------------------
# 2. The TransUNet Teacher Model
# ---------------------------------------------------------

class TransUNet_Teacher_HDR(nn.Module):
    def __init__(self, out_channels=4, dim=32, num_blocks=[4, 4, 4, 6], num_refinement_blocks=4, heads=[1, 2, 4, 8]):
        """
        dim=32, num_blocks=[4,4,4,6], and heads=[1,2,4,8] scales this model 
        to roughly ~19.5M parameters, acting as the heavy Teacher.
        Expects (4, H, W) packed Bayer input (B, G1, G2, R channels).
        """
        super().__init__()
        self.dim = dim

        # ======== First Layer (Patch Embedding) ========
        # Input is already packed (4, H, W) Bayer — no demosaic unshuffle needed.
        # PixelUnshuffle(2) on 4-ch input -> 4*4 = 16 channels at H/2, W/2
        self.bayer_unshuffle = nn.PixelUnshuffle(2)
        self.patch_embed = nn.Conv2d(16, dim, kernel_size=3, stride=1, padding=1)

        # ======== CNN Encoder ========
        # Level 1: High Resolution
        self.encoder_level_1 = nn.Sequential(*[
            ResidualConvBlock(dim) for _ in range(num_blocks[0])
        ])
        self.x_expo_1 = HeavyExposhare(dim)

        # Level 2: Mid Resolution
        self.down_unshuffle_1_2 = nn.PixelUnshuffle(2) 	
        self.encoder_level_2 = nn.Sequential(*[
            ResidualConvBlock(int(dim*4)) for _ in range(num_blocks[1])
        ])
        self.x_expo_2 = HeavyExposhare(int(dim*4))

        # ======== Transformer Bottleneck (Latent) ========
        # Level 3: Low Resolution Global Context (TransUNet core)
        self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)
        self.latent = nn.Sequential(*[
            RestormerBlock(dim=int(dim*16), num_heads=heads[3]) 
            for _ in range(num_blocks[3])
        ])
        self.latent_fusion = nn.Conv2d(int(dim*16), int(dim*16), kernel_size=1)

        # ======== CNN Decoder ========
        # Level 2 Decoder: Mid Resolution Fusion
        self.up_shuffle_3_2 = nn.PixelShuffle(2)	
        self.decoder_level_2 = nn.Sequential(*[
            ResidualConvBlock(int(dim*8)) for _ in range(num_blocks[2])
        ])
        self.reduce_chan_level_2 = nn.Conv2d(int(dim*8), int(dim*4), kernel_size=1)

        # Level 1 Decoder: High Resolution Fusion
        self.up_shuffle_2_1 = nn.PixelShuffle(2)	
        self.decoder_level_1 = nn.Sequential(*[
            ResidualConvBlock(int(dim*2)) for _ in range(num_blocks[0])
        ])

        # Level 0 Refinement: Full Resolution
        self.decoder_level_0 = nn.Sequential(*[
            ResidualConvBlock(int(dim*2)) for _ in range(num_refinement_blocks) 
        ])
        self.up_shuffle_1_0 = nn.PixelShuffle(2)  

        # Final Refinement
        self.refinement_conv1 = nn.Conv2d(int(dim//2), int(dim//2), kernel_size=3, padding=1)
        self.act_final = nn.GELU()
        self.refinement_conv2 = nn.Conv2d(int(dim//2), int(dim//2), kernel_size=3, padding=1)
        self.reduce_chan_final = nn.Conv2d(int(dim//2), out_channels, kernel_size=1)

    def forward(self, x, return_maps=False):
        # --- Encoder ---
        # x: (B, 4, H, W) packed Bayer -> (B, 16, H/2, W/2)
        x_shuffled = self.bayer_unshuffle(x)
        x_level1 = self.patch_embed(x_shuffled)
        
        x_level1 = self.encoder_level_1(x_level1)
        x_level1 = self.x_expo_1(x_level1) 				

        x_level2 = self.down_unshuffle_1_2(x_level1)		
        x_level2 = self.encoder_level_2(x_level2)
        x_level2 = self.x_expo_2(x_level2) 				

        # --- Transformer Latent ---
        x_latent = self.down_unshuffle_2_3(x_level2)
        x_latent = self.latent(x_latent)
        x_latent = self.latent_fusion(x_latent)
        
        # --- Decoder ---
        w_level2 = self.up_shuffle_3_2(x_latent) 			
        w_level2 = torch.cat([w_level2, x_level2], dim=1) 	
        w_level2 = self.decoder_level_2(w_level2)
        w_level2 = self.reduce_chan_level_2(w_level2) 		

        w_level1 = self.up_shuffle_2_1(w_level2) 			
        w_level1 = torch.cat([w_level1, x_level1], dim=1) 	
        w_level1 = self.decoder_level_1(w_level1) 			

        w_level0 = self.decoder_level_0(w_level1) 			
        w_level0 = self.up_shuffle_1_0(w_level0) 			
        
        # --- Refinement ---
        w_level0 = self.act_final(self.refinement_conv1(w_level0))
        w_level0 = self.refinement_conv2(w_level0)
        
        # Save map for KD loss before final channel reduction
        w_level0_refined_map = w_level0
        
        w_out = self.reduce_chan_final(w_level0) + x 		
        output = w_out.clamp(min=1/(2**20-1), max=1.0)

        if return_maps:
            return output, {
                "x_level1": x_level1,
                "x_level2": x_level2,
                "x_latent": x_latent,
                "w_level2": w_level2,
                "w_level1": w_level1,
                "w_level0_refined": w_level0_refined_map
            }
            
        return output

# ---------------------------------------------------------
# 3. Dual-SNR Wrapper (Matches your train.py requirements)
# ---------------------------------------------------------

class DualSNRTeacher(nn.Module):
    """
    Wraps the TransUNet Teacher into the SNR routing logic so it acts 
    as a drop-in replacement in your exact training loop.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser_low_snr = TransUNet_Teacher_HDR(**kwargs)
        self.denoiser_high_snr = TransUNet_Teacher_HDR(**kwargs)

    def forward(self, x, snr_map):
        out_low = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)

        blended_out = (1.0 - snr_map) * out_low + snr_map * out_high
        
        return blended_out, out_low, out_high