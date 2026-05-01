import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Efficient Sub-Modules
# ---------------------------------------------------------

class LightweightCNNBlock(nn.Module):
    """
    A fast, inverted-bottleneck style CNN block to replace Restormer 
    at high spatial resolutions. Uses Depthwise-Separable Convs.
    """
    def __init__(self, dim, expansion_ratio=2):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.pw1(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        return res + x

class FastExposhare(nn.Module):
    """
    Replaces heavy Deformable Convs with large-kernel Depthwise Convs
    to align features across exposures efficiently.
    """
    def __init__(self, dim):
        super().__init__()
        # Using 5x5 depthwise to get a larger receptive field without DCN overhead
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pw1 = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.pw2 = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.act(self.dwconv1(x))
        x = self.act(self.dwconv2(x))
        x = self.act(self.pw1(x))
        x = self.pw2(x)
        return res + x

# ---------------------------------------------------------
# 2. Existing Restormer Modules (Trimmed)
# ---------------------------------------------------------
# (Assume MDTA and GDFN are defined here as in your original blocks_Restormer.py)
from blocks_Restormer import MDTA, GDFN, LayerNorm

class TrimmedRestormerBlock(nn.Module):
    """Restormer block with a customizable, usually lower, MLP ratio."""
    def __init__(self, dim, num_heads, mlp_ratio=1.5, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, mlp_ratio, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------------------------------------------------
# 3. The Hybrid Model
# ---------------------------------------------------------

class Hybrid_Student_HDR(nn.Module):
    def __init__(self, out_channels=3, dim=8, num_blocks=[2,2,2,1], num_refinement_blocks=2, heads=[1,2,4,8]):
        super().__init__()
        self.dim = dim

        # First Layer
        self.demosaic_unshuffle = nn.PixelUnshuffle(2)
        self.patch_embed = nn.Conv2d(12, dim, kernel_size=3, stride=1, padding=1)

        # ======== Encoder ========
        # Level 1: HIGH RES -> Use Lightweight CNN
        self.encoder_level_1 = nn.Sequential(*[
            LightweightCNNBlock(dim) for _ in range(num_blocks[0])
        ])
        self.x_expo_1 = FastExposhare(dim)

        # Level 2: MID RES -> Use Lightweight CNN
        self.down_unshuffle_1_2 = nn.PixelUnshuffle(2) 	
        self.encoder_level_2 = nn.Sequential(*[
            LightweightCNNBlock(int(dim*4)) for _ in range(num_blocks[2])
        ])
        self.x_expo_2 = FastExposhare(int(dim*4))

        # ======== Latent ========
        # LOW RES -> Use Transformers (Restormer) for global context
        self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)
        self.latent = nn.Sequential(*[
            TrimmedRestormerBlock(dim=int(dim*16), num_heads=heads[3], mlp_ratio=1.5) 
            for _ in range(num_blocks[3])
        ])
        self.latent_fusion = nn.Conv2d(int(dim*16), int(dim*16), kernel_size=1)

        # ======== Decoder ========
        # Level 2 Decoder: MID RES -> CNN
        self.up_shuffle_3_2 = nn.PixelShuffle(2)	
        self.decoder_level_2 = nn.Sequential(*[
            LightweightCNNBlock(int(dim*8)) for _ in range(num_blocks[2])
        ])
        self.reduce_chan_level_2 = nn.Conv2d(int(dim*8), int(dim*4), kernel_size=1)

        # Level 1 Decoder: HIGH RES -> CNN
        self.up_shuffle_2_1 = nn.PixelShuffle(2)	
        self.decoder_level_1 = nn.Sequential(*[
            LightweightCNNBlock(int(dim*2)) for _ in range(num_blocks[1])
        ])

        # Level 0 Refinement: HIGH RES -> CNN (Reduced from 4 blocks to 2)
        self.decoder_level_0 = nn.Sequential(*[
            LightweightCNNBlock(int(dim*2)) for _ in range(num_refinement_blocks) 
        ])
        self.up_shuffle_1_0 = nn.PixelShuffle(2)  

        # Final Refinement
        self.refinement_trans = LightweightCNNBlock(int(dim//2))
        self.refinement_conv = nn.Conv2d(int(dim//2), int(dim//2), kernel_size=3, padding=1)
        self.reduce_chan_final = nn.Conv2d(int(dim//2), out_channels, kernel_size=1)

    def forward(self, x, return_maps=False):
        # Encoder
        x_shuffled = self.demosaic_unshuffle(x) 			
        x_level1 = self.patch_embed(x_shuffled) 			
        x_level1 = self.encoder_level_1(x_level1)
        x_level1 = self.x_expo_1(x_level1) 				

        x_level2 = self.down_unshuffle_1_2(x_level1)		
        x_level2 = self.encoder_level_2(x_level2)
        x_level2 = self.x_expo_2(x_level2) 				

        # Latent 
        x_latent = self.down_unshuffle_2_3(x_level2)
        x_latent = self.latent(x_latent)
        x_latent = self.latent_fusion(x_latent)
        
        # Decoder
        w_level2 = self.up_shuffle_3_2(x_latent) 			
        w_level2 = torch.cat([w_level2, x_level2], dim=1) 	
        w_level2 = self.decoder_level_2(w_level2)
        w_level2 = self.reduce_chan_level_2(w_level2) 		

        w_level1 = self.up_shuffle_2_1(w_level2) 			
        w_level1 = torch.cat([w_level1, x_level1], dim=1) 	
        w_level1 = self.decoder_level_1(w_level1) 			

        w_level0 = self.decoder_level_0(w_level1) 			
        
        w_level0 = self.up_shuffle_1_0(w_level0) 			
        
        # Refinement
        w_level0 = self.refinement_trans(w_level0) 		
        w_level0 = self.refinement_conv(w_level0) 	
        
        # Save the map after pixel unshuffle and before final channel reduction for distillation
        w_level0_refined_map = w_level0
        		
        w_out = self.reduce_chan_final(w_level0) + x 		
        
        output = w_out.clamp(min=1/(2**20-1))

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