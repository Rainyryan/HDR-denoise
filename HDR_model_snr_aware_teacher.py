import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from blocks_SwinIR import SwinTransformerBlock
from blocks_Restormer import RestormerBlock
from blocks_RepUNet import RepUNetBlock
from blocks_CODE import CODEBlock
from blocks_EfficienT_HDR import EfficienT_HDRBlock
from dcn.dcn import DeformableConv2d


##########################################################################
## Resizing modules
##########################################################################
class Downsample(nn.Module): # Reduce dimension by 2x, keep num of channel
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module): # Increase dimension by 2x, keep num of channel
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Exposure sharing modules
##########################################################################
class Exposhare(nn.Module):
    def __init__(self, dim):
        super(Exposhare, self).__init__()
        self.dfconv1 = DeformableConv2d(dim, dim*2, kernel_size=3, stride=1, padding=1)
        self.dfconv2 = DeformableConv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1)
        self.dfconv3 = DeformableConv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(dim*2, dim*2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim*2, dim*2, kernel_size=1)
        self.conv3 = nn.Conv2d(dim*2, dim, kernel_size=1)

    def forward(self, x):
        z = F.gelu(self.dfconv1(x))
        z = F.gelu(self.dfconv2(z))
        z = F.gelu(self.dfconv3(z))
        z = F.gelu(self.conv1(z))
        z = F.gelu(self.conv2(z))
        z = self.conv3(z)
        return x + z


class HDR_model(nn.Module):
    def __init__(self, 
        out_channels=3, 
        dim=16, 
        num_blocks=[4,4,4,2], 
        num_refinement_blocks=4, 
        heads=[1,2,4,8],
        bias=False,
        window_size=8
    ):
        super(HDR_model, self).__init__()
        self.dim = dim
        self.block_type = 'Restormer' 

        # ======== First Layer ========
        self.demosaic_unshuffle = nn.PixelUnshuffle(2)
        
        # MODIFIED: Input channels changed from 12 to 16 to accommodate the SNR map
        # (3 RGB + 1 SNR = 4 channels) * 4 (from PixelUnshuffle) = 16 channels
        self.patch_embed = nn.Conv2d(16, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # ======== Encoder ========
        self.encoder_level_1 = nn.Sequential(*[self._build_block(dim=dim, num_heads=heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_blocks[0])])			
        self.x_expo_1 = Exposhare(dim)

        self.down_unshuffle_1_2 = nn.PixelUnshuffle(2) 	
        self.encoder_level_2 = nn.Sequential(*[self._build_block(dim=int(dim*4), num_heads=heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_blocks[2])])			
        self.x_expo_2 = Exposhare(int(dim*4))

        self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)

        # ======== Latent ========
        self.latent = nn.Sequential(*[self._build_block(dim=int(dim*16), num_heads=heads[3], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_blocks[3])])			
        self.latent_fusion = nn.Conv2d(int(dim*16), int(dim*16), kernel_size=1, bias=bias)

        # ======== Decoder ========
        self.up_shuffle_3_2 = nn.PixelShuffle(2)	
        self.decoder_level_2 = nn.Sequential(*[self._build_block(dim=int(dim*8), num_heads=heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_blocks[2])])			
        self.reduce_chan_level_2 = nn.Conv2d(int(dim*8), int(dim*4), kernel_size=1, bias=bias)

        self.up_shuffle_2_1 = nn.PixelShuffle(2)	
        self.decoder_level_1 = nn.Sequential(*[self._build_block(dim=int(dim*2), num_heads=heads[1], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_blocks[1])])			
        
        # ======== Final blocks ========
        self.decoder_level_0 = nn.Sequential(*[self._build_block(dim=int(dim*2), num_heads=heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_refinement_blocks)])	
        self.up_shuffle_1_0 = nn.PixelShuffle(2)  

        self.refinement_trans = nn.Sequential(*[self._build_block(dim=int(dim//2), num_heads=heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_blocks[0])])			
        self.refinement_conv = nn.Sequential(*[nn.Conv2d(int(dim//2), int(dim//2), kernel_size=5, bias=bias, padding="same", padding_mode="reflect") for i in range(num_blocks[0])])			
        self.reduce_chan_final = nn.Conv2d(int(dim//2), out_channels, kernel_size=1, bias=bias)

    # ... [Keep _build_block and _run_sequence_and_collect_attn the same] ...
    def _build_block(self, dim, num_heads, window_size, shift_size):
        if self.block_type == 'Restormer':
            return RestormerBlock(dim=dim, num_heads=num_heads)
        # ... [keep other blocks] ...

    def _run_sequence_and_collect_attn(self, sequence, x, stage_name, t_attn_maps):
        for i, block in enumerate(sequence):
            if self.block_type == 'Restormer':
                x, attn_map = block(x, return_attn=True)
                t_attn_maps[f"{stage_name}_block{i}"] = attn_map
            else:
                x = block(x)
        return x

    # MODIFIED: Added snr_map argument
    def forward(self, x, snr_map=None, return_maps=False, return_transposed_attn=False):
        t_attn_maps = {}

        # ======== SNR MAP GENERATION ========
        # If the dataset didn't provide a perfect physical SNR map, we estimate it.
        # Physics model: Variance = 14 * Signal + ReadNoise(mean ~147.5)
        if snr_map is None:
            # Clamp to prevent negative values under root
            variance = 14.0 * torch.clamp(x, min=0) + 147.5 
            sigma = torch.sqrt(variance)
            # Create a 1-channel grayscale SNR map by averaging across RGB
            snr_map = torch.mean(x / (sigma + 1e-6), dim=1, keepdim=True)
            # Normalize to [0, 1] for stable gradient flow
            snr_map = snr_map / (snr_map.amax(dim=(1,2,3), keepdim=True) + 1e-6)

        # Concatenate 3-channel input with 1-channel SNR map -> 4 channels
        x_cat = torch.cat([x, snr_map], dim=1)

        # ======== Encoder ========
        x_shuffled = self.demosaic_unshuffle(x_cat) # 4 channels * 4 = 16 channels
        x_level1 = self.patch_embed(x_shuffled) 			
        
        if return_transposed_attn:
            x_level1 = self._run_sequence_and_collect_attn(self.encoder_level_1, x_level1, "enc1", t_attn_maps)
        else:
            x_level1 = self.encoder_level_1(x_level1)
            
        x_level1 = self.x_expo_1(x_level1) 				

        # ... [The rest of the forward pass remains exactly the same] ...
        x_level2 = self.down_unshuffle_1_2(x_level1)		
        if return_transposed_attn:
            x_level2 = self._run_sequence_and_collect_attn(self.encoder_level_2, x_level2, "enc2", t_attn_maps)
        else:
            x_level2 = self.encoder_level_2(x_level2)
        x_level2 = self.x_expo_2(x_level2) 				

        x_latent = self.down_unshuffle_2_3(x_level2)
        if return_transposed_attn:
            x_latent = self._run_sequence_and_collect_attn(self.latent, x_latent, "latent", t_attn_maps)
        else:
            x_latent = self.latent(x_latent)
        x_latent = self.latent_fusion(x_latent)
        
        w_level2 = self.up_shuffle_3_2(x_latent) 			
        w_level2 = torch.cat([w_level2, x_level2], dim=1) 	
        if return_transposed_attn:
            w_level2 = self._run_sequence_and_collect_attn(self.decoder_level_2, w_level2, "dec2", t_attn_maps)
        else:
            w_level2 = self.decoder_level_2(w_level2)
        w_level2 = self.reduce_chan_level_2(w_level2) 		

        w_level1 = self.up_shuffle_2_1(w_level2) 			
        w_level1 = torch.cat([w_level1, x_level1], dim=1) 	
        if return_transposed_attn:
            w_level1 = self._run_sequence_and_collect_attn(self.decoder_level_1, w_level1, "dec1", t_attn_maps)
        else:
            w_level1 = self.decoder_level_1(w_level1) 			

        w_level0 = self.decoder_level_0(w_level1) 			
        if return_transposed_attn:
            w_level0 = self._run_sequence_and_collect_attn(self.decoder_level_0, w_level0, "dec0", t_attn_maps)
        else:
            w_level0 = self.decoder_level_0(w_level0)
            
        w_level0 = self.up_shuffle_1_0(w_level0) 			
        
        if return_transposed_attn:
            w_level0 = self._run_sequence_and_collect_attn(self.refinement_trans, w_level0, "refine_trans", t_attn_maps)
        else:
            w_level0 = self.refinement_trans(w_level0) 		
            
        w_level0 = self.refinement_conv(w_level0) 			
        w_out = self.reduce_chan_final(w_level0) + x 		
        
        output = w_out.clamp(min=1/(2**20-1))
        
        if return_maps:
            return output, {
                "x_level1": x_level1,
                "x_level2": x_level2,
                "x_latent": x_latent,
                "w_level2": w_level2,
                "w_level1": w_level1,
                "w_level0_refined": w_level0
            }
            
        if return_transposed_attn:
            return output, t_attn_maps
            
        return output