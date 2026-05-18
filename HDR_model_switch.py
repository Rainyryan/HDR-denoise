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


##---------- HDR_model -----------------------
class HDR_model(nn.Module):
    def __init__(self, 
        out_channels=3, 
        dim = 16, #32
        num_blocks = [4,4,4,2], #[4,4,4,4], #[4,6,6,8], 
        num_refinement_blocks = 4, #4
        heads = [1,2,4,8],

        bias = False,
        window_size = 8
    ):
        super(HDR_model, self).__init__()
        self.dim = dim
        self.block_type = 'Restormer' # Choose from 'SwinIR', 'Restormer', 'CODE', 'RepUNet', or 'EfficienT_HDR'


        # ======== First Layer ========
        self.demosaic_unshuffle = nn.PixelUnshuffle(2)
        self.patch_embed = nn.Conv2d(12, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # ======== Encoder ========
        self.encoder_level_1 = nn.Sequential(*[self._build_block(dim=dim, num_heads=heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
            for i in range(num_blocks[0])])			## OUT: dim
        self.x_expo_1 = Exposhare(dim)

        ## From Level 1.5 to Level 2
        self.down_unshuffle_1_2 = nn.PixelUnshuffle(2) 	
        self.encoder_level_2 = nn.Sequential(*[self._build_block(dim=int(dim*4), num_heads=heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
            for i in range(num_blocks[2])])			## OUT: dim*4
        self.x_expo_2 = Exposhare(int(dim*4))

        ## From Level 2 to Level 3
        self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)

        # ======== Latent ========
        self.latent		 = nn.Sequential(*[self._build_block(dim=int(dim*16), num_heads=heads[3], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
            for i in range(num_blocks[3])])			## OUT: dim*16
        self.latent_fusion = nn.Conv2d(int(dim*16), int(dim*16), kernel_size=1, bias=bias)
                                                    ## IN: dim*16		OUT: dim*16
        # ======== Decoder ========
        ## From Level 3 to Level 2
        self.up_shuffle_3_2 = nn.PixelShuffle(2)	## IN: dim*16			OUT: dim*4
                                # Cat level 2 out	## CAT: (dim*4)			OUT: dim*4 + dim*4 = dim*8
        ## Level 2
        self.decoder_level_2 = nn.Sequential(*[self._build_block(dim=int(dim*8), num_heads=heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)  
            for i in range(num_blocks[2])])			## OUT: dim*8
        self.reduce_chan_level_2 = nn.Conv2d(int(dim*8), int(dim*4), kernel_size=1, bias=bias)
                                                    ## OUT: dim*4
        ## From Level 2 to Level 1
        self.up_shuffle_2_1 = nn.PixelShuffle(2)	## IN: dim*4			OUT: dim
                                # Cat level 1.5 out	## CAT: (dim)			OUT: dim + dim = dim*2
        ## Level 1
        self.decoder_level_1 = nn.Sequential(*[self._build_block(dim=int(dim*2), num_heads=heads[1], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
            for i in range(num_blocks[1])])			## OUT: dim*2
        
        # ======== Final blocks ========
        self.decoder_level_0 = nn.Sequential(*[self._build_block(dim=int(dim*2), num_heads=heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
            for i in range(num_refinement_blocks)])	## OUT: dim*2
        self.up_shuffle_1_0 = nn.PixelShuffle(2)  ## OUT: dim/2

        # ======== Final full-res blocks ========
        self.refinement_trans = nn.Sequential(*[self._build_block(dim=int(dim//2), num_heads=heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) 
            for i in range(num_blocks[0])])			## OUT: dim/2
        self.refinement_conv = nn.Sequential(*[nn.Conv2d(int(dim//2), int(dim//2), kernel_size=5, bias=bias, padding="same", padding_mode="reflect") 
            for i in range(num_blocks[0])])			## OUT: dim/2
        self.reduce_chan_final = nn.Conv2d(int(dim//2), out_channels, kernel_size=1, bias=bias)

    def _build_block(self, dim, num_heads, window_size, shift_size):
        """
        Factory method to safely instantiate blocks based on self.block_type.
        Handles the specific argument requirements of each block architecture.
        """
        if self.block_type == 'SwinIR':
            return SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size)
        
        elif self.block_type == 'Restormer':
            return RestormerBlock(dim=dim, num_heads=num_heads)
            
        elif self.block_type == 'CODE':
            return CODEBlock(dim=dim, num_heads=num_heads)
            
        elif self.block_type == 'RepUNet':
            return RepUNetBlock(dim=dim, num_heads=num_heads)
            
        elif self.block_type == 'EfficienT_HDR':
            return EfficienT_HDRBlock(dim=dim, num_heads=num_heads)
            
        else:
            raise ValueError(f"Unsupported block_type: '{self.block_type}'. Choose from 'SwinIR', 'Restormer', 'CODE', 'RepUNet', or 'EfficienT_HDR'.")

    def _run_sequence_and_collect_attn(self, sequence, x, stage_name, t_attn_maps):
        """
        Runs a tensor through an nn.Sequential of RestormerBlocks 
        and collects the attention map from every single block.
        """
        for i, block in enumerate(sequence):
            # Check if the block is a RestormerBlock to safely pass return_attn
            if self.block_type == 'Restormer':
                x, attn_map = block(x, return_attn=True)
                t_attn_maps[f"{stage_name}_block{i}"] = attn_map
            else:
                x = block(x) # Fallback if you switch to SwinIR/CODE later
        return x

    def forward(self, x, return_maps=False, return_transposed_attn=False):
            # x: B x 1 x H x W
            
            # Dictionary to store the C x C transposed attention maps
            t_attn_maps = {}

            # ======== Encoder ========
            x_shuffled = self.demosaic_unshuffle(x) 			
            x_level1 = self.patch_embed(x_shuffled) 			
            
            if return_transposed_attn:
                x_level1 = self._run_sequence_and_collect_attn(self.encoder_level_1, x_level1, "enc1", t_attn_maps)
            else:
                x_level1 = self.encoder_level_1(x_level1)
                
            x_level1 = self.x_expo_1(x_level1) 				

            x_level2 = self.down_unshuffle_1_2(x_level1)		
            
            if return_transposed_attn:
                x_level2 = self._run_sequence_and_collect_attn(self.encoder_level_2, x_level2, "enc2", t_attn_maps)
            else:
                x_level2 = self.encoder_level_2(x_level2)
                
            x_level2 = self.x_expo_2(x_level2) 				

            # ======== Latent Stage (Bottleneck) ========
            x_latent = self.down_unshuffle_2_3(x_level2)
            
            if return_transposed_attn:
                x_latent = self._run_sequence_and_collect_attn(self.latent, x_latent, "latent", t_attn_maps)
            else:
                x_latent = self.latent(x_latent)
                
            x_latent = self.latent_fusion(x_latent)
            
            # ======== Decoder ========
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
            
            # ======== Refinement ========
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

class DualSNRDenoiser(nn.Module):
    def __init__(self, **kwargs):
        super(DualSNRDenoiser, self).__init__()
        # Expert 1: Specializes in Low SNR (High Noise) regions
        self.denoiser_low_snr = HDR_model(**kwargs)
        
        # Expert 2: Specializes in High SNR (Low Noise) regions
        self.denoiser_high_snr = HDR_model(**kwargs)

    def forward(self, x, snr_map):
        """
        x: [B, C, H, W] noisy input
        snr_map: [B, 1, H, W] normalized between 0 and 1.
                 0 = extremely noisy (low SNR), 1 = clean (high SNR)
        """
        out_low = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)

        # Soft Blend: The SNR map dictates which model's output to trust for each pixel.
        # If snr_map is near 0, weight leans heavily toward out_low.
        # If snr_map is near 1, weight leans heavily toward out_high.
        blended_out = (1.0 - snr_map) * out_low + snr_map * out_high
        
        return blended_out, out_low, out_high
        

