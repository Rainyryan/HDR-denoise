import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from blocks_SwinIR import *


##########################################################################
## Lightweight Resizing modules
##########################################################################
class LightweightDownsample(nn.Module):
	"""Depthwise-separable downsampling"""
	def __init__(self, n_feat):
		super(LightweightDownsample, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, groups=n_feat, bias=False),  # Depthwise
			nn.Conv2d(n_feat, n_feat//2, kernel_size=1, bias=False),  # Pointwise
			nn.PixelUnshuffle(2)
		)

	def forward(self, x):
		return self.body(x)

class LightweightUpsample(nn.Module):
	"""Depthwise-separable upsampling"""
	def __init__(self, n_feat):
		super(LightweightUpsample, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(n_feat, n_feat*2, kernel_size=1, bias=False),  # Pointwise
			nn.PixelShuffle(2),
			nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, groups=n_feat, bias=False)  # Depthwise
		)

	def forward(self, x):
		return self.body(x)


##########################################################################
## Lightweight Exposure sharing modules
##########################################################################
class LightweightExposhare(nn.Module):
	"""Simplified exposure sharing without deformable conv"""
	def __init__(self, dim):
		super(LightweightExposhare, self).__init__()
		# Use depthwise-separable convs instead of deformable convs
		self.conv1 = nn.Sequential(
			nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),  # Depthwise
			nn.Conv2d(dim, dim*2, kernel_size=1, bias=False)  # Pointwise
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=False),  # Depthwise
			nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)  # Pointwise
		)

	def forward(self, x):
		z = F.gelu(self.conv1(x))
		z = self.conv2(z)
		return x + z


##########################################################################
## Lightweight Attention Block (Linear Attention)
##########################################################################
class LinearAttention(nn.Module):
	"""Efficient linear attention mechanism using depthwise conv"""
	def __init__(self, dim, num_heads=1, window_size=8):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.window_size = window_size
		
		# Simplified: Use depthwise conv for spatial mixing instead of full attention
		# This is much more parameter-efficient
		self.spatial_conv = nn.Sequential(
			nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),  # Depthwise
			nn.Conv2d(dim, dim, kernel_size=1, bias=False)  # Pointwise
		)
		self.channel_attn = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
			nn.Sigmoid()
		)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

	def forward(self, x):
		# Spatial mixing via depthwise conv
		out = self.spatial_conv(x)
		
		# Channel attention
		ca = self.channel_attn(out)
		out = out * ca
		
		# Final projection
		out = self.proj(out)
		
		return out


##########################################################################
## Lightweight Transformer Block
##########################################################################
class LightweightTransformerBlock(nn.Module):
	"""Lightweight alternative to SwinTransformerBlock"""
	def __init__(self, dim, num_heads=1, window_size=8, mlp_ratio=2):
		super().__init__()
		self.dim = dim
		
		# Use linear attention instead of window attention
		self.attn = LinearAttention(dim, num_heads, window_size)
		self.norm1 = nn.LayerNorm(dim)
		
		# Lightweight MLP with smaller ratio
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim, mlp_hidden_dim, kernel_size=1, bias=False),
			nn.GELU(),
			nn.Conv2d(mlp_hidden_dim, dim, kernel_size=1, bias=False)
		)
		self.norm2 = nn.LayerNorm(dim)

	def forward(self, x):
		B, C, H, W = x.shape
		
		# Attention
		shortcut = x
		x_norm = x.permute(0, 2, 3, 1).contiguous()  # B H W C
		x_norm = self.norm1(x_norm)
		x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # B C H W
		x = x + self.attn(x_norm)
		
		# MLP
		x_norm = x.permute(0, 2, 3, 1).contiguous()  # B H W C
		x_norm = self.norm2(x_norm)
		x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # B C H W
		x = x + self.mlp(x_norm)
		
		return x


##########################################################################
## Lightweight HDR Model
##########################################################################
class HDR_model(nn.Module):
	def __init__(self, 
		out_channels=3, 
		dim=8,  # 8, Reduced from 32
		num_blocks=[2,2,1,1],  #[2,2,1,1] Reduced from [4,4,4,4]
		num_refinement_blocks=1,  # Reduced from 4
		heads=[1,1,1,1],  # Reduced from [1,2,4,8]
		bias=False,
		window_size=8,
		mlp_ratio=2,  # Reduced from 4
		max_channels_factor=2  # Cap channels at dim*2 instead of dim*16
	):
		super(HDR_model, self).__init__()
		self.dim = dim
		max_channels = dim * max_channels_factor

		# ======== First Layer ========
		self.demosaic_unshuffle = nn.PixelUnshuffle(2)
		self.patch_embed = nn.Conv2d(12, dim, kernel_size=3, stride=1, padding=1, bias=bias)

		# ======== Encoder ========
		self.encoder_level_1 = nn.Sequential(*[
			LightweightTransformerBlock(dim=dim, num_heads=heads[0], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(num_blocks[0])
		])
		self.x_expo_1 = LightweightExposhare(dim)

		## From Level 1 to Level 2
		self.down_unshuffle_1_2 = nn.PixelUnshuffle(2)
		level2_dim = min(dim * 4, max_channels)
		self.encoder_level_2 = nn.Sequential(*[
			LightweightTransformerBlock(dim=level2_dim, num_heads=heads[1], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(num_blocks[1])
		])
		self.x_expo_2 = LightweightExposhare(level2_dim)

		## From Level 2 to Latent
		self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)
		latent_dim = min(dim * 16, max_channels)  # Cap at max_channels

		# ======== Latent (minimal) ========
		self.latent = nn.Sequential(*[
			LightweightTransformerBlock(dim=latent_dim, num_heads=heads[2], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(num_blocks[2])
		])
		self.latent_fusion = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=bias)

		# ======== Decoder ========
		## From Latent to Level 2
		self.up_shuffle_3_2 = nn.PixelShuffle(2)  # latent_dim -> level2_dim
		decoder2_dim = min(level2_dim * 2, max_channels)
		
		## Level 2 Decoder
		self.decoder_level_2 = nn.Sequential(*[
			LightweightTransformerBlock(dim=decoder2_dim, num_heads=heads[1], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(num_blocks[1])
		])
		self.reduce_chan_level_2 = nn.Conv2d(decoder2_dim, level2_dim, kernel_size=1, bias=bias)

		## From Level 2 to Level 1
		self.up_shuffle_2_1 = nn.PixelShuffle(2)  # level2_dim -> dim
		decoder1_dim = dim * 2
		
		## Level 1 Decoder
		self.decoder_level_1 = nn.Sequential(*[
			LightweightTransformerBlock(dim=decoder1_dim, num_heads=heads[0], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(num_blocks[0])
		])
		
		# ======== Final blocks ========
		self.decoder_level_0 = nn.Sequential(*[
			LightweightTransformerBlock(dim=decoder1_dim, num_heads=heads[0], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(num_refinement_blocks)
		])
		self.up_shuffle_1_0 = nn.PixelShuffle(2)  # decoder1_dim -> dim

		# ======== Final full-res blocks (minimal) ========
		final_dim = dim // 2
		self.refinement_trans = nn.Sequential(*[
			LightweightTransformerBlock(dim=final_dim, num_heads=heads[0], window_size=window_size, mlp_ratio=mlp_ratio) 
			for i in range(min(num_blocks[0], 2))  # Max 2 blocks
		])
		# Lightweight refinement conv
		self.refinement_conv = nn.Sequential(
			nn.Conv2d(final_dim, final_dim, kernel_size=3, bias=bias, padding=1, groups=final_dim),  # Depthwise
			nn.Conv2d(final_dim, final_dim, kernel_size=1, bias=bias)  # Pointwise
		)
		self.reduce_chan_final = nn.Conv2d(final_dim, out_channels, kernel_size=1, bias=bias)

	def forward(self, x):
		# x: B x 3 x H x W (demosaicked input, same as original model's xm input)
		# Note: Original model expects B x 1 x H x W but training uses xm (demosaicked)
		# We'll handle 3-channel input by converting to single channel for processing
		
		# Convert 3-channel RGB to single channel (luminance) for unshuffle
		# Using standard RGB->luminance weights
		x_single = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])  # B x 1 x H x W

		# Encoder level 0
		x_shuffled = self.demosaic_unshuffle(x_single)  # B x 12 x H/2 x W/2
		x_level1 = self.patch_embed(x_shuffled)  # B x dim x H/2 x W/2

		# Encoder level 1
		x_level1 = self.encoder_level_1(x_level1)
		x_level1 = self.x_expo_1(x_level1)

		# Encoder level 2
		x_level2 = self.down_unshuffle_1_2(x_level1)  # B x level2_dim x H/4 x W/4
		x_level2 = self.encoder_level_2(x_level2)
		x_level2 = self.x_expo_2(x_level2)

		# Encoder level latent
		x_latent = self.down_unshuffle_2_3(x_level2)  # B x latent_dim x H/8 x W/8
		x_latent = self.latent(x_latent)
		x_latent = self.latent_fusion(x_latent)

		# Decoder level 2
		w_level2 = self.up_shuffle_3_2(x_latent)  # B x level2_dim x H/4 x W/4
		w_level2 = torch.cat([w_level2, x_level2], dim=1)  # B x decoder2_dim x H/4 x W/4
		w_level2 = self.decoder_level_2(w_level2)
		w_level2 = self.reduce_chan_level_2(w_level2)  # B x level2_dim x H/4 x W/4

		# Decoder level 1
		w_level1 = self.up_shuffle_2_1(w_level2)  # B x dim x H/2 x W/2
		w_level1 = torch.cat([w_level1, x_level1], dim=1)  # B x decoder1_dim x H/2 x W/2
		w_level1 = self.decoder_level_1(w_level1)

		# Decoder level 0
		w_level0 = self.decoder_level_0(w_level1)
		w_level0 = self.up_shuffle_1_0(w_level0)  # B x final_dim x H x W
		
		# Refinement
		w_level0 = self.refinement_trans(w_level0)
		w_level0 = self.refinement_conv(w_level0)
		
		# Add residual connection (x is already 3-channel demosaicked input)
		w_out = self.reduce_chan_final(w_level0) + x
		
		return w_out.clamp(min=1/(2**20-1))


if __name__ == "__main__":
	# Test model
	model = HDR_model(dim=8, num_blocks=[2,2,1,1], num_refinement_blocks=1)
	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	
	print(f"Total params: {total_params:,}")
	print(f"Trainable params: {trainable_params:,}")
	print(f"Model size (MB, float32): {total_params * 4 / 1024 / 1024:.2f}")
	
	# Test forward pass
	x = torch.randn(1, 3, 128, 128)
	with torch.no_grad():
		y = model(x)
	print(f"Input shape: {x.shape}, Output shape: {y.shape}")
