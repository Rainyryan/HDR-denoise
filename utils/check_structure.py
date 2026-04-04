import torch

checkpoint_path = "./models_rest_300_999_hi_noise_dim32_4444/preexpand_hdr_best.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Use 'model_state_dict' if your saving script wrapped it, otherwise use checkpoint itself
state_dict = checkpoint.get('model_state_dict', checkpoint)

# 1. Check Dimensions (Look at the first convolution or embedding layer)
# If it's a Restormer/Transformer, 'patch_embed.weight' is usually the first layer
if 'patch_embed.weight' in state_dict:
    dim = state_dict['patch_embed.weight'].shape[0]
    print(f"Detected Model Dim: {dim}")

# 2. Check num_blocks (Count unique block indices in each level)
levels = ['encoder_level_1', 'encoder_level_2', 'latent', 'decoder_level_2', 'decoder_level_1']
num_blocks = []

for level in levels:
    # Find all keys belonging to this level and extract the block index (e.g., 'level_1.3.norm')
    block_indices = set()
    for key in state_dict.keys():
        if key.startswith(level):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    
    if block_indices:
        num_blocks.append(len(block_indices))

print(f"Detected num_blocks: {num_blocks}")