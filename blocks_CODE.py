import torch
import torch.nn as nn
import torch.nn.functional as F

class CondensedAttention(nn.Module):
    def __init__(self, dim, num_heads, ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.ratio = ratio # Downsampling ratio for 'superpixel' condensation
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(ratio) # Condense to superpixels
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).reshape(b, self.num_heads, -1, h * w) # [B, Head, C/H, HW]
        
        # Condense Key and Value to save memory
        kv = self.pool(x)
        kv = self.kv(kv).reshape(b, self.num_heads, -1, self.ratio * self.ratio)
        k, v = kv.chunk(2, dim=2)

        # Standard Scaled Dot-Product Attention on condensed space
        attn = (q.transpose(-2, -1) @ k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (k @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        return self.proj(out)

class CODEBlock(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        self.attn = CondensedAttention(dim, num_heads)
        self.norm = nn.GroupNorm(1, dim) # Efficient norm for HDR
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(x)
        return x