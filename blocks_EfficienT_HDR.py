import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualEmbedding(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), # Depthwise
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)
    
class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention 
    Computes attention across channels rather than pixels for efficiency.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Transposed Attention: (C/head x HW) @ (HW x C/head) -> (C/head x C/head)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out

class EfficienT_HDRBlock(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        # Dual-branch: Inverted Residual for efficiency + MDTA for global HDR context
        self.local_branch = InvertedResidualEmbedding(dim)
        self.global_branch = MDTA(dim, num_heads) # Reusing your Restormer MDTA
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # EfficienT-HDR often uses a parallel fusion approach
        local_feat = self.local_branch(x)
        
        # Global context (Transformer)
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        global_feat = self.global_branch(self.norm(x_flat).reshape(b, h, w, c).permute(0, 3, 1, 2))
        
        return local_feat + global_feat