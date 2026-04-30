"""
model/co_scale.py
-----------------
Co-Scale Layer.

Captures multi-scale dependencies between patches by aggregating
patch tokens at multiple spatial resolutions (1×1, 2×2, 4×4 pooling)
and then concatenating their contributions back to each patch token.

Input  : [B, N, d]   patch tokens
Output : [B, N, d]   co-scaled patch tokens  (same shape)

N = num_patches  (e.g. 196 for 224×224 with patch 16)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoScaleLayer(nn.Module):
    """
    Parameters
    ----------
    embed_dim   : int         — feature dimension d
    num_patches : int         — total patches N (must be a perfect square)
    scale_sizes : list[int]   — grid sizes to pool to, e.g. [1, 2, 4]
    dropout     : float
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_patches: int = 196,
        scale_sizes: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if scale_sizes is None:
            scale_sizes = [1, 2, 4]

        self.embed_dim   = embed_dim
        self.num_patches = num_patches
        self.n           = int(math.isqrt(num_patches))
        assert self.n * self.n == num_patches, "num_patches must be a perfect square"
        self.scale_sizes = [s for s in scale_sizes if s <= self.n]

        num_scales = len(self.scale_sizes)

        # One linear projection per scale to map pooled features → d
        self.scale_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in self.scale_sizes
        ])

        # Fusion: combine all scale contributions (num_scales × d → d)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.GELU(),
        )

        # Residual + norm
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, N, d]
        returns : [B, N, d]
        """
        B, N, d = x.shape

        # Reshape to spatial grid: [B, d, n, n]
        xg = x.transpose(1, 2).reshape(B, d, self.n, self.n)

        scale_features = []
        for s, proj in zip(self.scale_sizes, self.scale_projs):
            if s == self.n:
                # No pooling needed — same resolution
                pooled = xg
            else:
                pooled = F.adaptive_avg_pool2d(xg, output_size=(s, s))  # [B, d, s, s]

            # Upsample back to original spatial size
            up = F.interpolate(pooled, size=(self.n, self.n), mode="bilinear", align_corners=False)
            # [B, d, n, n] → [B, N, d]
            up = up.flatten(2).transpose(1, 2)
            up = proj(up)  # [B, N, d]
            scale_features.append(up)

        # Concatenate all scales along feature dim: [B, N, d * num_scales]
        multi = torch.cat(scale_features, dim=-1)
        fused = self.fusion(multi)   # [B, N, d]

        # Residual connection + LayerNorm
        out = self.norm(x + self.dropout(fused))
        return out
