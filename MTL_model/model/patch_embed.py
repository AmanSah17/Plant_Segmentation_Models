"""
model/patch_embed.py
--------------------
Splits an input image into non-overlapping patches and projects each patch
to the embedding dimension d using a strided Conv2d (standard ViT approach).

Input  : [B, 3, H, W]
Output : [B, N, d]   where N = (H/P)*(W/P) and P = patch_size
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Parameters
    ----------
    img_size   : int   — square image side (default 224)
    patch_size : int   — square patch side (default 16)
    in_chans   : int   — input channels (3 for RGB)
    embed_dim  : int   — output embedding dimension d (default 768)
    dropout    : float — dropout after projection
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim   = embed_dim

        # Projection: each patch (P × P × 3) → embed_dim via a single Conv2d
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, H, W]
        returns : [B, N, d]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Expected {self.img_size}×{self.img_size}, got {H}×{W}"
        )
        # [B, d, H/P, W/P]
        x = self.proj(x)
        # [B, d, N_h, N_w] -> [B, N, d]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d]
        x = self.norm(x)
        x = self.dropout(x)
        return x
