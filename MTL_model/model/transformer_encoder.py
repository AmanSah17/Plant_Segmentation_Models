"""
model/transformer_encoder.py
-----------------------------
Transformer Encoder (Fig. 2 of the paper).

Processes the shared representation Z through:
  1. Linear Projection (to higher-dimensional space)
  2. Positional Encoding (learnable)
  3. N × [MHSA + Add&Norm + FFN(ReLU) + Add&Norm]

Input  : [B, N, d]
Output : [B, N, d]
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer (paper Section 4.2 steps 3-6)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = int(embed_dim * mlp_ratio)

        # Step 3-4: Multi-head self-attention + Add & Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # Step 5-6: Feed-Forward (ReLU) + Add & Norm
        # Note: paper uses ReLU in FFN (not GELU) — kept faithful
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHSA + Add & Norm
        x_n = self.norm1(x)
        a, _ = self.attn(x_n, x_n, x_n)
        x = x + self.attn_drop(a)
        # FFN + Add & Norm
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    """
    Full transformer encoder:
      Linear Projection → Positional Encoding → N × EncoderLayer

    Parameters
    ----------
    embed_dim   : int   — input/output dimension d
    num_heads   : int
    num_layers  : int   — depth N
    mlp_ratio   : float
    dropout     : float
    num_patches : int   — needed for positional encoding size
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_patches: int = 196,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = num_patches

        # Step 1: Linear Projection (optional up-projection; keep d→d here)
        self.linear_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        # Step 2: Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Steps 3-6: Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : [B, N, d]  — shared representation from cross-attention
        returns : [B, N, d]  — encoded features
        """
        # Linear projection
        z = self.linear_proj(z)
        # Positional encoding
        z = self.dropout(z + self.pos_embed)
        # Transformer layers
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        return z
