"""
model/co_attention.py
---------------------
Co-Attention Layer.

Captures global context and cross-scale interactions between patch tokens
using Multi-Head Self-Attention (MHSA).

Input  : [B, N, d]
Output : [B, N, d]  (CA = attended patch representations)
"""

import torch
import torch.nn as nn


class CoAttentionLayer(nn.Module):
    """
    Parameters
    ----------
    embed_dim  : int   — feature dimension d
    num_heads  : int   — number of attention heads
    mlp_ratio  : float — FFN hidden-dim = embed_dim * mlp_ratio
    dropout    : float — attention + projection dropout
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ── Multi-Head Self-Attention ──────────────────────────────────
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # ── Feed-Forward Network ───────────────────────────────────────
        ffn_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, N, d]
        returns : [B, N, d]
        """
        # Self-attention with residual (Add & Norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # MHSA
        x = x + self.attn_drop(attn_out)

        # FFN with residual (Add & Norm)
        x = x + self.ffn(self.norm2(x))
        return x
