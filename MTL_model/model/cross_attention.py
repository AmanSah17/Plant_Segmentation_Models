"""
model/cross_attention.py
------------------------
Cross-Attention Module (Fig. 3 of the paper).

Takes co-scaled (CS) and co-attention (CA) representations from
both the segmentation and classification branches and fuses them
into a shared representation Z.

Architecture (paper Fig. 3):
  Branch 1: CS  → QKV → MHSA → Add&Norm → FFN → Add&Norm → Z1
  Branch 2: CA  → QKV → MHSA → Add&Norm → FFN → Add&Norm → Z2
  Z = MHCA(Q=Z1, K=Z2, V=Z2)  (Multi-Head Cross-Attention)

Input  : cs [B, N, d], ca [B, N, d]
Output : Z  [B, N, d]
"""

import torch
import torch.nn as nn


class CrossAttentionModule(nn.Module):
    """
    Parameters
    ----------
    embed_dim  : int
    num_heads  : int
    mlp_ratio  : float
    dropout    : float
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = int(embed_dim * mlp_ratio)

        # ── Branch 1: process CS (co-scaled) ──────────────────────────
        self.norm1_b1   = nn.LayerNorm(embed_dim)
        self.mhsa_b1    = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_b1    = nn.Dropout(dropout)
        self.norm2_b1   = nn.LayerNorm(embed_dim)
        self.ffn_b1     = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # ── Branch 2: process CA (co-attention) ───────────────────────
        self.norm1_b2   = nn.LayerNorm(embed_dim)
        self.mhsa_b2    = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_b2    = nn.Dropout(dropout)
        self.norm2_b2   = nn.LayerNorm(embed_dim)
        self.ffn_b2     = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # ── Multi-Head Cross-Attention: fuse Z1 and Z2 ────────────────
        self.norm_mhca  = nn.LayerNorm(embed_dim)
        self.mhca       = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_mhca  = nn.Dropout(dropout)
        self.norm_out   = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def _branch(self, x, norm1, mhsa, drop, norm2, ffn):
        """Single parallel branch: MHSA + Add&Norm → FFN + Add&Norm."""
        # MHSA block
        x_n   = norm1(x)
        a, _  = mhsa(x_n, x_n, x_n)
        x     = x + drop(a)
        # FFN block
        x     = x + ffn(norm2(x))
        return x

    # ------------------------------------------------------------------
    def forward(
        self, cs: torch.Tensor, ca: torch.Tensor
    ) -> torch.Tensor:
        """
        cs : [B, N, d]  — co-scaled representation
        ca : [B, N, d]  — co-attention representation
        returns Z : [B, N, d]
        """
        # Branch 1  (CS path)
        z1 = self._branch(cs, self.norm1_b1, self.mhsa_b1,
                           self.drop_b1, self.norm2_b1, self.ffn_b1)

        # Branch 2  (CA path)
        z2 = self._branch(ca, self.norm1_b2, self.mhsa_b2,
                           self.drop_b2, self.norm2_b2, self.ffn_b2)

        # MHCA: Q = Z1, K = Z2, V = Z2  → Z
        z1_n    = self.norm_mhca(z1)
        z2_n    = self.norm_mhca(z2)
        z, _    = self.mhca(z1_n, z2_n, z2_n)
        z       = self.norm_out(z1 + self.drop_mhca(z))
        return z
