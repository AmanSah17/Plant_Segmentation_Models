"""
model/seg_head.py
-----------------
Segmentation Head (Fig. 4 of the paper).

Architecture (3 blocks with residual connections):
  Block 1: LN + MHSA  — query tokens attend to themselves (GT mask queries during training;
                          learned seg-query tokens during inference)
  Block 2: LN + MHCA  — cross-attend with shared representation Z
  Block 3: LN + MLP(ReLU FC layers)
  → Reshape + upsample to [B, num_seg_classes, H, W]

The output is a full-resolution segmentation logit map.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Parameters
    ----------
    embed_dim      : int  — feature dimension d
    num_heads      : int
    num_seg_classes: int  — output segmentation classes (115)
    num_patches    : int  — N = (img_size / patch_size)²
    img_size       : int  — original image side (for upsample target)
    patch_size     : int  — needed to compute output spatial size
    mlp_ratio      : float
    dropout        : float
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_seg_classes: int = 115,
        num_patches: int = 196,
        img_size: int = 224,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim       = embed_dim
        self.num_patches     = num_patches
        self.num_seg_classes = num_seg_classes
        self.n               = int(math.isqrt(num_patches))   # grid side
        self.img_size        = img_size
        self.patch_size      = patch_size
        ffn_dim = int(embed_dim * mlp_ratio)

        # Learned query tokens for inference (DETR-style)
        # Shape: [1, N, d]  — one query per patch position
        self.seg_queries = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.seg_queries, std=0.02)

        # ── Block 1: LN + MHSA (query tokens self-attend) ─────────────
        self.norm1  = nn.LayerNorm(embed_dim)
        self.mhsa1  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1  = nn.Dropout(dropout)

        # ── Block 2: LN + MHCA (cross-attend with Z) ──────────────────
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mhca2  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop2  = nn.Dropout(dropout)

        # ── Block 3: LN + MLP (ReLU, multiple FC) ─────────────────────
        self.norm3  = nn.LayerNorm(embed_dim)
        self.mlp3   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Final projection + upsample ───────────────────────────────
        # Map each patch token to num_seg_classes, then upsample to full res
        self.patch_proj = nn.Linear(embed_dim, num_seg_classes)

        # Optional: light convolutional upsampler for smoother masks
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(num_seg_classes, num_seg_classes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def _get_queries(self, gt_mask_tokens, batch_size, device):
        """
        During training  : use GT mask tokens (passed from outside)
        During inference : use learned seg_queries repeated for batch
        """
        if gt_mask_tokens is not None:
            return gt_mask_tokens   # [B, N, d]
        return self.seg_queries.expand(batch_size, -1, -1)

    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        gt_mask_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        z              : [B, N, d]  — shared representation from TransformerEncoder
        gt_mask_tokens : [B, N, d] | None  — GT mask query tokens (training only)

        returns : [B, num_seg_classes, H, W]  — segmentation logits
        """
        B = z.shape[0]
        device = z.device

        # ── Block 1: self-attention on queries ─────────────────────────
        q = self._get_queries(gt_mask_tokens, B, device)   # [B, N, d]
        q_n = self.norm1(q)
        a1, _ = self.mhsa1(q_n, q_n, q_n)
        q = q + self.drop1(a1)

        # ── Block 2: cross-attend queries with Z ───────────────────────
        q_n = self.norm2(q)
        z_n = self.norm2(z)
        a2, _ = self.mhca2(q_n, z_n, z_n)
        q = q + self.drop2(a2)

        # ── Block 3: MLP ───────────────────────────────────────────────
        q = q + self.mlp3(self.norm3(q))

        # ── Project to seg classes + upsample ─────────────────────────
        # [B, N, num_seg_classes]
        seg_tokens = self.patch_proj(q)

        # Reshape to spatial grid: [B, num_seg_classes, n, n]
        seg_map = seg_tokens.permute(0, 2, 1).reshape(
            B, self.num_seg_classes, self.n, self.n
        )

        # Upsample to full image resolution [B, num_seg_classes, H, W]
        seg_map = F.interpolate(
            seg_map, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        seg_map = self.upsample_conv(seg_map)
        return seg_map
