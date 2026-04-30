"""
model/cls_head.py
-----------------
Classification Head (Fig. 4 of the paper).

Architecture (2 blocks):
  Block 1: LN + MHCA  — query = class tokens derived from GT label + predicted mask + Z
                         keys/values = shared representation Z
  Block 2: LN + MLP(Sigmoid FC layers)
  → 114-class logits (per-image disease classification)

During training   : GT class embedding is used as the query
During inference  : learned class query token is used
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Parameters
    ----------
    embed_dim      : int  — feature dimension d
    num_heads      : int
    num_cls_classes: int  — output disease classes (114)
    num_patches    : int  — N
    mlp_ratio      : float
    dropout        : float
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_cls_classes: int = 114,
        num_patches: int = 196,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim       = embed_dim
        self.num_cls_classes = num_cls_classes
        self.num_patches     = num_patches
        ffn_dim = int(embed_dim * mlp_ratio)

        # Learned class query token (inference / fallback)
        # Shape: [1, 1, d]  — single CLS token
        self.cls_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_query, std=0.02)

        # GT class embedding lookup (training) — maps label → d-dim query
        self.cls_embed = nn.Embedding(num_cls_classes, embed_dim)

        # Projection to fuse: cls_embed + predicted mask global pool + Z mean → d
        # (mask_global: 1×d, Z_mean: 1×d, cls_embed: 1×d → concat 3d → d)
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Block 1: LN + MHCA (Q = fused query, K/V = Z) ────────────
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhca  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        # ── Block 2: LN + MLP (Sigmoid FC layers, paper uses Sigmoid) ─
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )

        # ── Final classifier ──────────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, num_cls_classes)

    # ------------------------------------------------------------------
    def _build_query(self, z, seg_logits, gt_label, device):
        """
        Construct the query token for the classification head.

        During training (gt_label is not None):
          Q = f(cls_embed(gt_label), global_pool(seg_logits→d), mean(Z))

        During inference:
          Q = learned cls_query token
        """
        B = z.shape[0]

        if gt_label is not None:
            # GT class embedding [B, d]
            cls_emb = self.cls_embed(gt_label)             # [B, d]

            # Global-pool the segmentation logits into a d-dim vector
            # seg_logits: [B, num_cls, H, W] → mean over spatial → [B, num_seg_cls]
            # project → [B, d]
            if seg_logits is not None:
                seg_pooled = seg_logits.mean(dim=[2, 3])   # [B, num_seg_cls]
                # Use a simple repeat to match embed_dim if needed
                # (embed_dim // num_seg_cls repeating trick)
                num_seg = seg_pooled.shape[1]
                if num_seg != self.embed_dim:
                    # Interpolate to embed_dim
                    seg_pooled = torch.nn.functional.interpolate(
                        seg_pooled.unsqueeze(1), size=self.embed_dim, mode="linear", align_corners=False
                    ).squeeze(1)   # [B, d]
            else:
                seg_pooled = torch.zeros(B, self.embed_dim, device=device)

            # Z global mean [B, d]
            z_mean = z.mean(dim=1)                         # [B, d]

            # Fuse all three: [B, 3d] → [B, d]
            fused = torch.cat([cls_emb, seg_pooled, z_mean], dim=-1)  # [B, 3d]
            query = self.query_proj(fused).unsqueeze(1)    # [B, 1, d]
        else:
            query = self.cls_query.expand(B, -1, -1)       # [B, 1, d]

        return query

    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        seg_logits: torch.Tensor = None,
        gt_label: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        z          : [B, N, d]  — shared representation
        seg_logits : [B, num_seg_classes, H, W] | None
        gt_label   : [B] LongTensor | None  (training only)

        returns : [B, num_cls_classes]  — classification logits
        """
        device = z.device

        # ── Block 1: MHCA ─────────────────────────────────────────────
        q = self._build_query(z, seg_logits, gt_label, device)  # [B, 1, d]
        q_n = self.norm1(q)
        z_n = self.norm1(z)
        a, _ = self.mhca(q_n, z_n, z_n)
        q = q + self.drop1(a)   # [B, 1, d]

        # ── Block 2: MLP ───────────────────────────────────────────────
        q = q + self.mlp(self.norm2(q))   # [B, 1, d]

        # ── Classify ───────────────────────────────────────────────────
        logits = self.classifier(q.squeeze(1))  # [B, num_cls_classes]
        return logits
