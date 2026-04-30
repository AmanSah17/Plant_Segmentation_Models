"""
model/pdlc_vit.py
-----------------
PDLC-ViT: Plant Disease Localization and Classification Vision Transformer.

Full MTL model assembling all modules per the paper (Fig. 1):

  Image → PatchEmbedding
       → Segmentation Branch: CoScale → CoAttention  → CS, CA  ─┐
       → Classification Branch: CoScale → CoAttention → CS, CA  ─┤
                                                                   ↓
                                              CrossAttentionModule (CS, CA → Z)
                                                                   ↓
                                              TransformerEncoder  (Z → enc_z)
                                                   ↓              ↓
                                          SegmentationHead   ClassificationHead
                                                   ↓              ↓
                                          [B, 115, H, W]    [B, 114]
"""

import torch
import torch.nn as nn

from .patch_embed          import PatchEmbedding
from .co_scale             import CoScaleLayer
from .co_attention         import CoAttentionLayer
from .cross_attention      import CrossAttentionModule
from .transformer_encoder  import TransformerEncoder
from .seg_head             import SegmentationHead
from .cls_head             import ClassificationHead


class PDLCViT(nn.Module):
    """
    Parameters
    ----------
    img_size        : int   — square input side (224)
    patch_size      : int   — patch side (16)
    in_chans        : int   — input channels (3)
    embed_dim       : int   — d (768)
    num_heads       : int   — attention heads (12)
    num_enc_layers  : int   — transformer encoder depth (6)
    mlp_ratio       : float — FFN expansion (4.0)
    dropout         : float — global dropout (0.2)
    num_seg_classes : int   — segmentation output classes (115)
    num_cls_classes : int   — classification output classes (114)
    scale_sizes     : list  — co-scale pool sizes ([1,2,4])
    """

    def __init__(
        self,
        img_size:        int   = 224,
        patch_size:      int   = 16,
        in_chans:        int   = 3,
        embed_dim:       int   = 768,
        num_heads:       int   = 12,
        num_enc_layers:  int   = 6,
        mlp_ratio:       float = 4.0,
        dropout:         float = 0.2,
        num_seg_classes: int   = 115,
        num_cls_classes: int   = 114,
        scale_sizes:     list  = None,
    ):
        super().__init__()
        if scale_sizes is None:
            scale_sizes = [1, 2, 4]

        num_patches = (img_size // patch_size) ** 2

        # ── Shared Patch Embedding ─────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, dropout=dropout,
        )

        # ── Segmentation Branch ────────────────────────────────────────
        self.seg_coscale  = CoScaleLayer(embed_dim, num_patches, scale_sizes, dropout)
        self.seg_coattn   = CoAttentionLayer(embed_dim, num_heads, mlp_ratio, dropout)

        # ── Classification Branch ──────────────────────────────────────
        self.cls_coscale  = CoScaleLayer(embed_dim, num_patches, scale_sizes, dropout)
        self.cls_coattn   = CoAttentionLayer(embed_dim, num_heads, mlp_ratio, dropout)

        # ── Cross-Attention Module ─────────────────────────────────────
        self.cross_attn   = CrossAttentionModule(embed_dim, num_heads, mlp_ratio, dropout)

        # ── Transformer Encoder ────────────────────────────────────────
        self.transformer  = TransformerEncoder(
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_enc_layers, mlp_ratio=mlp_ratio,
            dropout=dropout, num_patches=num_patches,
        )

        # ── Task Heads ─────────────────────────────────────────────────
        self.seg_head = SegmentationHead(
            embed_dim=embed_dim, num_heads=num_heads,
            num_seg_classes=num_seg_classes, num_patches=num_patches,
            img_size=img_size, patch_size=patch_size,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.cls_head = ClassificationHead(
            embed_dim=embed_dim, num_heads=num_heads,
            num_cls_classes=num_cls_classes, num_patches=num_patches,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x:               torch.Tensor,
        gt_label:        torch.Tensor = None,
        gt_mask_tokens:  torch.Tensor = None,
    ):
        """
        x              : [B, 3, H, W]  — input image batch
        gt_label       : [B] LongTensor | None  — GT class label (training)
        gt_mask_tokens : [B, N, d] | None        — GT mask tokens (training)

        Returns
        -------
        seg_logits : [B, num_seg_classes, H, W]
        cls_logits : [B, num_cls_classes]
        """
        # ── Shared Patch Embedding ─────────────────────────────────────
        tokens = self.patch_embed(x)          # [B, N, d]

        # ── Parallel Branches ──────────────────────────────────────────
        # Segmentation branch
        cs_seg = self.seg_coscale(tokens)     # [B, N, d]
        ca_seg = self.seg_coattn(cs_seg)      # [B, N, d]

        # Classification branch
        cs_cls = self.cls_coscale(tokens)     # [B, N, d]
        ca_cls = self.cls_coattn(cs_cls)      # [B, N, d]

        # ── Cross-Attention: fuse CS and CA representations ────────────
        # Paper uses co-scaled from seg branch and co-attention from cls branch
        z = self.cross_attn(cs_seg, ca_cls)   # [B, N, d]

        # ── Transformer Encoder ────────────────────────────────────────
        enc_z = self.transformer(z)           # [B, N, d]

        # ── Task Heads ─────────────────────────────────────────────────
        seg_logits = self.seg_head(enc_z, gt_mask_tokens)
        cls_logits = self.cls_head(enc_z, seg_logits, gt_label)

        return seg_logits, cls_logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
