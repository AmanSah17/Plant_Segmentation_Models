"""
losses.py
---------
Multi-Task Learning loss functions.

L_MTL = λ_loc × L_seg + λ_cls × L_cls

L_seg : Weighted cross-entropy over H×W pixels (115 classes)
         Weights inversely proportional to pixel frequency (from class_pixel_counts.csv)
L_cls : Standard cross-entropy (114 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from config import (
    CLASS_PIXEL_CSV, NUM_SEG_CLASSES, NUM_CLS_CLASSES,
    LAMBDA_LOC, LAMBDA_CLS, BACKGROUND_CLASS_ID,
)


# ─────────────────────────────────────────────────────────────────────
def compute_seg_class_weights(device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency weights for the segmentation cross-entropy loss.

    Class weights are computed as:
        w_c = total_pixels / (num_classes × pixel_count_c)
    clipped to [0.05, 20.0] to avoid extreme values.

    Returns weights tensor of shape [NUM_SEG_CLASSES] on `device`.
    """
    df = pd.read_csv(CLASS_PIXEL_CSV)

    # Build pixel count array indexed by class_id (0-indexed by class_id 0..114)
    weights = np.zeros(NUM_SEG_CLASSES, dtype=np.float32)

    total_pixels = df["pixel_count"].sum()

    for _, row in df.iterrows():
        cid = int(row["class_id"])
        if 0 <= cid < NUM_SEG_CLASSES:
            weights[cid] = total_pixels / (NUM_SEG_CLASSES * float(row["pixel_count"] + 1))

    # Background is already in the CSV (class_id 0); no special handling needed
    # Clip extreme values
    weights = np.clip(weights, 0.05, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────
class SegmentationLoss(nn.Module):
    """
    Pixel-wise weighted cross-entropy for segmentation.
    Eqn (1): L_loc = (1/HW) Σ L_loc(M_ij, M_gt_ij)
    """

    def __init__(self, weight: torch.Tensor = None, ignore_index: int = -100):
        super().__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,    # [B, C, H, W]  logits
        target: torch.Tensor,  # [B, H, W]     LongTensor with class_id values
    ) -> torch.Tensor:
        """Returns scalar mean loss."""
        weight = self.weight.to(pred.device) if self.weight is not None else None
        loss = F.cross_entropy(
            pred, target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="mean",
        )
        return loss


# ─────────────────────────────────────────────────────────────────────
class ClassificationLoss(nn.Module):
    """
    Standard cross-entropy for disease classification.
    Eqn (2): L_cls = L_cls(L, y)
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        pred: torch.Tensor,   # [B, num_cls_classes]  logits
        target: torch.Tensor, # [B]  LongTensor with class index 0-113
    ) -> torch.Tensor:
        return self.ce(pred, target)


# ─────────────────────────────────────────────────────────────────────
class MTLLoss(nn.Module):
    """
    Combined Multi-Task Loss.
    Eqn (3): L_MTL = λ_loc × L_loc + λ_cls × L_cls

    Parameters
    ----------
    lambda_loc      : float — weight for localisation/segmentation loss
    lambda_cls      : float — weight for classification loss
    seg_weights     : Tensor | None — per-class pixel weights for seg CE
    """

    def __init__(
        self,
        lambda_loc: float = LAMBDA_LOC,
        lambda_cls: float = LAMBDA_CLS,
        seg_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.lambda_loc = lambda_loc
        self.lambda_cls = lambda_cls
        self.seg_loss = SegmentationLoss(weight=seg_weights)
        self.cls_loss = ClassificationLoss()

    def forward(
        self,
        seg_pred:   torch.Tensor,  # [B, 115, H, W]
        cls_pred:   torch.Tensor,  # [B, 114]
        seg_target: torch.Tensor,  # [B, H, W]
        cls_target: torch.Tensor,  # [B]
    ):
        """
        Returns
        -------
        total_loss : scalar Tensor
        seg_loss   : scalar Tensor
        cls_loss   : scalar Tensor
        """
        l_seg = self.seg_loss(seg_pred, seg_target)
        l_cls = self.cls_loss(cls_pred, cls_target)
        total = self.lambda_loc * l_seg + self.lambda_cls * l_cls
        return total, l_seg, l_cls
