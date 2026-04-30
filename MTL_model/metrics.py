"""
metrics.py
----------
Evaluation metrics for the MTL pipeline.

Segmentation:
  - Per-class IoU
  - Mean IoU (mIoU) — background excluded
  - Pixel Accuracy

Classification:
  - Top-1 Accuracy
  - Per-class F1
  - Macro-F1

All metrics are computed over accumulated batch predictions.
"""

import torch
import numpy as np
from config import NUM_SEG_CLASSES, NUM_CLS_CLASSES, BACKGROUND_CLASS_ID


# ─────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    """
    Accumulates confusion matrix across batches and computes IoU metrics.

    Usage:
        metrics = SegmentationMetrics()
        for batch in dataloader:
            metrics.update(seg_preds, seg_targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int = NUM_SEG_CLASSES,
                 ignore_background: bool = True):
        self.num_classes        = num_classes
        self.ignore_background  = ignore_background
        self.reset()

    def reset(self):
        self.conf_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds   : [B, C, H, W]  logits  OR  [B, H, W]  predicted class indices
        targets : [B, H, W]     LongTensor
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)   # [B, H, W]

        preds   = preds.cpu().long().view(-1)
        targets = targets.cpu().long().view(-1)

        # Mask out invalid targets
        valid = (targets >= 0) & (targets < self.num_classes)
        preds   = preds[valid]
        targets = targets[valid]

        # Accumulate into confusion matrix
        indices = self.num_classes * targets + preds
        cm = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Returns dict with per-class IoU, mIoU, pixel accuracy."""
        cm = self.conf_matrix.float()

        tp = torch.diag(cm)                       # [C]
        fp = cm.sum(0) - tp                        # predicted as c but not c
        fn = cm.sum(1) - tp                        # c but predicted as other

        iou_per_class = tp / (tp + fp + fn + 1e-6)  # [C]

        # Pixel accuracy
        pixel_acc = tp.sum() / (cm.sum() + 1e-6)

        # mIoU: exclude background
        if self.ignore_background:
            iou_disease = iou_per_class[1:]         # classes 1-114
            mIoU = iou_disease[iou_disease > 0].mean().item()
        else:
            mIoU = iou_per_class.mean().item()

        return {
            "mIoU":          mIoU,
            "pixel_acc":     pixel_acc.item(),
            "iou_per_class": iou_per_class.tolist(),  # length NUM_SEG_CLASSES
        }


# ─────────────────────────────────────────────────────────────────────
class ClassificationMetrics:
    """
    Accumulates predictions and targets for classification metrics.
    """

    def __init__(self, num_classes: int = NUM_CLS_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._preds   = []
        self._targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : [B, num_classes]
        targets : [B]  LongTensor
        """
        preds = logits.argmax(dim=1).cpu()
        self._preds.append(preds)
        self._targets.append(targets.cpu())

    def compute(self) -> dict:
        preds   = torch.cat(self._preds).numpy()
        targets = torch.cat(self._targets).numpy()

        # Top-1 accuracy
        top1 = float((preds == targets).mean())

        # Per-class precision / recall / F1
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            tp[c] = ((preds == c) & (targets == c)).sum()
            fp[c] = ((preds == c) & (targets != c)).sum()
            fn[c] = ((preds != c) & (targets == c)).sum()

        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)

        macro_f1  = float(f1[tp + fn > 0].mean())   # exclude absent classes

        return {
            "top1_accuracy":  top1,
            "macro_f1":       macro_f1,
            "f1_per_class":   f1.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────
class MTLMetrics:
    """Convenience wrapper for both seg and cls metrics."""

    def __init__(self):
        self.seg = SegmentationMetrics()
        self.cls = ClassificationMetrics()

    def reset(self):
        self.seg.reset()
        self.cls.reset()

    def update(self, seg_preds, seg_targets, cls_logits, cls_targets):
        self.seg.update(seg_preds, seg_targets)
        self.cls.update(cls_logits, cls_targets)

    def compute(self) -> dict:
        seg_res = self.seg.compute()
        cls_res = self.cls.compute()
        return {**seg_res, **cls_res}
