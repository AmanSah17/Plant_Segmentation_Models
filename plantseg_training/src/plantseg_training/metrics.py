from __future__ import annotations

from typing import Any

import numpy as np
import torch


def dice_loss_with_logits(logits: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * labels, dim=dims)
    cardinality = torch.sum(probs + labels, dim=dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return bce + dice_loss_with_logits(logits, labels)


def compute_binary_segmentation_metrics(eval_pred: Any, threshold: float = 0.5) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = probs >= threshold
    targets = labels >= 0.5

    tp = np.logical_and(preds, targets).sum(dtype=np.float64)
    fp = np.logical_and(preds, np.logical_not(targets)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(preds), targets).sum(dtype=np.float64)
    tn = np.logical_and(np.logical_not(preds), np.logical_not(targets)).sum(dtype=np.float64)

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
    }
