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


def focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(labels >= 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(labels >= 0.5, torch.full_like(labels, alpha), torch.full_like(labels, 1.0 - alpha))
    return (alpha_t * torch.pow(1.0 - pt, gamma) * bce).mean()


def segmentation_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    loss_cfg = loss_cfg or {}
    if logits.shape[1] > 1:
        return multiclass_segmentation_loss_with_logits(logits, labels, loss_cfg)

    pos_weight_value = float(loss_cfg.get("pos_weight", 1.0))
    pos_weight = torch.tensor([pos_weight_value], device=logits.device, dtype=logits.dtype)

    weighted_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    dice = dice_loss_with_logits(logits, labels, smooth=float(loss_cfg.get("dice_smooth", 1.0)))
    focal = focal_loss_with_logits(
        logits,
        labels,
        alpha=float(loss_cfg.get("focal_alpha", 0.75)),
        gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )

    return (
        float(loss_cfg.get("bce_weight", 0.5)) * weighted_bce
        + float(loss_cfg.get("dice_weight", 1.0)) * dice
        + float(loss_cfg.get("focal_weight", 1.0)) * focal
    )


def multiclass_dice_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    smooth: float = 1.0,
    include_background: bool = False,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dim=dims)
    cardinality = torch.sum(probs + one_hot, dim=dims)
    dice_loss = 1.0 - ((2.0 * intersection + smooth) / (cardinality + smooth))
    if not include_background and num_classes > 1:
        dice_loss = dice_loss[1:]
        if class_weights is not None:
            class_weights = class_weights[1:]
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        weights = weights / weights.mean().clamp_min(1e-6)
        return (dice_loss * weights).mean()
    return dice_loss.mean()


def multiclass_focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(logits, labels.long(), weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return (torch.pow(1.0 - pt, gamma) * ce).mean()


def multiclass_segmentation_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    labels = labels.long()
    class_weights = loss_cfg.get("class_weights")
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)

    ce = torch.nn.functional.cross_entropy(logits, labels, weight=weight_tensor)
    dice = multiclass_dice_loss_with_logits(
        logits,
        labels,
        class_weights=weight_tensor,
        smooth=float(loss_cfg.get("dice_smooth", 1.0)),
        include_background=bool(loss_cfg.get("include_background_in_dice", False)),
    )
    focal = multiclass_focal_loss_with_logits(
        logits,
        labels,
        class_weights=weight_tensor,
        gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )
    return (
        float(loss_cfg.get("ce_weight", 1.0)) * ce
        + float(loss_cfg.get("dice_weight", 1.0)) * dice
        + float(loss_cfg.get("focal_weight", 0.5)) * focal
    )


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
    positive_ratio = targets.mean(dtype=np.float64)
    predicted_positive_ratio = preds.mean(dtype=np.float64)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
        "positive_ratio": float(positive_ratio),
        "predicted_positive_ratio": float(predicted_positive_ratio),
    }


def compute_multiclass_segmentation_metrics(eval_pred: Any, num_classes: int) -> dict[str, float]:
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 4:
        preds = preds.argmax(axis=1)

    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    valid = labels >= 0
    preds = preds[valid]
    labels = labels[valid]

    eps = 1e-7
    ious = []
    dices = []
    recalls = []
    precisions = []
    present_classes = 0
    for class_id in range(1, num_classes):
        pred_c = preds == class_id
        label_c = labels == class_id
        label_sum = label_c.sum(dtype=np.float64)
        pred_sum = pred_c.sum(dtype=np.float64)
        if label_sum == 0:
            continue
        present_classes += 1
        tp = np.logical_and(pred_c, label_c).sum(dtype=np.float64)
        fp = np.logical_and(pred_c, np.logical_not(label_c)).sum(dtype=np.float64)
        fn = np.logical_and(np.logical_not(pred_c), label_c).sum(dtype=np.float64)
        ious.append((tp + eps) / (tp + fp + fn + eps))
        dices.append((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
        recalls.append((tp + eps) / (tp + fn + eps))
        precisions.append((tp + eps) / (tp + fp + eps))

    pixel_accuracy = (preds == labels).mean(dtype=np.float64)
    foreground = labels > 0
    foreground_accuracy = (preds[foreground] == labels[foreground]).mean(dtype=np.float64) if foreground.any() else 0.0

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "pixel_accuracy": float(pixel_accuracy),
        "foreground_accuracy": float(foreground_accuracy),
        "present_classes": float(present_classes),
    }
