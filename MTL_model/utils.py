"""
utils.py
--------
Utility functions for the PDLC-ViT MTL pipeline:
  - Checkpoint save / load
  - Reproducibility seed setting
  - GT mask → patch token projection (training helper)
  - Visualisation helper (overlay prediction on image)
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import (
    CHECKPOINT_DIR, NUM_SEG_CLASSES, IMG_SIZE, EMBED_DIM,
    IMG_MEAN, IMG_STD,
)


# ─────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────
def save_checkpoint(
    state: dict,
    filename: str,
    is_best: bool = False,
):
    """
    Save model checkpoint.

    state should contain: epoch, model_state_dict, optimizer_state_dict,
                          scheduler_state_dict, best_val_miou, config_dict
    """
    path = CHECKPOINT_DIR / filename
    torch.save(state, str(path))
    if is_best:
        best_path = CHECKPOINT_DIR / "best_model.pth"
        torch.save(state, str(best_path))
        print(f"[Checkpoint] New best saved → {best_path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load a checkpoint.
    Returns: epoch, best_val_miou
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch          = ckpt.get("epoch", 0)
    best_val_miou  = ckpt.get("best_val_miou", 0.0)
    print(f"[Checkpoint] Loaded from {path}  (epoch {epoch}, best_mIoU {best_val_miou:.4f})")
    return epoch, best_val_miou


# ─────────────────────────────────────────────────────────────────────
def mask_to_patch_tokens(
    mask: torch.Tensor,
    patch_proj: torch.nn.Linear,
    patch_size: int = 16,
    embed_dim: int = EMBED_DIM,
) -> torch.Tensor:
    """
    Convert a GT segmentation mask [B, H, W] into patch-level token
    embeddings [B, N, d] to be used as query tokens in the seg head.

    Strategy:
      1. One-hot encode the mask into [B, C, H, W]
      2. Average-pool over each patch region → [B, C, n, n]
      3. Flatten spatial dims → [B, N, C]
      4. Project C → d via patch_proj linear layer
    """
    B, H, W = mask.shape
    n = H // patch_size
    C = NUM_SEG_CLASSES

    # One-hot: [B, C, H, W]  float
    one_hot = torch.zeros(B, C, H, W, device=mask.device, dtype=torch.float32)
    one_hot.scatter_(1, mask.unsqueeze(1), 1.0)

    # Average pool to patch grid: [B, C, n, n]
    import torch.nn.functional as F
    pooled = F.avg_pool2d(one_hot, kernel_size=patch_size, stride=patch_size)

    # Flatten spatial: [B, N, C]
    N = n * n
    pooled = pooled.flatten(2).transpose(1, 2)   # [B, N, C]

    # Project to embed_dim: [B, N, d]
    tokens = patch_proj(pooled)
    return tokens


# ─────────────────────────────────────────────────────────────────────
def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised image tensor [3, H, W] back to uint8 numpy [H, W, 3].
    """
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std  = np.array(IMG_STD,  dtype=np.float32)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # [H, W, 3]
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def colormap_for_classes(num_classes: int = NUM_SEG_CLASSES) -> np.ndarray:
    """Return an (N, 3) uint8 colour palette for `num_classes` classes."""
    np.random.seed(0)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]    # background = black
    palette[1:] = np.random.randint(50, 255, size=(num_classes - 1, 3))
    return palette


def save_prediction_overlay(
    image_tensor: torch.Tensor,     # [3, H, W]
    pred_mask: torch.Tensor,        # [H, W]  predicted class IDs
    gt_mask: torch.Tensor,          # [H, W]  ground-truth class IDs
    save_path: str,
    alpha: float = 0.5,
):
    """Save side-by-side: image | GT overlay | prediction overlay."""
    palette = colormap_for_classes()
    img = denormalize_image(image_tensor)

    def overlay(base, mask):
        coloured = palette[mask.cpu().numpy()]   # [H, W, 3]
        return (base * (1 - alpha) + coloured * alpha).astype(np.uint8)

    gt_ov   = overlay(img, gt_mask)
    pred_ov = overlay(img, pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, arr, title in zip(
        axes, [img, gt_ov, pred_ov], ["Image", "GT Mask", "Prediction"]
    ):
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training if validation loss doesn't improve by `min_delta`
    for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-3):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
