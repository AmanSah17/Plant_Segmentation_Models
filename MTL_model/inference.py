"""
inference.py
------------
Inference script for the PDLC-ViT MTL model.

Usage:
    # Evaluate on test set (reports full metrics)
    python MTL_model/inference.py --checkpoint checkpoints/best_model.pth --split test

    # Single image prediction
    python MTL_model/inference.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg

Outputs:
  - Console: mIoU, pixel accuracy, top-1 accuracy, macro-F1
  - Per-class IoU table
  - Prediction overlay PNGs saved to MTL_model/inference_outputs/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED,
    IMG_SIZE, PATCH_SIZE, EMBED_DIM,
    NUM_HEADS, NUM_ENC_LAYERS, MLP_RATIO, DROPOUT,
    NUM_SEG_CLASSES, NUM_CLS_CLASSES, CO_SCALE_SIZES,
    CLASS_MAP_CSV, MTL_DIR, IMG_MEAN, IMG_STD,
)
from dataset       import PlantSegMTLDataset
from augmentations import get_test_transform
from model         import PDLCViT
from losses        import MTLLoss, compute_seg_class_weights
from metrics       import MTLMetrics
from utils         import (
    set_seed, load_checkpoint, save_prediction_overlay, denormalize_image,
)

OUTPUT_DIR = MTL_DIR / "inference_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
def load_class_map() -> dict:
    """Return {class_id: disease_name} mapping."""
    df = pd.read_csv(CLASS_MAP_CSV)
    return {int(r["class_id"]): r["Disease"] for _, r in df.iterrows()}


# ─────────────────────────────────────────────────────────────────────
def preprocess_single_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a single image to tensor [1, 3, H, W]."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_test_transform()
    result = transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
    tensor = result["image"].unsqueeze(0)   # [1, 3, H, W]
    return tensor


# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_single(model, image_tensor: torch.Tensor, device: torch.device):
    """
    Run inference on a single image.
    Returns:
        seg_pred  : [H, W]  predicted class IDs
        cls_pred  : int     predicted disease class (0-indexed)
        cls_probs : [114]   softmax probabilities
    """
    model.eval()
    x = image_tensor.to(device)
    seg_logits, cls_logits = model(x, gt_label=None, gt_mask_tokens=None)
    seg_pred  = seg_logits[0].argmax(dim=0).cpu()        # [H, W]
    cls_probs = torch.softmax(cls_logits[0], dim=0).cpu()
    cls_pred  = cls_probs.argmax().item()
    return seg_pred, cls_pred, cls_probs


# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test_set(
    model:     PDLCViT,
    loader:    DataLoader,
    criterion: MTLLoss,
    device:    torch.device,
    class_map: dict,
    save_n:    int = 20,    # number of overlay images to save
):
    """Evaluate on test set (NO augmentation). Save N overlay images."""
    model.eval()
    metrics    = MTLMetrics()
    total_loss = total_seg = total_cls = 0.0
    n_batches  = 0
    n_saved    = 0

    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        masks   = batch["mask"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        # Inference: no GT tokens
        seg_logits, cls_logits = model(images, gt_label=None, gt_mask_tokens=None)

        loss, l_seg, l_cls = criterion(seg_logits, cls_logits, masks, labels)
        total_loss += loss.item()
        total_seg  += l_seg.item()
        total_cls  += l_cls.item()
        n_batches  += 1

        metrics.update(seg_logits, masks, cls_logits, labels)

        # Save overlay images (first N images across batches)
        if n_saved < save_n:
            seg_preds = seg_logits.argmax(dim=1).cpu()
            for i in range(min(images.shape[0], save_n - n_saved)):
                save_path = str(OUTPUT_DIR / f"pred_{n_saved:04d}.png")
                save_prediction_overlay(
                    image_tensor=images[i].cpu(),
                    pred_mask=seg_preds[i],
                    gt_mask=masks[i].cpu(),
                    save_path=save_path,
                )
                n_saved += 1

    n = max(n_batches, 1)
    results = metrics.compute()
    results["test_loss"]     = total_loss / n
    results["test_seg_loss"] = total_seg  / n
    results["test_cls_loss"] = total_cls  / n
    return results


# ─────────────────────────────────────────────────────────────────────
def print_results(results: dict, class_map: dict):
    print("\n" + "=" * 70)
    print("  PDLC-ViT MTL — Evaluation Results")
    print("=" * 70)
    print(f"  Total Loss        : {results.get('test_loss', results.get('val_loss', '-')):.4f}")
    print(f"  Seg Loss          : {results.get('test_seg_loss', results.get('val_seg_loss', '-')):.4f}")
    print(f"  Cls Loss          : {results.get('test_cls_loss', results.get('val_cls_loss', '-')):.4f}")
    print(f"  mIoU (no bg)      : {results['mIoU']:.4f}")
    print(f"  Pixel Accuracy    : {results['pixel_acc']:.4f}")
    print(f"  Top-1 Accuracy    : {results['top1_accuracy']:.4f}")
    print(f"  Macro-F1          : {results['macro_f1']:.4f}")
    print("-" * 70)
    print("  Per-Class IoU:")
    iou_list = results["iou_per_class"]
    rows = []
    for cid, iou in enumerate(iou_list):
        name = class_map.get(cid, f"class_{cid}")
        rows.append((cid, name, iou))
    # Sort by IoU descending
    rows.sort(key=lambda x: -x[2])
    for cid, name, iou in rows:
        bar = "█" * int(iou * 20)
        print(f"    {cid:3d} {name[:35]:<35} {iou:.4f}  {bar}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PDLC-ViT Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--image",      type=str, default=None,
                        help="Path to a single image for prediction")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--save_n",     type=int, default=20,
                        help="Number of overlay images to save")
    args = parser.parse_args()

    set_seed(SEED)
    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_map = load_class_map()

    # ── Build model ───────────────────────────────────────────────────
    model = PDLCViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_enc_layers=NUM_ENC_LAYERS,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        num_seg_classes=NUM_SEG_CLASSES, num_cls_classes=NUM_CLS_CLASSES,
        scale_sizes=CO_SCALE_SIZES,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── Single image mode ─────────────────────────────────────────────
    if args.image:
        img_tensor = preprocess_single_image(args.image)
        seg_pred, cls_pred, cls_probs = infer_single(model, img_tensor, device)

        disease_name = class_map.get(cls_pred + 1, f"disease_{cls_pred}")
        print(f"\nPredicted Disease : {disease_name} (class {cls_pred})")
        print(f"Confidence        : {cls_probs[cls_pred].item():.4f}")
        print(f"Seg mask shape    : {seg_pred.shape}, unique classes: {seg_pred.unique().tolist()}")

        # Save overlay (no GT mask available for single image)
        dummy_gt = torch.zeros_like(seg_pred)
        save_path = str(OUTPUT_DIR / "single_image_pred.png")
        save_prediction_overlay(img_tensor[0], seg_pred, dummy_gt, save_path)
        print(f"Overlay saved → {save_path}")
        return

    # ── Test/Val set evaluation ───────────────────────────────────────
    dataset = PlantSegMTLDataset(split=args.split, transform=get_test_transform())
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=PlantSegMTLDataset.collate_fn,
    )
    print(f"[Inference] {args.split} split: {len(dataset)} images")

    seg_weights = compute_seg_class_weights(device)
    criterion   = MTLLoss(seg_weights=seg_weights).to(device)

    results = evaluate_test_set(model, loader, criterion, device, class_map, args.save_n)
    print_results(results, class_map)
    print(f"\n  Overlay images saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
