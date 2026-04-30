"""
validate.py
-----------
Validation loop for the PDLC-ViT MTL model.

Can be called from train.py or run standalone:
    python validate.py --checkpoint checkpoints/best_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED,
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_SEG_CLASSES,
    NUM_HEADS, NUM_ENC_LAYERS, MLP_RATIO, DROPOUT,
    NUM_CLS_CLASSES, CO_SCALE_SIZES,
)
from dataset       import PlantSegMTLDataset
from augmentations import get_transform
from model         import PDLCViT
from losses        import MTLLoss, compute_seg_class_weights
from metrics       import MTLMetrics
from utils         import set_seed, load_checkpoint, mask_to_patch_tokens
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch(
    model:      torch.nn.Module,
    loader:     DataLoader,
    criterion:  MTLLoss,
    device:     torch.device,
    use_gt_tokens: bool = True,
    mask_proj:  nn.Module = None,   # trained Linear(NUM_SEG_CLASSES → embed_dim)
):
    """
    Run one validation pass.

    Parameters
    ----------
    mask_proj : nn.Linear | None
        The same mask-to-token projection used during training.  Pass this
        explicitly so embed_dim stays consistent with the model.
        If None, a fresh Linear(NUM_SEG_CLASSES → EMBED_DIM) is created
        (for standalone validation against a checkpoint).

    Returns
    -------
    metrics_dict : dict with keys:
        val_loss, val_seg_loss, val_cls_loss,
        mIoU, pixel_acc, top1_accuracy, macro_f1
    """
    model.eval()
    metrics = MTLMetrics()

    total_loss = 0.0
    total_seg  = 0.0
    total_cls  = 0.0
    n_batches  = 0

    # Use the caller-supplied mask_proj (correct embed_dim) or fall back to
    # a fresh one at the default EMBED_DIM for standalone validation.
    if mask_proj is None:
        mask_token_proj = nn.Linear(NUM_SEG_CLASSES, EMBED_DIM, bias=False).to(device)
    else:
        mask_token_proj = mask_proj

    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        masks   = batch["mask"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        # Build GT mask tokens for seg head (optional during val)
        gt_mask_tokens = None
        if use_gt_tokens:
            gt_mask_tokens = mask_to_patch_tokens(
                masks, mask_token_proj, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM
            )

        seg_logits, cls_logits = model(
            images,
            gt_label=None,
            gt_mask_tokens=gt_mask_tokens,
        )

        loss, l_seg, l_cls = criterion(seg_logits, cls_logits, masks, labels)

        total_loss += loss.item()
        total_seg  += l_seg.item()
        total_cls  += l_cls.item()
        n_batches  += 1

        metrics.update(seg_logits, masks, cls_logits, labels)

    n = max(n_batches, 1)
    result = metrics.compute()
    result["val_loss"]     = total_loss / n
    result["val_seg_loss"] = total_seg  / n
    result["val_cls_loss"] = total_cls  / n
    return result


# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate PDLC-ViT MTL model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Validate] Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────
    transform = get_transform(args.split)
    dataset   = PlantSegMTLDataset(split=args.split, transform=transform)
    loader    = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=PlantSegMTLDataset.collate_fn,
    )
    print(f"[Validate] {args.split} split: {len(dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────
    model = PDLCViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_enc_layers=NUM_ENC_LAYERS,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        num_seg_classes=NUM_SEG_CLASSES, num_cls_classes=NUM_CLS_CLASSES,
        scale_sizes=CO_SCALE_SIZES,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)

    # ── Loss ──────────────────────────────────────────────────────────
    seg_weights = compute_seg_class_weights(device)
    criterion   = MTLLoss(seg_weights=seg_weights).to(device)

    # ── Run ───────────────────────────────────────────────────────────
    results = validate_one_epoch(model, loader, criterion, device)

    print("\n" + "=" * 50)
    print(f"  Validation Results ({args.split} split)")
    print("=" * 50)
    print(f"  Total Loss     : {results['val_loss']:.4f}")
    print(f"  Seg Loss       : {results['val_seg_loss']:.4f}")
    print(f"  Cls Loss       : {results['val_cls_loss']:.4f}")
    print(f"  mIoU           : {results['mIoU']:.4f}")
    print(f"  Pixel Accuracy : {results['pixel_acc']:.4f}")
    print(f"  Top-1 Accuracy : {results['top1_accuracy']:.4f}")
    print(f"  Macro-F1       : {results['macro_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
