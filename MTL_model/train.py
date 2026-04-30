"""
train.py
--------
Main training script for the PDLC-ViT MTL model.

Usage (from within activated torch_gpu env):
    python MTL_model/train.py
    python MTL_model/train.py --epochs 100 --batch_size 16 --resume checkpoints/last.pth

MLflow experiment: "PDLC-ViT-MTL"
Logs per epoch:
  train_loss, train_seg_loss, train_cls_loss
  val_loss,   val_seg_loss,   val_cls_loss
  val_mIoU,   val_pixel_acc, val_top1_accuracy, val_macro_f1
  learning_rate
  val_iou_class_{i} (per-class IoU for all 115 classes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED,
    IMG_SIZE, PATCH_SIZE, EMBED_DIM,
    NUM_HEADS, NUM_ENC_LAYERS, MLP_RATIO, DROPOUT,
    NUM_SEG_CLASSES, NUM_CLS_CLASSES, CO_SCALE_SIZES,
    LEARNING_RATE, WEIGHT_DECAY,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE,
    EARLY_STOP_DELTA, EARLY_STOP_PATIENCE,
    LAMBDA_LOC, LAMBDA_CLS, MAX_EPOCHS,
    MLFLOW_URI, MLFLOW_EXPERIMENT, MLFLOW_RUN_TAGS,
    CHECKPOINT_DIR, AMP_ENABLED,
)
from dataset       import PlantSegMTLDataset
from augmentations import get_transform
from model         import PDLCViT
from losses        import MTLLoss, compute_seg_class_weights
from validate      import validate_one_epoch
from utils         import (
    set_seed, save_checkpoint, load_checkpoint,
    mask_to_patch_tokens, EarlyStopping,
)


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train PDLC-ViT MTL")
    p.add_argument("--epochs",       type=int,   default=MAX_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LEARNING_RATE)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--run_name",     type=str,   default=None,
                   help="MLflow run name (auto-generated if not set)")
    p.add_argument("--lambda_loc",   type=float, default=LAMBDA_LOC)
    p.add_argument("--lambda_cls",   type=float, default=LAMBDA_CLS)
    p.add_argument("--num_enc_layers", type=int, default=NUM_ENC_LAYERS)
    p.add_argument("--accum_steps",  type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--dry_run",      action="store_true",
                   help="Run quickly on a tiny subset of data to verify the pipeline")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:         PDLCViT,
    loader:        DataLoader,
    criterion:     MTLLoss,
    optimizer:     torch.optim.Optimizer,
    device:        torch.device,
    mask_proj:     nn.Linear,
    epoch:         int,
    accum_steps:   int = 1,
    scaler:        torch.amp.GradScaler = None,
    use_amp:       bool = False,
):
    """
    One training epoch.
    Returns dict: train_loss, train_seg_loss, train_cls_loss
    """
    model.train()

    total_loss = 0.0
    total_seg  = 0.0
    total_cls  = 0.0
    n_batches  = 0
    t0         = time.time()

    for batch_idx, batch in enumerate(loader):
        images  = batch["image"].to(device, non_blocking=True)   # [B,3,H,W]
        masks   = batch["mask"].to(device, non_blocking=True)    # [B,H,W]
        labels  = batch["label"].to(device, non_blocking=True)   # [B]

        with torch.amp.autocast("cuda", enabled=use_amp):
            # Build GT mask tokens for seg head during training
            gt_mask_tokens = mask_to_patch_tokens(
                masks, mask_proj, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM
            )

            seg_logits, cls_logits = model(
                images,
                gt_label=labels,
                gt_mask_tokens=gt_mask_tokens,
            )

            loss, l_seg, l_cls = criterion(seg_logits, cls_logits, masks, labels)
            loss_scaled = loss / accum_steps

        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_seg  += l_seg.item()
        total_cls  += l_cls.item()
        n_batches  += 1

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d} | Step {batch_idx+1:4d}/{len(loader)} | "
                f"Loss {loss.item():.4f} "
                f"(seg {l_seg.item():.4f}, cls {l_cls.item():.4f}) | "
                f"{elapsed:.1f}s"
            )

    n = max(n_batches, 1)
    return {
        "train_loss":     total_loss / n,
        "train_seg_loss": total_seg  / n,
        "train_cls_loss": total_cls  / n,
    }


# ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(SEED)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  PDLC-ViT MTL Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = PlantSegMTLDataset(split="Training", transform=get_transform("Training"))
    val_ds   = PlantSegMTLDataset(split="Validation",   transform=get_transform("Validation"))

    if args.dry_run:
        print("  [DRY RUN] Subsetting datasets to 32 samples each for smoke testing!")
        train_ds.meta = train_ds.meta.head(32).reset_index(drop=True)
        val_ds.meta = val_ds.meta.head(32).reset_index(drop=True)

    real_batch_size = max(1, args.batch_size // args.accum_steps)
    print(f"  Effective Batch: {args.batch_size} | Real Batch: {real_batch_size} | Accum Steps: {args.accum_steps}")

    train_loader = DataLoader(
        train_ds, batch_size=real_batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
        collate_fn=PlantSegMTLDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=real_batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=PlantSegMTLDataset.collate_fn,
    )
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────
    model = PDLCViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_enc_layers=args.num_enc_layers,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        num_seg_classes=NUM_SEG_CLASSES, num_cls_classes=NUM_CLS_CLASSES,
        scale_sizes=CO_SCALE_SIZES,
    ).to(device)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    # ── Mask-to-token projection (shared across epochs) ───────────────
    mask_proj = nn.Linear(NUM_SEG_CLASSES, EMBED_DIM, bias=False).to(device)

    # ── Loss ──────────────────────────────────────────────────────────
    seg_weights = compute_seg_class_weights(device)
    criterion   = MTLLoss(
        lambda_loc=args.lambda_loc,
        lambda_cls=args.lambda_cls,
        seg_weights=seg_weights,
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────
    # Group mask_proj parameters with model (train jointly)
    all_params = list(model.parameters()) + list(mask_proj.parameters())
    optimizer = torch.optim.Adam(
        all_params, lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    # ── LR Scheduler (paper: reduce by 0.5 every 15 epochs) ──────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE, min_lr=1e-6,
    )

    # ── Early stopping ────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_DELTA)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_miou  = 0.0
    if args.resume:
        start_epoch, best_val_miou = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device
        )

    # Mixed-precision scaler
    use_amp = AMP_ENABLED and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = args.run_name or f"pdlc_vit_enc{args.num_enc_layers}_bs{args.batch_size}"

    with mlflow.start_run(run_name=run_name, tags=MLFLOW_RUN_TAGS):
        # Log hyperparameters
        mlflow.log_params({
            "img_size":       IMG_SIZE,
            "patch_size":     PATCH_SIZE,
            "embed_dim":      EMBED_DIM,
            "num_heads":      NUM_HEADS,
            "num_enc_layers": args.num_enc_layers,
            "mlp_ratio":      MLP_RATIO,
            "dropout":        DROPOUT,
            "batch_size":     args.batch_size,
            "learning_rate":  args.lr,
            "weight_decay":   WEIGHT_DECAY,
            "lambda_loc":     args.lambda_loc,
            "lambda_cls":     args.lambda_cls,
            "num_seg_classes": NUM_SEG_CLASSES,
            "num_cls_classes": NUM_CLS_CLASSES,
            "num_parameters": n_params,
        })

        # ── Training Loop ────────────────────────────────────────────
        for epoch in range(start_epoch + 1, args.epochs + 1):
            t_epoch = time.time()
            print(f"\n[Epoch {epoch}/{args.epochs}]")

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, mask_proj, epoch,
                accum_steps=args.accum_steps, scaler=scaler, use_amp=use_amp
            )

            # Validate
            val_metrics = validate_one_epoch(
                model, val_loader, criterion, device,
                use_gt_tokens=False, mask_proj=mask_proj
            )

            # LR step
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["val_loss"])

            # ── Print epoch summary ───────────────────────────────────
            epoch_time = time.time() - t_epoch
            print(
                f"  Train Loss {train_metrics['train_loss']:.4f}"
                f" (seg {train_metrics['train_seg_loss']:.4f},"
                f" cls {train_metrics['train_cls_loss']:.4f}) | "
                f"Val Loss {val_metrics['val_loss']:.4f}"
                f" (seg {val_metrics['val_seg_loss']:.4f},"
                f" cls {val_metrics['val_cls_loss']:.4f}) | "
                f"mIoU {val_metrics['mIoU']:.4f} | "
                f"ClsAcc {val_metrics['top1_accuracy']:.4f} | "
                f"F1 {val_metrics['macro_f1']:.4f} | "
                f"LR {current_lr:.2e} | {epoch_time:.1f}s"
            )

            # ── MLflow logging ────────────────────────────────────────
            log_dict = {
                # Train metrics
                "train_loss":        train_metrics["train_loss"],
                "train_seg_loss":    train_metrics["train_seg_loss"],
                "train_cls_loss":    train_metrics["train_cls_loss"],
                # Val metrics
                "val_loss":          val_metrics["val_loss"],
                "val_seg_loss":      val_metrics["val_seg_loss"],
                "val_cls_loss":      val_metrics["val_cls_loss"],
                "val_mIoU":          val_metrics["mIoU"],
                "val_pixel_acc":     val_metrics["pixel_acc"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "val_macro_f1":      val_metrics["macro_f1"],
                "learning_rate":     current_lr,
            }

            # Per-class IoU (all 115 seg classes)
            for cls_i, iou_val in enumerate(val_metrics["iou_per_class"]):
                log_dict[f"val_iou_class_{cls_i}"] = iou_val

            mlflow.log_metrics(log_dict, step=epoch)

            # ── Checkpoint ───────────────────────────────────────────
            is_best = val_metrics["mIoU"] > best_val_miou
            if is_best:
                best_val_miou = val_metrics["mIoU"]

            save_checkpoint(
                state={
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "mask_proj_state_dict": mask_proj.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_miou":        best_val_miou,
                    "val_metrics":          {k: v for k, v in val_metrics.items()
                                             if k != "iou_per_class"},
                },
                filename=f"epoch_{epoch:04d}.pth",
                is_best=is_best,
            )

            # ── Early stopping ────────────────────────────────────────
            if early_stop(val_metrics["val_loss"]):
                print(f"\n[Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
                break

        # ── Log best model to MLflow ─────────────────────────────────
        print(f"\n[Training Complete] Best val mIoU: {best_val_miou:.4f}")
        mlflow.log_metric("best_val_mIoU", best_val_miou)

        best_ckpt = str(CHECKPOINT_DIR / "best_model.pth")
        if os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
