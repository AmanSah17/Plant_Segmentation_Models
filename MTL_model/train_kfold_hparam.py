"""
train_kfold_hparam.py
---------------------
Combined Hyperparameter Search (Optuna) and K-Fold Cross Validation.

Usage:
  python MTL_model/train_kfold_hparam.py
  python MTL_model/train_kfold_hparam.py --k_folds 5 --n_trials 20 --batch_size 64 --accum_steps 4
"""

import sys
import os
import argparse
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import optuna
from sklearn.model_selection import StratifiedKFold

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED,
    IMG_SIZE, PATCH_SIZE, NUM_SEG_CLASSES, NUM_CLS_CLASSES,
    CO_SCALE_SIZES, WEIGHT_DECAY, LAMBDA_LOC, LAMBDA_CLS,
    MLFLOW_URI, MLFLOW_EXPERIMENT, MLFLOW_RUN_TAGS,
    KFOLD_CHECKPOINT_DIR, KFOLD_K, KFOLD_EPOCHS, N_OPTUNA_TRIALS,
    HP_EMBED_DIMS, HP_LR_VALUES, HP_ENC_LAYERS, HP_DROPOUT_VALS,
    EARLY_STOP_DELTA, EARLY_STOP_PATIENCE_KFOLD, AMP_ENABLED,
)

from dataset import PlantSegMTLDataset, get_full_trainval_meta
from augmentations import get_transform
from model import PDLCViT
from losses import MTLLoss, compute_seg_class_weights
from utils import set_seed, mask_to_patch_tokens, EarlyStopping
from validate import validate_one_epoch
from train import train_one_epoch


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Optuna HP Search + K-Fold CV")
    p.add_argument("--k_folds",      type=int, default=KFOLD_K)
    p.add_argument("--n_trials",     type=int, default=N_OPTUNA_TRIALS)
    p.add_argument("--epochs",       type=int, default=KFOLD_EPOCHS)
    p.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    p.add_argument("--accum_steps",  type=int, default=1,
                   help="Gradient accumulation steps (useful if batch_size is large)")
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--dry_run",      action="store_true",
                   help="Run 1 epoch/1 trial/2 folds for testing")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
def build_model_and_optim(hparams: dict, device: torch.device):
    """Factory function for model, criterion, and optimiser."""
    embed_dim = hparams["embed_dim"]
    num_heads = max(1, embed_dim // 64)  # Force head_dim = 64

    model = PDLCViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=embed_dim,
        num_heads=num_heads, num_enc_layers=hparams["num_enc_layers"],
        mlp_ratio=4.0, dropout=hparams["dropout"],
        num_seg_classes=NUM_SEG_CLASSES, num_cls_classes=NUM_CLS_CLASSES,
        scale_sizes=CO_SCALE_SIZES,
    ).to(device)

    mask_proj = nn.Linear(NUM_SEG_CLASSES, embed_dim, bias=False).to(device)
    
    seg_weights = compute_seg_class_weights(device)
    criterion = MTLLoss(seg_weights=seg_weights).to(device)

    all_params = list(model.parameters()) + list(mask_proj.parameters())
    optimizer = torch.optim.Adam(
        all_params, lr=hparams["lr"], weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    return model, mask_proj, criterion, optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────
def run_fold(
    fold_idx: int,
    train_idx,
    val_idx,
    meta_pool,
    hparams: dict,
    args,
    device: torch.device,
    is_trial: bool = False
):
    """
    Run training for a single fold.
    If is_trial=True, runs within Optuna context (reports to MLflow as a child run).
    """
    print(f"\n[{'Trial' if is_trial else 'CV'} - Fold {fold_idx}] Building DataLoaders...")

    # Build datasets for this fold
    train_ds = PlantSegMTLDataset.from_indices(
        meta_pool, train_idx, transform=get_transform("train")
    )
    val_ds = PlantSegMTLDataset.from_indices(
        meta_pool, val_idx, transform=get_transform("val")
    )

    # Adjust batch size for gradient accumulation
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

    model, mask_proj, criterion, optimizer, scheduler = build_model_and_optim(hparams, device)
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE_KFOLD, min_delta=EARLY_STOP_DELTA)

    # Mixed-precision scaler (no-op on CPU)
    use_amp = AMP_ENABLED and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_miou = 0.0
    best_metrics = {}

    run_name = f"Trial_Fold_{fold_idx}" if is_trial else f"CV_Fold_{fold_idx}"

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(hparams)
        mlflow.log_param("fold", fold_idx)

        epochs = 1 if args.dry_run else args.epochs

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            # --- TRAIN ---
            model.train()
            train_loss, train_seg, train_cls = 0.0, 0.0, 0.0
            n_batches = 0
            
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device, non_blocking=True)
                masks  = batch["mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    gt_mask_tokens = mask_to_patch_tokens(
                        masks, mask_proj, patch_size=PATCH_SIZE, embed_dim=hparams["embed_dim"]
                    )
                    seg_logits, cls_logits = model(images, gt_label=labels, gt_mask_tokens=gt_mask_tokens)
                    loss, l_seg, l_cls = criterion(seg_logits, cls_logits, masks, labels)
                    # Scale loss by accumulation steps
                    loss_scaled = loss / args.accum_steps

                scaler.scale(loss_scaled).backward()

                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item()
                train_seg  += l_seg.item()
                train_cls  += l_cls.item()
                n_batches  += 1

            train_metrics = {
                "train_loss": train_loss / n_batches,
                "train_seg_loss": train_seg / n_batches,
                "train_cls_loss": train_cls / n_batches,
            }

            # --- VALIDATE ---
            val_metrics = validate_one_epoch(
                model, val_loader, criterion, device,
                use_gt_tokens=False, mask_proj=mask_proj,
            )

            scheduler.step(val_metrics["val_loss"])

            # --- LOG ---
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "val_mIoU": val_metrics["mIoU"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": current_lr,
            }, step=epoch)

            print(
                f"  Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val mIoU: {val_metrics['mIoU']:.4f} | "
                f"{time.time()-t0:.1f}s"
            )

            if val_metrics["mIoU"] > best_val_miou:
                best_val_miou = val_metrics["mIoU"]
                best_metrics = val_metrics

            if early_stop(val_metrics["val_loss"]):
                print(f"  [Early Stop] Triggered at epoch {epoch}")
                break

    return best_metrics


# ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  PDLC-ViT: Optuna HP Search + K-Fold CV")
    print(f"  Device: {device} | Accum Steps: {args.accum_steps}")
    print("=" * 60)

    # 1. Load merged dataset
    meta_pool = get_full_trainval_meta()
    print(f"\n[Data] Loaded merged pool of {len(meta_pool)} samples.")
    
    y = meta_pool["Index"].values  # for stratified split
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=SEED)
    folds = list(skf.split(X=meta_pool, y=y))

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Optuna Hyperparameter Search (using Fold 0 only)
    # ─────────────────────────────────────────────────────────────────
    def objective(trial):
        hparams = {
            "embed_dim":      trial.suggest_categorical("embed_dim", HP_EMBED_DIMS),
            "lr":             trial.suggest_categorical("lr", HP_LR_VALUES),
            "num_enc_layers": trial.suggest_categorical("num_enc_layers", HP_ENC_LAYERS),
            "dropout":        trial.suggest_categorical("dropout", HP_DROPOUT_VALS),
        }
        
        train_idx, val_idx = folds[0]
        
        if args.dry_run:
            train_idx = train_idx[:100]
            val_idx   = val_idx[:50]

        best_metrics = run_fold(
            fold_idx=0,
            train_idx=train_idx,
            val_idx=val_idx,
            meta_pool=meta_pool,
            hparams=hparams,
            args=args,
            device=device,
            is_trial=True
        )
        
        # We want to maximize mIoU
        return best_metrics.get("mIoU", 0.0)

    print("\n" + "=" * 60)
    print(f"  PHASE 1: Hyperparameter Search ({args.n_trials} trials)")
    print("=" * 60)

    with mlflow.start_run(run_name="Optuna_Search_Parent"):
        study = optuna.create_study(direction="maximize")
        n_trials = 1 if args.dry_run else args.n_trials
        study.optimize(objective, n_trials=n_trials)

        best_hparams = study.best_trial.params
        print("\n[Optuna] Best Trial:")
        for k, v in best_hparams.items():
            print(f"  {k}: {v}")
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_search_mIoU", study.best_trial.value)

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: Full K-Fold CV with Best HParams
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  PHASE 2: Full {args.k_folds}-Fold CV with Best Config")
    print("=" * 60)

    kfold_metrics = []

    k_folds_to_run = 2 if args.dry_run else args.k_folds

    with mlflow.start_run(run_name="KFold_Eval_Parent"):
        mlflow.log_params(best_hparams)
        
        for k in range(k_folds_to_run):
            train_idx, val_idx = folds[k]
            
            if args.dry_run:
                train_idx = train_idx[:100]
                val_idx   = val_idx[:50]

            metrics = run_fold(
                fold_idx=k,
                train_idx=train_idx,
                val_idx=val_idx,
                meta_pool=meta_pool,
                hparams=best_hparams,
                args=args,
                device=device,
                is_trial=False
            )
            kfold_metrics.append(metrics)

            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"fold_{k}_{key}", val)

        # ─────────────────────────────────────────────────────────────────
        # Phase 3: Summary
        # ─────────────────────────────────────────────────────────────────
        import numpy as np
        
        miou_list = [m["mIoU"] for m in kfold_metrics]
        f1_list   = [m["macro_f1"] for m in kfold_metrics]
        acc_list  = [m["top1_accuracy"] for m in kfold_metrics]

        print("\n" + "=" * 60)
        print("  FINAL CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  mIoU:     {np.mean(miou_list):.4f} ± {np.std(miou_list):.4f}")
        print(f"  Macro F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
        print(f"  Top-1 Acc:{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")

        mlflow.log_metric("cv_mean_mIoU", np.mean(miou_list))
        mlflow.log_metric("cv_std_mIoU",  np.std(miou_list))


if __name__ == "__main__":
    main()
