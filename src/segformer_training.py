from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import albumentations as A
import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


SPLIT_TO_DIR = {"Training": "train", "Validation": "val", "Test": "test"}


@dataclass
class TrainConfig:
    data_root: str = "plantsegv2"
    model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    task_mode: str = "multiclass"  # "binary" or "multiclass"
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 20
    lr: float = 6e-5
    weight_decay: float = 1e-2
    focal_gamma: float = 2.0
    loss_name: str = "focal"  # "ce", "focal", or "ce_focal"
    focal_weight: float = 1.0
    ce_weight: float = 0.25
    use_class_weights: bool = True
    use_sampler: bool = True
    max_weight: float = 10.0
    ignore_index: int = 255
    amp: bool = True
    grad_clip_norm: float = 1.0
    seed: int = 42
    experiment_name: str = "plantsegv2-segformer"
    run_name: str = "segformer-b2-custom"
    output_dir: str = "outputs/segformer_b2_plantsegv2"
    mlflow_tracking_uri: str = "file:./mlruns"
    # Smoke-test knobs: when smoke_test=True, only smoke_n_samples samples are used
    smoke_test: bool = False
    smoke_n_samples: int = 32  # total samples across train+val+test for smoke test


def resolve_project_root(start: str | Path) -> Path:
    start = Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "plantsegv2").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root above: {start}")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # GTX 1650/Pascal can hang on first iteration if benchmark is True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_metadata(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    meta = pd.read_csv(data_root / "Metadatav2.csv")
    meta["split_dir"] = meta["Split"].map(SPLIT_TO_DIR)
    meta["image_path"] = meta.apply(lambda r: data_root / "images" / r["split_dir"] / r["Name"], axis=1)
    meta["mask_path"] = meta.apply(lambda r: data_root / "annotations" / r["split_dir"] / r["Label file"], axis=1)
    meta = meta[meta["image_path"].map(Path.exists) & meta["mask_path"].map(Path.exists)].copy()
    meta["Index"] = meta["Index"].astype(int)
    meta["mask_ratio"] = meta["Mask ratio"].astype(float)
    return meta


def build_label_maps(meta: pd.DataFrame, task_mode: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    if task_mode == "binary":
        id2label = {0: "background", 1: "diseased_tissue"}
    elif task_mode == "multiclass":
        pairs = meta[["Index", "Disease"]].drop_duplicates().sort_values("Index")
        max_label_id = int(pairs["Index"].max()) + 1
        id2label = {0: "background"}
        for label_id in range(1, max_label_id + 1):
            id2label[label_id] = f"unused_class_{label_id}"
        for _, row in pairs.iterrows():
            id2label[int(row["Index"]) + 1] = str(row["Disease"])
    else:
        raise ValueError("task_mode must be 'binary' or 'multiclass'")
    label2id = {name: idx for idx, name in id2label.items()}
    return id2label, label2id


def get_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.15),
                A.Affine(
                    translate_percent=(-0.04, 0.04),
                    scale=(0.88, 1.12),
                    rotate=(-20, 20),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.35),
                A.HueSaturationValue(p=0.2),
            ],
            is_check_shapes=False
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            ],
            is_check_shapes=False
        )


class PlantSegDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_processor: AutoImageProcessor,
        task_mode: str,
        image_size: int,
        train: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_processor = image_processor
        self.task_mode = task_mode
        self.transforms = get_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(row["mask_path"]), dtype=np.int64)
        
        # Robustness fix: Ensure image and mask have same dimensions
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        if self.task_mode == "binary":
            mask = (mask > 0).astype(np.int64)

        augmented = self.transforms(image=image, mask=mask)
        
        encoded = self.image_processor(
            images=augmented["image"],
            segmentation_maps=augmented["mask"],
            return_tensors="pt",
        )
        
        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"].squeeze(0).long()
        
        # Final safety check before returning to loader
        # SegformerImageProcessor uses 255 as default ignore_index if reduce_labels is False
        num_classes = self.image_processor.num_labels
        labels[labels >= num_classes] = 255 
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": str(row["image_path"]),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
    }


def compute_pixel_counts(
    frame: pd.DataFrame,
    num_labels: int,
    task_mode: str,
    cache_path: str | Path,
) -> np.ndarray:
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("num_labels") == num_labels and cached.get("task_mode") == task_mode:
            return np.asarray(cached["counts"], dtype=np.float64)

    counts = np.zeros(num_labels, dtype=np.float64)
    for mask_path in tqdm(frame["mask_path"], desc="Counting mask pixels"):
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)
        if task_mode == "binary":
            mask = (mask > 0).astype(np.int64)
        valid = (mask >= 0) & (mask < num_labels)
        counts += np.bincount(mask[valid].ravel(), minlength=num_labels)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"num_labels": num_labels, "task_mode": task_mode, "counts": counts.tolist()}, indent=2),
        encoding="utf-8",
    )
    return counts


def make_class_weights(counts: np.ndarray, max_weight: float = 10.0) -> torch.Tensor:
    counts = counts.astype(np.float64)
    nonzero = counts > 0
    weights = np.ones_like(counts, dtype=np.float64)
    median = np.median(counts[nonzero])
    weights[nonzero] = median / counts[nonzero]
    weights = np.clip(weights, 0.05, max_weight)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    """Vectorised WeightedRandomSampler - avoids slow iterrows on large DataFrames."""
    disease_counts = frame["Disease"].value_counts()
    # Map disease -> 1/count using vectorised lookup
    disease_weight_series = 1.0 / frame["Disease"].map(disease_counts)
    lesion_boost = 1.0 + frame["mask_ratio"].clip(upper=0.50)
    sample_weights = (disease_weight_series * lesion_boost).astype(float)
    weights_tensor = torch.tensor(sample_weights.values, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)


class SegmentationFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        valid = targets != self.ignore_index
        ce_valid = ce[valid]
        if ce_valid.numel() == 0:
            return logits.sum() * 0.0
        pt = torch.exp(-ce_valid)
        loss = (1.0 - pt).pow(self.gamma) * ce_valid
        if self.alpha is not None:
            target_valid = targets[valid]
            alpha_t = self.alpha.to(logits.device).gather(0, target_valid)
            loss = alpha_t * loss
        return loss.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        loss_name: str,
        class_weights: Optional[torch.Tensor],
        ignore_index: int,
        focal_gamma: float,
        ce_weight: float,
        focal_weight: float,
    ) -> None:
        super().__init__()
        self.loss_name = loss_name
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal = SegmentationFocalLoss(gamma=focal_gamma, alpha=class_weights, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "ce":
            return self.ce(logits, targets)
        if self.loss_name == "focal":
            return self.focal(logits, targets)
        if self.loss_name == "ce_focal":
            return self.ce_weight * self.ce(logits, targets) + self.focal_weight * self.focal(logits, targets)
        raise ValueError("loss_name must be 'ce', 'focal', or 'ce_focal'")


@torch.no_grad()
def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_labels: int,
    ignore_index: int,
) -> torch.Tensor:
    valid = targets != ignore_index
    preds = preds[valid].view(-1)
    targets = targets[valid].view(-1)
    keep = (targets >= 0) & (targets < num_labels)
    inds = num_labels * targets[keep] + preds[keep].clamp(0, num_labels - 1)
    confusion += torch.bincount(inds, minlength=num_labels**2).reshape(num_labels, num_labels).to(confusion.device)
    return confusion


def metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, float]:
    confusion = confusion.float()
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = torch.zeros_like(tp)
    iou[valid] = tp[valid] / denom[valid].clamp_min(1.0)
    pixel_acc = tp.sum() / confusion.sum().clamp_min(1.0)
    mean_acc = (tp[valid] / confusion.sum(dim=1)[valid].clamp_min(1.0)).mean() if valid.any() else torch.tensor(0.0)
    return {
        "miou": float(iou[valid].mean().item()) if valid.any() else 0.0,
        "pixel_accuracy": float(pixel_acc.item()),
        "mean_accuracy": float(mean_acc.item()),
        "foreground_miou": float(iou[1:][valid[1:]].mean().item()) if valid[1:].any() else 0.0,
    }


def create_model(config: TrainConfig, id2label: Dict[int, str], label2id: Dict[str, int]) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        semantic_loss_ignore_index=config.ignore_index,
    )
    return model


def build_dataloaders(
    config: TrainConfig,
    meta: pd.DataFrame,
    image_processor: AutoImageProcessor,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_frame = meta[meta["split_dir"] == "train"].copy()
    val_frame = meta[meta["split_dir"] == "val"].copy()
    test_frame = meta[meta["split_dir"] == "test"].copy()

    # Smoke-test mode: subsample each split so an epoch finishes in seconds
    if config.smoke_test:
        n = max(config.smoke_n_samples, config.batch_size * 2)
        n_train = max(int(n * 0.7), config.batch_size)
        n_val = max(int(n * 0.2), config.batch_size)
        n_test = max(int(n * 0.1), config.batch_size)
        train_frame = train_frame.sample(min(n_train, len(train_frame)), random_state=config.seed).reset_index(drop=True)
        val_frame = val_frame.sample(min(n_val, len(val_frame)), random_state=config.seed).reset_index(drop=True)
        test_frame = test_frame.sample(min(n_test, len(test_frame)), random_state=config.seed).reset_index(drop=True)
        print(f"[SMOKE TEST] Using {len(train_frame)} train / {len(val_frame)} val / {len(test_frame)} test samples")

    train_ds = PlantSegDataset(train_frame, image_processor, config.task_mode, config.image_size, train=True)
    val_ds = PlantSegDataset(val_frame, image_processor, config.task_mode, config.image_size, train=False)
    test_ds = PlantSegDataset(test_frame, image_processor, config.task_mode, config.image_size, train=False)

    # pin_memory only makes sense when num_workers > 0 (Windows: num_workers=0 is common)
    use_pin_memory = config.num_workers > 0
    # persistent_workers saves fork/join overhead when num_workers > 0
    use_persistent = config.num_workers > 0

    # In smoke-test mode, don't use sampler so we don't re-build weights for tiny subset
    sampler = make_sampler(train_frame) if (config.use_sampler and not config.smoke_test) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def forward_logits(model: nn.Module, pixel_values: torch.Tensor, label_shape: Tuple[int, int]) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    return F.interpolate(logits, size=label_shape, mode="bilinear", align_corners=False)


@torch.no_grad()
def amp_forward_is_stable(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> bool:
    if device.type != "cuda":
        return False
    try:
        batch = next(iter(loader))
    except StopIteration:
        return False

    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    was_training = model.training
    model.eval()
    try:
        with autocast(device_type="cuda", enabled=True):
            logits = forward_logits(model, pixel_values, label_shape=labels.shape[-2:])
        return bool(torch.isfinite(logits).all().item())
    finally:
        model.train(was_training)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_labels: int,
    ignore_index: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    amp: bool = True,
    grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = torch.zeros((num_labels, num_labels), dtype=torch.int64, device=device)

    iterator = tqdm(loader, leave=False, desc="train" if training else "eval", disable=False)
    for i, batch in enumerate(iterator):
        if i == 0:
            print(f"DEBUG: Processing first { 'train' if training else 'val' } batch...")
        
        # Windows/GTX 1650: non_blocking=True can sometimes cause PCIe sync issues
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if i == 0:
            print(f"DEBUG: Batch moved to {device}. Running forward pass...")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = forward_logits(model, pixel_values, label_shape=labels.shape[-2:])
            loss = criterion(logits, labels)

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite logits detected. Disable AMP for this run by setting CONFIG.amp = False "
                "or use the train_model auto-fallback path."
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss detected. This usually means AMP instability or an invalid batch."
            )

        if training:
            if i == 0:
                print("DEBUG: Running backward pass and optimizer step...")
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        
        if i == 0:
            print(f"DEBUG: Iteration 0 complete.")

        preds = torch.argmax(logits.detach(), dim=1)
        update_confusion_matrix(confusion, preds, labels, num_labels, ignore_index)
        total_loss += float(loss.detach().item()) * pixel_values.size(0)
        iterator.set_postfix(loss=float(loss.detach().item()))

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def save_checkpoint(
    model: nn.Module,
    image_processor: AutoImageProcessor,
    output_dir: str | Path,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
) -> Path:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    image_processor.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (checkpoint_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return checkpoint_dir


def train_model(config: TrainConfig) -> Dict[str, Any]:
    """Main training loop. Supports smoke_test mode via config.smoke_test=True."""
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = load_metadata(config.data_root)
    id2label, label2id = build_label_maps(meta, config.task_mode)
    num_labels = len(id2label)
    print(f"Labels: {num_labels} | Task: {config.task_mode} | Smoke: {config.smoke_test}")

    image_processor = AutoImageProcessor.from_pretrained(
        config.model_name,
        do_resize=False,
        do_reduce_labels=False,
        use_fast=False,
    )
    # Ensure image_processor knows about the current num_labels for clamping
    image_processor.num_labels = num_labels
    
    train_loader, val_loader, test_loader = build_dataloaders(config, meta, image_processor)
    print(f"Batches per epoch — train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}")

    # In smoke-test mode, skip the expensive full-mask pixel count scan; use uniform weights
    if config.smoke_test or not config.use_class_weights:
        class_weights = None
        if config.smoke_test:
            print("[SMOKE TEST] Skipping pixel count computation — using uniform class weights.")
    else:
        counts = compute_pixel_counts(
            meta[meta["split_dir"] == "train"],
            num_labels=num_labels,
            task_mode=config.task_mode,
            cache_path=output_dir / f"pixel_counts_{config.task_mode}.json",
        )
        class_weights = make_class_weights(counts, config.max_weight).to(device)

    model = create_model(config, id2label, label2id).to(device)
    amp_enabled = bool(config.amp and device.type == "cuda")
    if amp_enabled and not amp_forward_is_stable(model, train_loader, device):
        print("AMP warmup check failed. Falling back to full precision (amp=False).")
        amp_enabled = False
    print(f"AMP enabled: {amp_enabled}")

    criterion = CombinedSegmentationLoss(
        loss_name=config.loss_name,
        class_weights=class_weights,
        ignore_index=config.ignore_index,
        focal_gamma=config.focal_gamma,
        ce_weight=config.ce_weight,
        focal_weight=config.focal_weight,
    )
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = GradScaler(device="cuda", enabled=amp_enabled)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    best_miou = -math.inf
    best_path: Optional[Path] = None
    history: List[Dict[str, float]] = []

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(asdict(config))
        mlflow.log_params({"num_labels": num_labels, "device": str(device), "amp_enabled": amp_enabled})
        if class_weights is not None:
            mlflow.log_dict({"class_weights": class_weights.detach().cpu().tolist()}, "class_weights.json")

        for epoch in range(1, config.epochs + 1):
            print(f"\n--- Epoch {epoch}/{config.epochs} ---")
            train_metrics = run_one_epoch(
                model,
                train_loader,
                criterion,
                device,
                num_labels,
                config.ignore_index,
                optimizer=optimizer,
                scaler=scaler,
                amp=amp_enabled,
                grad_clip_norm=config.grad_clip_norm,
            )
            val_metrics = run_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                num_labels,
                config.ignore_index,
                amp=amp_enabled,
            )
            scheduler.step()

            row = {f"train_{k}": v for k, v in train_metrics.items()}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["epoch"] = epoch
            row["lr"] = scheduler.get_last_lr()[0]
            history.append(row)

            print(
                f"  train_loss={train_metrics['loss']:.4f}  train_miou={train_metrics['miou']:.4f}"
                f"  val_loss={val_metrics['loss']:.4f}  val_miou={val_metrics['miou']:.4f}"
                f"  lr={row['lr']:.2e}"
            )

            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
                    mlflow.log_metric(key, value, step=epoch)

            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                best_path = save_checkpoint(model, image_processor, output_dir, epoch, val_metrics, config)
                mlflow.log_artifacts(str(best_path), artifact_path="best_checkpoint")
                print(f"  ✓ New best val mIoU: {best_miou:.4f} → checkpoint saved")

            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print("\nRunning test evaluation...")
        test_metrics = run_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            num_labels,
            config.ignore_index,
            amp=amp_enabled,
        )
        for key, value in test_metrics.items():
            writer.add_scalar(f"test_{key}", value, config.epochs)
            mlflow.log_metric(f"test_{key}", value)
        mlflow.log_artifact(str(output_dir / "history.csv"))
        print(f"Test metrics: {test_metrics}")

    writer.close()
    return {
        "best_checkpoint": str(best_path) if best_path else None,
        "best_val_miou": best_miou,
        "test_metrics": test_metrics,
        "history": history,
    }


def run_smoke_test(config: TrainConfig, epochs: int = 5, n_samples: int = 32) -> Dict[str, Any]:
    """Run a fast smoke test on a tiny dataset subset before full training.

    Args:
        config: Base TrainConfig to derive from.
        epochs: Number of smoke-test epochs (default 5).
        n_samples: Total samples across all splits (default 32).

    Returns:
        Results dict identical to train_model().
    """
    from dataclasses import replace as dc_replace

    smoke_config = dc_replace(
        config,
        smoke_test=True,
        smoke_n_samples=n_samples,
        epochs=epochs,
        use_class_weights=False,   # skip expensive pixel scan
        use_sampler=False,         # no sampler needed for tiny subset
        run_name=config.run_name + "-smoke",
        output_dir=str(Path(config.output_dir).with_name(Path(config.output_dir).name + "_smoke")),
        experiment_name=config.experiment_name + "-smoke",
    )
    print("=" * 60)
    print(f"SMOKE TEST: {epochs} epochs, ~{n_samples} samples total")
    print("=" * 60)
    results = train_model(smoke_config)
    print("=" * 60)
    print("SMOKE TEST COMPLETE")
    print(f"  Best val mIoU : {results['best_val_miou']:.4f}")
    print(f"  Test metrics  : {results['test_metrics']}")
    print("=" * 60)
    return results
