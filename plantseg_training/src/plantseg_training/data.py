from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


SPLIT_TO_DIR = {
    "Training": "train",
    "Validation": "val",
    "Test": "test",
}


@dataclass(frozen=True)
class PlantSegDataConfig:
    root_dir: Path
    metadata_csv: Path
    image_size: int = 256
    task: str = "binary_segmentation"


class PlantSegDataset(Dataset):
    """PlantSeg image/mask dataset for semantic segmentation."""

    def __init__(self, cfg: PlantSegDataConfig, split: str) -> None:
        if cfg.task != "binary_segmentation":
            raise ValueError(f"Unsupported task for this baseline: {cfg.task}")

        self.cfg = cfg
        self.split = split
        self.split_dir = SPLIT_TO_DIR[split]
        metadata = pd.read_csv(cfg.metadata_csv)
        self.frame = metadata.loc[metadata["Split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows found for split {split!r} in {cfg.metadata_csv}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.cfg.root_dir / "images" / self.split_dir / row["Name"]
        mask_path = self.cfg.root_dir / "annotations" / self.split_dir / row["Label file"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [self.cfg.image_size, self.cfg.image_size], antialias=True)
        mask = TF.resize(mask, [self.cfg.image_size, self.cfg.image_size], interpolation=Image.Resampling.NEAREST)

        image_tensor = TF.to_tensor(image)
        mask_array = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.float32)).unsqueeze(0)

        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def make_datasets(dataset_cfg: dict[str, Any]) -> tuple[PlantSegDataset, PlantSegDataset, PlantSegDataset]:
    cfg = PlantSegDataConfig(
        root_dir=Path(dataset_cfg["root_dir"]),
        metadata_csv=Path(dataset_cfg["metadata_csv"]),
        image_size=int(dataset_cfg.get("image_size", 256)),
        task=dataset_cfg.get("task", "binary_segmentation"),
    )
    return (
        PlantSegDataset(cfg, "Training"),
        PlantSegDataset(cfg, "Validation"),
        PlantSegDataset(cfg, "Test"),
    )


def segmentation_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
    }
