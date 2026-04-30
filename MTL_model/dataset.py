"""
dataset.py
----------
PlantSegMTLDataset — joint segmentation + classification dataset.

Segmentation label:
  PNG mask files under  masks/{split}/<stem>.png
  Pixel value = class_id  (0 = background, 1-114 = disease)
  Output: LongTensor [H, W]  (values 0-114)

Classification label:
  Loaded from Metadatav2.csv  →  column "Index"  (0-indexed, range 0-113)
  Output: scalar LongTensor
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    IMAGES_DIR, MASKS_DIR, METADATA_CSV,
    NUM_SEG_CLASSES, NUM_CLS_CLASSES,
)


def get_full_trainval_meta() -> pd.DataFrame:
    """
    Return merged train + val rows from Metadatav2.csv.
    Used by K-fold CV to build the combined pool before splitting.
    Excludes background-only samples (Index < 0).
    """
    meta = pd.read_csv(METADATA_CSV)
    meta["_split_key"] = meta["Split"].str.lower()
    pool = meta[meta["_split_key"].isin(["train", "training", "val", "validation"])].copy()
    # Keep only valid classification label indices [0, NUM_CLS_CLASSES)
    pool = pool[(pool["Index"] >= 0) & (pool["Index"] < NUM_CLS_CLASSES)].reset_index(drop=True)
    return pool


class PlantSegMTLDataset(Dataset):
    """
    Parameters
    ----------
    split : str
        One of  'train' | 'val' | 'test'
    transform : albumentations.Compose | None
        Joint image+mask transform. If None, only converts to tensor.
    """

    # Map Metadatav2.csv "Split" column values to folder names
    _SPLIT_MAP = {
        "training": "train",
        "validation": "val",
        "test": "test",
    }

    def __init__(self, split: str, transform=None):
        super().__init__()
        self.split = split.lower()
        self.folder = self._SPLIT_MAP.get(self.split, self.split)
        self.transform = transform

        self.img_dir  = IMAGES_DIR / self.folder
        self.mask_dir = MASKS_DIR  / self.folder

        # Load metadata and filter for this split
        meta = pd.read_csv(METADATA_CSV)
        # Normalise the "Split" column
        meta["_split_key"] = meta["Split"].str.lower()
        self.meta = meta[meta["_split_key"] == self.split].reset_index(drop=True)

        if len(self.meta) == 0:
            raise ValueError(
                f"No entries found for split='{split}' in {METADATA_CSV}.\n"
                f"Available values: {meta['Split'].unique().tolist()}"
            )

        # Validate class indices
        bad = self.meta[(self.meta["Index"] < 0) | (self.meta["Index"] >= NUM_CLS_CLASSES)]
        if len(bad) > 0:
            # Exclude background-only images and out-of-bounds indices
            self.meta = self.meta[(self.meta["Index"] >= 0) & (self.meta["Index"] < NUM_CLS_CLASSES)].reset_index(drop=True)

        self._verify_files_exist()

    # ------------------------------------------------------------------
    @classmethod
    def from_indices(
        cls,
        meta_df: pd.DataFrame,
        indices,
        transform=None,
        folder: str = "train",
    ) -> "PlantSegMTLDataset":
        """
        Create a dataset view from an arbitrary subset of a pre-loaded DataFrame.
        Used by K-fold cross-validation to build per-fold train/val subsets without
        re-reading the CSV file on every fold.

        Parameters
        ----------
        meta_df  : pd.DataFrame — full merged pool (from get_full_trainval_meta())
        indices  : array-like int — row indices into meta_df for this subset
        transform : albumentations.Compose | None
        folder   : str — images/masks sub-folder ('train' is fine since all pool
                   samples are from train or val and share the same image roots)
        """
        instance = object.__new__(cls)
        instance.split     = "kfold"
        instance.folder    = folder
        instance.transform = transform
        # Build img_dir / mask_dir using the "train" folder as canonical root
        # (images may live in train/ or val/ — we resolve per-sample below)
        instance.img_dir   = IMAGES_DIR  # base; actual sub-folder per row
        instance.mask_dir  = MASKS_DIR   # base; actual sub-folder per row
        instance.meta      = meta_df.iloc[list(indices)].reset_index(drop=True)
        # Derive per-row image directory from the original Split column
        return instance

    def _get_img_dir(self, row) -> Path:
        """Return the correct images sub-folder for a given metadata row."""
        split_key = str(row.get("_split_key", "train")).lower()
        folder = self._SPLIT_MAP.get(split_key, "train")
        return IMAGES_DIR / folder

    def _get_mask_dir(self, row) -> Path:
        """Return the correct masks sub-folder for a given metadata row."""
        split_key = str(row.get("_split_key", "train")).lower()
        folder = self._SPLIT_MAP.get(split_key, "train")
        return MASKS_DIR / folder

    # ------------------------------------------------------------------
    def _verify_files_exist(self):
        """Warn (don't crash) if a handful of files are missing."""
        missing_img  = 0
        missing_mask = 0
        for row in self.meta.itertuples():
            img_path  = self.img_dir  / row.Name
            stem      = Path(row.Name).stem + ".png"
            mask_path = self.mask_dir / stem
            if not img_path.exists():
                missing_img += 1
            if not mask_path.exists():
                missing_mask += 1
        if missing_img or missing_mask:
            print(
                f"[PlantSegMTLDataset] split='{self.split}': "
                f"{missing_img} missing images, {missing_mask} missing masks "
                f"(out of {len(self.meta)} total)"
            )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.meta)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]

        # ── Image ──────────────────────────────────────────────────
        # For K-fold subsets resolve each sample's folder from its Split column
        if self.split == "kfold":
            img_dir  = self._get_img_dir(row)
            mask_dir = self._get_mask_dir(row)
        else:
            img_dir  = self.img_dir
            mask_dir = self.mask_dir

        img_path = img_dir / row["Name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H×W×3, uint8

        # ── Mask ──────────────────────────────────────────────────
        stem      = Path(row["Name"]).stem + ".png"
        mask_path = mask_dir / stem
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # H×W, uint8
        if mask is None:
            # If mask is missing, create a zeros mask (background only)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # Clip mask values to valid range (0 to NUM_SEG_CLASSES-1)
        mask = np.clip(mask, 0, NUM_SEG_CLASSES - 1).astype(np.uint8)

        # ── Classification label ────────────────────────────────────────
        cls_label = int(row["Index"])  # 0-indexed, range 0-113

        # ── Augmentation ───────────────────────────────────────────────
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]   # Tensor [3, H, W]  (float)
            mask  = transformed["mask"]    # Tensor [H, W]     (long after cast below)
        else:
            # Fallback: plain numpy → tensor (no normalisation)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask)

        # Ensure correct dtypes
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image))
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask))

        mask  = mask.long()                          # [H, W]
        label = torch.tensor(cls_label, dtype=torch.long)  # scalar

        return {
            "image":     image,    # FloatTensor [3, IMG_SIZE, IMG_SIZE]
            "mask":      mask,     # LongTensor  [IMG_SIZE, IMG_SIZE]
            "label":     label,    # LongTensor  scalar (0-113)
            "img_name":  row["Name"],
            "disease":   row["Disease"],
            "plant":     row["Plant"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        """Default collate — keeps img_name / disease / plant as lists."""
        images   = torch.stack([b["image"]  for b in batch])
        masks    = torch.stack([b["mask"]   for b in batch])
        labels   = torch.stack([b["label"]  for b in batch])
        names    = [b["img_name"] for b in batch]
        diseases = [b["disease"]  for b in batch]
        plants   = [b["plant"]    for b in batch]
        return {
            "image":    images,
            "mask":     masks,
            "label":    labels,
            "img_name": names,
            "disease":  diseases,
            "plant":    plants,
        }
