"""
cache_dataset.py
----------------
Preprocesses the dataset (resize only, keeping uint8) and saves the resulting 
arrays to disk as `.pt` files. This skips the cv2.imread and initial resize 
bottleneck during training, while still allowing dynamic random augmentations.

Usage:
  python MTL_model/cache_dataset.py
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_ROOT, PROJECT_ROOT, IMG_SIZE
from dataset import PlantSegMTLDataset

CACHE_DIR = PROJECT_ROOT / "MTL_model" / "dataset_cache"

def cache_split(split: str):
    print(f"\n[Caching] Split: {split} (Size: {IMG_SIZE}x{IMG_SIZE})")
    
    # We load the dataset WITHOUT any transforms so we get raw uint8 arrays
    try:
        ds = PlantSegMTLDataset(split=split, transform=None)
    except ValueError as e:
        print(f"Skipping {split}: {e}")
        return
        
    split_cache_dir = CACHE_DIR / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(len(ds)), desc=f"Caching {split}"):
        # Since transform=None, dataset returns un-augmented but float tensors right now.
        # Wait, the fallback in dataset.py converts to float tensor!
        # Let's bypass dataset.py and just read manually to ensure uint8.
        row = ds.meta.iloc[i]
        
        if ds.split == "kfold":
            img_dir  = ds._get_img_dir(row)
            mask_dir = ds._get_mask_dir(row)
        else:
            img_dir  = ds.img_dir
            mask_dir = ds.mask_dir

        img_path = img_dir / row["Name"]
        image = cv2.imread(str(img_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        else:
            continue

        stem = Path(row["Name"]).stem + ".png"
        mask_path = mask_dir / stem
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Clip mask values to valid range
        from config import NUM_SEG_CLASSES
        mask = np.clip(mask, 0, NUM_SEG_CLASSES - 1).astype(np.uint8)

        cls_label = int(row["Index"])

        sample = {
            "image": image,          # uint8 numpy array [H, W, 3]
            "mask": mask,            # uint8 numpy array [H, W]
            "label": cls_label,
            "img_name": row["Name"],
            "disease": row["Disease"],
            "plant": row["Plant"],
        }
        
        save_path = split_cache_dir / f"{Path(row['Name']).stem}.pt"
        torch.save(sample, save_path)

if __name__ == "__main__":
    print(f"Starting Dataset Caching to: {CACHE_DIR}")
    for s in ["train", "val", "test"]:
        cache_split(s)
    print("\n[Done] All splits cached successfully.")

