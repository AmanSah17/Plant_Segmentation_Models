"""
augmentations.py
----------------
Albumentations-based transforms for train / val / test splits.

Rules (per user requirement):
  - Train : full augmentation
  - Val   : augmentation (resize + light normalise only)
  - Test  : NO augmentation — only resize + normalise
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE, IMG_MEAN, IMG_STD


def get_train_transform():
    """
    Heavy augmentation for the training set.
    All spatial transforms are applied jointly to image AND mask.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(0.6, 1.0), ratio=(0.75, 1.33),
            interpolation=1, p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(1, IMG_SIZE // 8), hole_width_range=(1, IMG_SIZE // 8),
            fill=0, fill_mask=0, p=0.2
        ),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    """
    Light augmentation for the validation set:
    resize, centre-crop, normalise.
    """
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def get_test_transform():
    """
    Test set: strictly NO augmentation — only resize + normalise.
    """
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def get_transform(split: str):
    """
    Helper: return the correct transform for a given split name.
    split in {'train', 'Training', 'val', 'Validation', 'test', 'Test'}
    """
    s = split.lower()
    if s in ("train", "training"):
        return get_train_transform()
    elif s in ("val", "validation"):
        return get_val_transform()
    elif s in ("test",):
        return get_test_transform()
    else:
        raise ValueError(f"Unknown split: {split!r}")
