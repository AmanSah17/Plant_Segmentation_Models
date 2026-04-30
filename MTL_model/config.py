"""
config.py
---------
Central configuration for the PDLC-ViT MTL pipeline.
All paths, hyperparameters, and constants live here.
"""

import os
from pathlib import Path

# ────────────────────────────────────────────────────────────
# PROJECT ROOTS
# ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("F:/PyTorch_GPU/Plant_Seg")

# Raw dataset (images + annotations)
DATA_ROOT       = PROJECT_ROOT / "Data_exploration/data/archive/plantsegv2"
IMAGES_DIR      = DATA_ROOT / "images"          # {train,val,test}/<name>.jpg
COCO_JSON       = DATA_ROOT / "coco_annotations.json"
METADATA_CSV    = DATA_ROOT / "Metadatav2.csv"

# Pre-processed multiclass masks (pixel = class_id 0-114)
MASKS_DIR       = PROJECT_ROOT / "plantseg_training/processed/multiclass/masks"  # {train,val,test}/<name>.png

# Reports / class info
REPORTS_DIR     = PROJECT_ROOT / "plantseg_training/processed/multiclass/reports"
CLASS_MAP_CSV   = REPORTS_DIR / "class_map.csv"
CLASS_PIXEL_CSV = REPORTS_DIR / "class_pixel_counts.csv"

# ────────────────────────────────────────────────────────────
# OUTPUT / ARTEFACTS
# ────────────────────────────────────────────────────────────
MTL_DIR              = PROJECT_ROOT / "MTL_model"
CHECKPOINT_DIR       = MTL_DIR / "checkpoints"
KFOLD_CHECKPOINT_DIR = MTL_DIR / "checkpoints" / "kfold"   # per-fold checkpoints
MLFLOW_URI           = "file:///" + str(MTL_DIR / "mlruns").replace("\\", "/")  # local MLflow tracking store

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
KFOLD_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────
# CLASS COUNTS  (derived from class_map.csv)
# ────────────────────────────────────────────────────────────
# Segmentation: 115 classes — 0=background, 1-114=disease
NUM_SEG_CLASSES = 115
# Classification: 114 disease classes — Index 0-113 (= class_id - 1)
NUM_CLS_CLASSES = 114
# Background class id in the mask
BACKGROUND_CLASS_ID = 0

# ────────────────────────────────────────────────────────────
# IMAGE / PATCH SETTINGS
# ────────────────────────────────────────────────────────────
IMG_SIZE    = 256           # ViT standard (512 needs >12 GB VRAM at batch 16)
PATCH_SIZE  = 16            # non-overlapping patch side
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196

# ImageNet normalisation stats
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS  (PDLC-ViT, paper Table 2)
# ────────────────────────────────────────────────────────────
EMBED_DIM       = 768       # patch embedding dimension d
NUM_HEADS       = 8         # multi-head attention heads (768/8 = 96)
MLP_RATIO       = 4         # FFN hidden dim = EMBED_DIM * MLP_RATIO
NUM_ENC_LAYERS  = 8         # transformer encoder depth
DROPOUT         = 0.2       # applied to encoder and decoder

# Co-scale: number of scales to aggregate
CO_SCALE_SIZES  = [1, 2, 4]  # pool patches to 1×1, 2×2, 4×4 and re-expand

# Classification head learnable query tokens (DETR-style for inference)
NUM_CLS_QUERIES = 1         # single class query
NUM_SEG_QUERIES = NUM_PATCHES  # one query per patch position

# ────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS  (paper Table 2)
# ────────────────────────────────────────────────────────────
BATCH_SIZE         = 16
NUM_WORKERS        = 4
MAX_EPOCHS         = 120

LEARNING_RATE      = 1e-4
LR_REDUCE_FACTOR   = 0.5
LR_REDUCE_PATIENCE = 15       # epochs with no val-loss improvement
WEIGHT_DECAY       = 5e-4     # L2 regularisation λ

# Early stopping: stop if val-loss improvement < threshold for patience epochs
EARLY_STOP_DELTA        = 1e-3
EARLY_STOP_PATIENCE     = 20
# Tighter patience for K-fold / HP-search (fewer epochs per fold)
EARLY_STOP_PATIENCE_KFOLD = 20

# Mixed-precision (AMP) — speeds up training and halves VRAM on Ampere+
AMP_ENABLED = True

# MTL loss weights
LAMBDA_LOC = 1.0
LAMBDA_CLS = 1.0

# ────────────────────────────────────────────────────────────
# MLFLOW
# ────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT  = "PDLC-ViT-MTL"
MLFLOW_RUN_TAGS    = {
    "model": "PDLC-ViT",
    "dataset": "PlantSegV2",
    "task": "MTL-segmentation-classification",
}

# ────────────────────────────────────────────────────────────
# K-FOLD CROSS-VALIDATION + HP TUNING
# ────────────────────────────────────────────────────────────
KFOLD_K         = 5          # default number of folds
KFOLD_EPOCHS    = 120        # max epochs per fold
N_OPTUNA_TRIALS = 10         # number of HP search trials

# Hyperparameter search space
HP_EMBED_DIMS    = [768]                 # locked to 768 as requested
HP_LR_VALUES     = [5e-5, 1e-4, 5e-4]    # decreased learning rates
HP_ENC_LAYERS    = [8]                   # locked to 8 as requested
HP_DROPOUT_VALS  = [0.1, 0.2, 0.3]       # candidate dropout rates

# ────────────────────────────────────────────────────────────
# MISC
# ────────────────────────────────────────────────────────────
SEED = 42
PIN_MEMORY = True
