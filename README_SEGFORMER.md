# PlantSegV2: SegFormer Fine-Tuning for Disease Segmentation

This branch contains the implementation of **SegFormer (B2)** for multiclass plant disease segmentation on the **PlantSegV2** dataset. The pipeline is optimized for local execution on mid-range hardware (e.g., NVIDIA GTX 1650) with specific robustness fixes for Windows environments.

## 🚀 Key Features

- **Two-Stage Training**: Includes a fast "Smoke Test" (5 epochs on a tiny subset) to validate the pipeline before committing to full training.
- **Optimized Data Pipeline**: 
    - Vectorized `WeightedRandomSampler` for instant startup.
    - Automated handling of mask label inconsistencies (clamping invalid pixels).
    - On-the-fly shape consistency checks (auto-resizing misaligned masks).
- **Hardware-Aware Settings**:
    - `amp=False` for GTX 1650 stability (prevents NaN logits).
    - `num_workers=0` for Windows notebook stability.
- **Experiment Tracking**: Integrated with **MLflow** for detailed metric logging and checkpoint management.

## 🛠️ Methodologies

### 1. Model Architecture
We use the **SegFormer** architecture, specifically the `nvidia/segformer-b2-finetuned-ade-512-512` checkpoint. SegFormer is preferred for its efficient Mix Transformer (MiT) encoder and simple MLP decoder, which achieves state-of-the-art performance with fewer parameters than traditional U-Net variants.

### 2. Loss Function: CE + Focal
To handle the class imbalance inherent in plant disease datasets (where background often dominates over tiny lesion areas), we use a **Combined Segmentation Loss**:
- **Weighted Cross Entropy**: Focuses on rare disease classes.
- **Focal Loss**: Penalizes "easy" background pixels and forces the model to focus on hard-to-segment lesion boundaries.

### 3. Robustness Optimizations
- **Label Clamping**: Some masks in the dataset contain pixel values outside the defined metadata range. Our loader automatically clamps these to an `ignore_index` (255) to prevent `CUDA device-side assert` crashes.
- **Shape Alignment**: Automatically detects and fixes dimension mismatches between images and masks using nearest-neighbor interpolation.

## 📋 How to Replicate

### 1. Environment Setup
```bash
# Recommended environment: Python 3.9+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers albumentations opencv-python pandas mlflow tqdm
```

### 2. Dataset Structure
Ensure your data is organized as follows:
```text
plantsegv2/
├── Metadatav2.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

### 3. Running Training
Use the optimized training script:
```bash
python train_segformer.py --epochs 150 --batch_size 2
```

### 4. Configuration
You can modify `TrainConfig` in `src/segformer_training.py` or via command-line arguments in `train_segformer.py` to tune the learning rate, model variant, or loss weights.

## 📊 Results & Tracking
Metric logs, including mIoU and Pixel Accuracy, are saved to the `mlruns` directory. Use `mlflow ui` to visualize training progress.
Checkpoints are saved in `outputs/segformer_final/` upon reaching new best Validation mIoU.
