# Google Colab Training Guide: PDLC-ViT MTL

This guide explains how to transition the PDLC-ViT Multi-Task Learning pipeline from a local Windows environment to Google Colab for high-performance training on NVIDIA A100/L4 GPUs.

## 1. Prerequisites
1. Upload your `archive.zip` (containing the `plantsegv2` dataset) to your Google Drive.
2. Ensure your GitHub repository is public or you have a personal access token for cloning.

## 2. Step-by-Step Setup in Colab

Create a new Colab notebook and set the **Runtime Type** to **GPU**.

### Step 2.1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2.2: Extract Dataset to Local Runtime
*Training directly from Drive is extremely slow due to I/O overhead. Always unzip to the local `/content/` directory.*

```python
# Replace 'path/to/your/archive.zip' with the actual path in your Drive
!unzip -q "/content/drive/MyDrive/archive.zip" -d "/content/dataset"
```

### Step 2.3: Clone the Repository
```python
%cd /content
!git clone https://github.com/AmanSah17/Plant_Segmentation_Models.git
%cd Plant_Segmentation_Models
```

### Step 2.4: Install Dependencies
```python
!pip install -q albumentations==2.0.4 mlflow optuna opencv-python-headless
```

### Step 2.5: Patch the Configuration
The repository is configured for Windows paths (F: drive). Run this block to dynamically patch `config.py` for the Colab environment:

```python
import os

config_path = "MTL_model/config.py"
with open(config_path, "r") as f:
    content = f.read()

# Update Project Roots
content = content.replace('Path("F:/PyTorch_GPU/Plant_Seg")', 'Path("/content/Plant_Segmentation_Models")')
# Update Data Roots (pointing to unzipped dataset)
content = content.replace('PROJECT_ROOT / "Data_exploration/data/archive/plantsegv2"', 'Path("/content/dataset/plantsegv2")')

# Optimization for Colab A100/L4
content = content.replace('NUM_WORKERS        = 4', 'NUM_WORKERS        = 8')
content = content.replace('BATCH_SIZE         = 16', 'BATCH_SIZE         = 64')

with open(config_path, "w") as f:
    f.write(content)

print("✅ config.py successfully patched for Colab.")
```

## 3. Training Execution

Set the required environment variable and launch the training script. On Colab GPUs, you can typically use a `batch_size` of 64 with `accum_steps` of 1 or 2.

```python
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Run for 120 Epochs
!python MTL_model/train.py --epochs 120 --batch_size 64 --accum_steps 1
```

## 4. Monitoring Progress
You can launch an MLflow UI or sync the `mlruns` directory back to your Google Drive to persist the training logs:
```python
!cp -r /content/Plant_Segmentation_Models/MTL_model/mlruns /content/drive/MyDrive/PDLC_ViT_Logs
```
