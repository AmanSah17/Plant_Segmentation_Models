"""
SegFormer Fine-Tuning Script for Plant Disease Segmentation (PlantSegV2)
Optimized for: Local execution, Windows, NVIDIA GTX 1650 (4GB VRAM)

Features:
- Two-stage training: Smoke test (5 epochs, tiny subset) -> Full training.
- Optimized Dataloaders: num_workers=0 for Windows stability, vectorized sampler.
- Resource Efficient: AMP disabled (stability), small batch size, image resizing.
- Experiment Tracking: MLflow integration.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Add src to path if needed
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

import torch
import pandas as pd
import mlflow
from transformers import AutoImageProcessor

# Local imports
from segformer_training import (
    TrainConfig, 
    run_smoke_test, 
    train_model, 
    create_model,
    build_dataloaders
)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on PlantSegV2 dataset")
    parser.add_argument("--data_root", type=str, default="plantsegv2", help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (GTX 1650: 2 is safe)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of full training epochs")
    parser.add_argument("--skip_smoke", action="store_true", help="Skip the initial smoke test")
    args = parser.parse_args()

    # 1. Define Configuration
    # Optimized for GTX 1650: amp=False, num_workers=0
    config = TrainConfig(
        data_root=args.data_root,
        model_name="nvidia/segformer-b2-finetuned-ade-512-512",
        task_mode="multiclass",
        image_size=512,
        batch_size=args.batch_size,
        num_workers=0,  # CRITICAL for Windows
        epochs=args.epochs,
        lr=6e-5,
        weight_decay=1e-2,
        loss_name="ce_focal",
        use_class_weights=True,
        use_sampler=True,
        amp=False,      # CRITICAL for GTX 1650 stability
        experiment_name="plantsegv2-segformer-final",
        run_name="segformer-b2-optimized-run",
        output_dir=str(PROJECT_ROOT / "outputs" / "segformer_final"),
        mlflow_tracking_uri="file:" + str(PROJECT_ROOT / "mlruns"),
    )

    print("\n" + "="*60)
    print(" SEGFORMER FINE-TUNING PIPELINE ".center(60, "="))
    print("="*60)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Data: {config.data_root}")
    print(f"Output: {config.output_dir}")
    print("="*60 + "\n")

    # 2. Stage 1: Mandatory Smoke Test
    smoke_passed = True
    if not args.skip_smoke:
        print(">>> STAGE 1: Starting Smoke Test (5 epochs, 32 samples)...")
        try:
            # Note: run_smoke_test uses num_workers=0 internally for safety
            run_smoke_test(config, epochs=5, n_samples=32)
            print("\n✅ SMOKE TEST PASSED.")
        except Exception as e:
            print("\n❌ SMOKE TEST FAILED.")
            traceback.print_exc()
            smoke_passed = False
            sys.exit(1)
    else:
        print(">>> Skipping Smoke Test as requested.")

    # 3. Stage 2: Full Training
    if smoke_passed:
        print("\n>>> STAGE 2: Starting Full Training...")
        
        # Ensure MLflow is ready
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        
        # Enable CUDA blocking for better error reporting on Windows
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        try:
            results = train_model(config)
            
            print("\n" + "="*60)
            print(" TRAINING COMPLETE ".center(60, "="))
            print("="*60)
            print(f"Best Val mIoU: {results.get('best_val_miou', 0):.4f}")
            print(f"Best Checkpoint: {results.get('best_checkpoint', 'N/A')}")
            
            summary_path = Path(config.output_dir) / "final_summary.txt"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
            print(f"Summary saved to: {summary_path}")
            
        except Exception as e:
            print("\n❌ FULL TRAINING FAILED.")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
