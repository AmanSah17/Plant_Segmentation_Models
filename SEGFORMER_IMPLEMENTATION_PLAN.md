# SegFormer PlantSegV2 Implementation Plan

## Goal

Fine-tune `nvidia/segformer-b2-finetuned-ade-512-512` for PlantSegV2 semantic segmentation with strong imbalance handling and experiment tracking.

## Dataset Strategy

The local dataset has paired files:

- Images: `plantsegv2/images/{train,val,test}/*.jpg`
- Masks: `plantsegv2/annotations/{train,val,test}/*.png`
- Metadata: `plantsegv2/Metadatav2.csv`
- COCO metadata: `plantsegv2/coco_annotations.json`

Mask convention:

- `0`: background
- `1..115`: diseased foreground class IDs, matching `Metadatav2.csv` disease `Index + 1`
- One metadata index is absent, so the training code creates a dense label space with an unused placeholder class.

Supported modes:

- `binary`: background vs diseased tissue. Best first experiment for robust lesion localization.
- `multiclass`: background plus disease-specific foreground classes. Best for disease-aware segmentation once the binary baseline is stable.

## Model Strategy

Use Hugging Face `SegformerForSemanticSegmentation` with:

- pretrained encoder/decoder from `nvidia/segformer-b2-finetuned-ade-512-512`
- replacement classifier head sized to the PlantSegV2 label space
- `ignore_mismatched_sizes=True`, expected because ADE20K has a different class count

## Imbalance Strategy

The implementation combines three controls:

1. Median-frequency pixel class weights for `CrossEntropyLoss`.
2. Focal loss to emphasize hard and rare pixels.
3. `WeightedRandomSampler` to oversample underrepresented disease images and images with meaningful foreground area.

Recommended loss sequence:

1. Start with `loss_name="ce_focal"`.
2. If training is unstable, use `loss_name="ce"` for a few epochs.
3. If rare disease classes remain weak, use `loss_name="focal"` and inspect foreground mIoU.

## Metrics and Tracking

Metrics logged to TensorBoard and MLflow:

- loss
- mean IoU
- foreground mean IoU
- pixel accuracy
- mean accuracy

Prefer validation foreground mIoU for checkpoint promotion. Pixel accuracy can look strong even when the model mostly predicts background.

## Execution

Activate the CUDA environment before running:

```powershell
cd F:\PyTorch_GPU\Plant_desease_segmentation
. F:\PyTorch_GPU\torch_gpu\Scripts\Activate.ps1
jupyter lab
```

Main notebook:

```text
notebooks/01_segformer_plantsegv2_training.ipynb
```

Reusable training module:

```text
src/segformer_training.py
```

TensorBoard:

```powershell
tensorboard --logdir outputs
```

MLflow:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

## Development Sequence

1. Run notebook environment and dataset audit cells.
2. Run class weight computation and inspect rare classes.
3. Run dataloader and batch visualization cells.
4. Run model forward smoke test.
5. Run `DEBUG_CONFIG` for one epoch.
6. Run full `CONFIG` training.
7. Compare runs in MLflow; promote by validation foreground mIoU.
8. Evaluate once on test after choosing the best validation checkpoint.

## References

- Hugging Face SegFormer documentation: https://huggingface.co/docs/transformers/model_doc/segformer
- SegFormer paper: https://arxiv.org/abs/2105.15203
- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch TensorBoard support: https://docs.pytorch.org/docs/stable/tensorboard.html
- MLflow PyTorch API: https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html
