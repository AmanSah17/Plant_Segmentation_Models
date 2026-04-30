from __future__ import annotations

from typing import Any

from plantseg_training.models.unet import UNetForBinarySegmentation


def build_model(model_cfg: dict[str, Any]):
    name = model_cfg["name"].lower()
    if name == "unet":
        return UNetForBinarySegmentation(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 1)),
            features=tuple(model_cfg.get("features", [64, 128, 256, 512])),
            dropout=float(model_cfg.get("dropout", 0.0)),
            loss=model_cfg.get("loss", {}),
        )
    raise ValueError(f"Unknown model name: {model_cfg['name']}")
