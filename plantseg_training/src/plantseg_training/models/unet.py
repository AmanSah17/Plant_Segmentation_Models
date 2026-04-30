from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from plantseg_training.metrics import bce_dice_loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = [
            ("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(out_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(out_channels)),
            ("relu2", nn.ReLU(inplace=True)),
        ]
        if dropout > 0:
            layers.append(("dropout", nn.Dropout2d(dropout)))
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetForBinarySegmentation(nn.Module):
    """U-Net with a Hugging Face Trainer-compatible forward signature."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("This first U-Net baseline expects num_classes=1 for binary masks.")

        self.down_blocks = nn.ModuleList()
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.down_blocks.append(ConvBlock(current_channels, feature, dropout=dropout))
            current_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout=dropout)

        current_channels = features[-1] * 2
        for feature in reversed(features):
            self.up_transposes.append(nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(feature * 2, feature, dropout=dropout))
            current_channels = feature

        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None, **_: Any) -> dict[str, torch.Tensor]:
        skips = []
        x = pixel_values
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, upsample in enumerate(self.up_transposes):
            x = upsample(x)
            skip = skips[idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.up_blocks[idx](x)

        logits = self.classifier(x)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = bce_dice_loss(logits, labels)
        return output
