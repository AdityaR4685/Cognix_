"""ResNet-18 backbone with a regression head for weather prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class WeatherPredictor(nn.Module):
    """ResNet-18 feature extractor + linear head → N weather scalars."""

    def __init__(self, n_outputs: int = 5) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        self.head = nn.Linear(512, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)  # (B, 512)
        return self.head(feat)              # (B, n_outputs)
