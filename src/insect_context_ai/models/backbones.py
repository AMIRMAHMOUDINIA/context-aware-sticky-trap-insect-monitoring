from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small CNN used for software smoke tests and CPU-only debugging."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(96, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


@dataclass
class BackboneBundle:
    module: nn.Module
    output_dim: int


def _torchvision_backbone(name: str, pretrained: bool) -> BackboneBundle:
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError(
            "torchvision could not be imported. Install a torchvision version compatible "
            "with your PyTorch build, or use backbone='tiny_cnn' for smoke tests."
        ) from exc

    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        output_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return BackboneBundle(model, output_dim)
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        output_dim = int(model.classifier[0].in_features)
        model.classifier = nn.Identity()
        return BackboneBundle(model, output_dim)
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        output_dim = int(model.classifier[1].in_features)
        model.classifier = nn.Identity()
        return BackboneBundle(model, output_dim)
    raise ValueError(
        f"Unsupported backbone {name!r}. Choose tiny_cnn, resnet18, "
        "mobilenet_v3_small, or efficientnet_b0."
    )


def build_backbone(name: str, pretrained: bool) -> BackboneBundle:
    name = name.lower()
    if name == "tiny_cnn":
        module = TinyCNN(output_dim=128)
        return BackboneBundle(module, module.output_dim)
    return _torchvision_backbone(name, pretrained)
