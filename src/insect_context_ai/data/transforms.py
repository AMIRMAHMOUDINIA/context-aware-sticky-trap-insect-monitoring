from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


@dataclass
class ImageTransform:
    """Small torchvision-independent image transformation pipeline."""

    image_size: int
    train: bool = False
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
        )
        if self.train:
            if random.random() < self.horizontal_flip:
                image = ImageOps.mirror(image)
            if random.random() < self.vertical_flip:
                image = ImageOps.flip(image)
            if self.brightness > 0:
                factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                image = ImageEnhance.Brightness(image).enhance(factor)
            if self.contrast > 0:
                factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                image = ImageEnhance.Contrast(image).enhance(factor)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized CHW tensor to an RGB float array."""
    image = tensor.detach().cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image
