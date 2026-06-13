from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from .data.transforms import ImageTransform, denormalize_image
from .experiment import load_checkpoint_model
from .utils.reproducibility import resolve_device


def _last_convolution(module: nn.Module) -> nn.Conv2d:
    convolutions = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    if not convolutions:
        raise ValueError("No convolutional layer was found for Grad-CAM.")
    return convolutions[-1]


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    context_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = _last_convolution(model.image_encoder)

    def hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor, context_tensor)
        probabilities = torch.softmax(logits, dim=1)
        chosen = int(probabilities.argmax(dim=1).item()) if target_class is None else int(target_class)
        if chosen < 0 or chosen >= logits.shape[1]:
            raise ValueError(
                f"target_class must be between 0 and {logits.shape[1] - 1}; received {chosen}."
            )
        logits[0, chosen].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        activation = activations[-1][0]
        gradient = gradients[-1][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        maximum = cam.max()
        if maximum.detach().item() > 0:
            cam = cam / maximum
        return cam.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[0]
    finally:
        handle.remove()


def run(
    checkpoint: str | Path,
    image_path: str | Path,
    output: str | Path,
    device_label: str = "",
    trap_color: str = "",
    target_class: int | None = None,
    requested_device: str = "auto",
) -> dict:
    device = resolve_device(requested_device)
    model, config, label_encoder, context_encoder = load_checkpoint_model(checkpoint, device)
    image_size = int(config["data"].get("image_size", 224))
    transform = ImageTransform(image_size=image_size, train=False)
    with Image.open(image_path) as image:
        tensor = transform(image.copy()).unsqueeze(0).to(device)

    context_columns = list(config["data"].get("context_columns", []))
    values = {"device": device_label, "trap_color": trap_color}
    context_row = pd.Series({column: values.get(column, "") for column in context_columns})
    context_values = context_encoder.transform_row(context_row, context_columns)
    context_tensor = torch.tensor([context_values], dtype=torch.long, device=device)

    cam, probabilities = generate_gradcam(model, tensor, context_tensor, target_class)
    image = denormalize_image(tensor[0])
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image)
    axis.imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    predicted_index = int(np.argmax(probabilities))
    predicted_label = label_encoder.idx_to_class[predicted_index]
    axis.set_title(f"Grad-CAM: {predicted_label} ({probabilities[predicted_index]:.3f})")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities.tolist(),
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM overlay.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", required=False, help="Deprecated; config is stored in checkpoint."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device-label", default="")
    parser.add_argument("--trap-color", default="")
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        checkpoint=args.checkpoint,
        image_path=args.image,
        output=args.output,
        device_label=args.device_label,
        trap_color=args.trap_color,
        target_class=args.target_class,
        requested_device=args.device,
    )
    print(
        f"Saved {result['output']}; predicted {result['predicted_label']} "
        f"with confidence {result['confidence']:.3f}."
    )


if __name__ == "__main__":
    main()
