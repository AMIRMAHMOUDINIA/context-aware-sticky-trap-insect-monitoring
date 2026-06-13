#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

BACKGROUND = {
    "blue": (55, 110, 190),
    "yellow": (235, 210, 55),
    "white": (235, 235, 235),
    "transparent": (190, 190, 190),
}


def _draw_insect(
    species: str, trap_color: str, device: str, rng: np.random.Generator
) -> Image.Image:
    size = 96
    base = Image.new("RGB", (size, size), BACKGROUND[trap_color])
    draw = ImageDraw.Draw(base)
    centre_x = int(rng.integers(35, 61))
    centre_y = int(rng.integers(35, 61))
    if species == "tribolium_castaneum":
        body = (25, 10, 5)
        draw.ellipse((centre_x - 10, centre_y - 17, centre_x + 10, centre_y + 17), fill=body)
        draw.line((centre_x, centre_y - 15, centre_x, centre_y + 15), fill=(90, 40, 20), width=2)
    else:
        body = (15, 15, 10)
        draw.ellipse((centre_x - 9, centre_y - 14, centre_x + 9, centre_y + 14), fill=body)
        draw.polygon(
            [
                (centre_x - 3, centre_y - 14),
                (centre_x + 3, centre_y - 14),
                (centre_x + 1, centre_y - 26),
            ],
            fill=body,
        )

    for offset in (-8, 0, 8):
        draw.line(
            (centre_x - 8, centre_y + offset, centre_x - 18, centre_y + offset + 5),
            fill=body,
            width=2,
        )
        draw.line(
            (centre_x + 8, centre_y + offset, centre_x + 18, centre_y + offset + 5),
            fill=body,
            width=2,
        )

    if device == "webcam":
        base = base.resize((64, 64), Image.Resampling.BILINEAR).resize(
            (size, size), Image.Resampling.BILINEAR
        )
        base = base.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif device == "smartphone":
        base = base.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    noise = rng.normal(0, 4 if device == "dslr" else 7, (size, size, 3))
    array = np.clip(np.asarray(base, dtype=float) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def generate(output_dir: Path, images_per_combination: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    species_values = ["tribolium_castaneum", "sitophilus_oryzae"]
    devices = ["dslr", "webcam", "smartphone"]
    colours = ["blue", "yellow", "white", "transparent"]
    counter = 0
    for species in species_values:
        for device in devices:
            for colour in colours:
                for replicate in range(images_per_combination):
                    image = _draw_insect(species, colour, device, rng)
                    filename = f"{counter:04d}_{species}_{device}_{colour}.png"
                    image.save(image_dir / filename)
                    records.append(
                        {
                            "image_path": filename,
                            "species": species,
                            "device": device,
                            "trap_color": colour,
                            "split": "",
                            "group_id": f"{species}_{device}_{colour}_{replicate}",
                        }
                    )
                    counter += 1
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "metadata.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic images for software smoke tests."
    )
    parser.add_argument("--output-dir", default="data/smoke")
    parser.add_argument("--images-per-combination", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = generate(Path(args.output_dir), args.images_per_combination, args.seed)
    print(f"Generated {len(frame)} synthetic smoke-test images in {args.output_dir}")


if __name__ == "__main__":
    main()
