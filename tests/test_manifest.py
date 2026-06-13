from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from insect_context_ai.data.manifest import load_manifest, resolve_image_path


def test_manifest_rejects_path_escape(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    manifest = tmp_path / "metadata.csv"
    pd.DataFrame([{"image_path": "../outside.png", "species": "a"}]).to_csv(
        manifest, index=False
    )

    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest, image_root)


def test_manifest_rejects_absolute_path(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "image.png"
    Image.new("RGB", (8, 8)).save(image)

    with pytest.raises(ValueError, match="relative"):
        resolve_image_path(image_root, image.resolve())
