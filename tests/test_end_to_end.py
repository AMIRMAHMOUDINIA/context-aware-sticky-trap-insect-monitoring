from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from insect_context_ai.train import run


def test_tiny_end_to_end_training(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    records = []
    for index in range(24):
        species = "dark" if index % 2 == 0 else "light"
        pixel = 30 if species == "dark" else 220
        image = Image.new("RGB", (40, 40), (pixel, pixel, pixel))
        filename = f"image_{index:03d}.png"
        image.save(image_root / filename)
        records.append(
            {
                "image_path": filename,
                "species": species,
                "device": "dslr" if index % 3 else "webcam",
                "trap_color": "yellow" if index % 4 else "blue",
                "split": "",
                "group_id": "",
            }
        )
    manifest = tmp_path / "metadata.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    output_dir = tmp_path / "reports"
    config = {
        "seed": 5,
        "output_dir": str(output_dir),
        "data": {
            "manifest": str(manifest),
            "image_root": str(image_root),
            "target_column": "species",
            "context_columns": ["device", "trap_color"],
            "split_strategy": "stratified",
            "train_fraction": 0.5,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "group_column": None,
            "holdout_column": None,
            "holdout_value": None,
            "image_size": 40,
            "num_workers": 0,
        },
        "model": {
            "backbone": "tiny_cnn",
            "pretrained": False,
            "use_context": True,
            "embedding_dim": 2,
            "hidden_dim": 8,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
            "class_weighted_loss": True,
            "device": "cpu",
        },
        "augmentation": {
            "horizontal_flip": 0.0,
            "vertical_flip": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics = run(config_path)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["manifest_sha256"]) == 64
    for filename in [
        "best.pt",
        "metrics.json",
        "predictions.csv",
        "history.csv",
        "confusion_matrix.png",
        "reliability_diagram.png",
        "run_summary.md",
        "split_assignments.csv",
        "subgroup_metrics.csv",
        "training_history.png",
    ]:
        assert (output_dir / filename).is_file()
