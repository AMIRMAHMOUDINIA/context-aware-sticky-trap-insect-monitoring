from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from insect_context_ai.active_learning import run


def test_active_learning_pairs_training_seed_across_strategies(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    records = []
    for index in range(40):
        species = "dark" if index % 2 == 0 else "light"
        pixel = 35 if species == "dark" else 220
        filename = f"image_{index:03d}.png"
        Image.new("RGB", (32, 32), (pixel, pixel, pixel)).save(image_root / filename)
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
    config = {
        "seed": 17,
        "output_dir": str(tmp_path / "active_reports"),
        "data": {
            "manifest": str(manifest),
            "image_root": str(image_root),
            "target_column": "species",
            "context_columns": ["device", "trap_color"],
            "split_strategy": "stratified",
            "train_fraction": 0.6,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "group_column": None,
            "enforce_group_disjoint": True,
            "holdout_column": None,
            "holdout_value": None,
            "image_size": 32,
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
        "active_learning": {
            "strategies": ["random", "entropy"],
            "initial_fraction": 0.25,
            "query_fraction": 0.25,
            "rounds": 1,
            "repeats": 1,
        },
    }
    config_path = tmp_path / "active.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    results = run(config_path)
    initial = results[results["round"] == 0].sort_values("strategy")
    assert len(initial) == 2
    assert initial["labelled_count"].nunique() == 1
    assert initial["macro_f1"].nunique() == 1
    assert (tmp_path / "active_reports" / "learning_curve.csv").is_file()
    assert (tmp_path / "active_reports" / "learning_curve.png").is_file()
