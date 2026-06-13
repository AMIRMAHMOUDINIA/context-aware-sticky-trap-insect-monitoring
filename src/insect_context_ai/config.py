from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {key!r} must be a mapping.")
    return value


def _validate_probability(value: object, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; received {number}.")
    return number


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration values that could invalidate an experiment."""
    required = {"seed", "output_dir", "data", "model", "training", "augmentation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    try:
        int(config["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer.") from exc
    if not str(config["output_dir"]).strip():
        raise ValueError("output_dir must not be empty.")

    data = _require_mapping(config, "data")
    for key in ["manifest", "image_root", "target_column"]:
        if not str(data.get(key, "")).strip():
            raise ValueError(f"data.{key} must not be empty.")
    fractions = [
        float(data.get("train_fraction", 0.7)),
        float(data.get("val_fraction", 0.15)),
        float(data.get("test_fraction", 0.15)),
    ]
    if any(value < 0 for value in fractions):
        raise ValueError(f"Data split fractions must be non-negative; received {fractions}.")
    if sum(fractions) <= 0 or fractions[0] <= 0 or fractions[1] <= 0:
        raise ValueError("Training and validation fractions must both be greater than zero.")
    if int(data.get("image_size", 224)) <= 0:
        raise ValueError("data.image_size must be positive.")
    if int(data.get("num_workers", 0)) < 0:
        raise ValueError("data.num_workers must be non-negative.")

    model = _require_mapping(config, "model")
    if int(model.get("embedding_dim", 8)) <= 0:
        raise ValueError("model.embedding_dim must be positive.")
    if int(model.get("hidden_dim", 128)) <= 0:
        raise ValueError("model.hidden_dim must be positive.")
    _validate_probability(model.get("dropout", 0.3), "model.dropout")

    training = _require_mapping(config, "training")
    for key in ["epochs", "batch_size", "patience"]:
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    for key in ["learning_rate", "weight_decay"]:
        if float(training.get(key, 0.0)) < 0:
            raise ValueError(f"training.{key} must be non-negative.")
    if float(training.get("learning_rate", 0.0)) == 0:
        raise ValueError("training.learning_rate must be greater than zero.")

    augmentation = _require_mapping(config, "augmentation")
    for key in ["horizontal_flip", "vertical_flip", "brightness", "contrast"]:
        _validate_probability(augmentation.get(key, 0.0), f"augmentation.{key}")

    active = config.get("active_learning")
    if active is not None:
        if not isinstance(active, dict):
            raise ValueError("active_learning must be a mapping.")
        _validate_probability(active.get("initial_fraction", 0.1), "active_learning.initial_fraction")
        query_fraction = _validate_probability(
            active.get("query_fraction", 0.1), "active_learning.query_fraction"
        )
        if query_fraction == 0:
            raise ValueError("active_learning.query_fraction must be greater than zero.")
        if int(active.get("rounds", 5)) < 0:
            raise ValueError("active_learning.rounds must be non-negative.")
        if int(active.get("repeats", 3)) <= 0:
            raise ValueError("active_learning.repeats must be positive.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a resolved configuration to YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(deepcopy(config), handle, sort_keys=False)
