import pytest

from insect_context_ai.config import validate_config


def base_config():
    return {
        "seed": 1,
        "output_dir": "reports/test",
        "data": {
            "manifest": "data/metadata.csv",
            "image_root": "data/images",
            "target_column": "species",
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "image_size": 224,
            "num_workers": 0,
        },
        "model": {"embedding_dim": 8, "hidden_dim": 64, "dropout": 0.2},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "patience": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "augmentation": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.0,
            "brightness": 0.1,
            "contrast": 0.1,
        },
    }


def test_config_rejects_negative_split_fraction():
    config = base_config()
    config["data"]["test_fraction"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(config)


def test_config_rejects_invalid_augmentation_probability():
    config = base_config()
    config["augmentation"]["horizontal_flip"] = 1.2
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_config(config)


def test_config_rejects_zero_active_learning_query_fraction():
    config = base_config()
    config["active_learning"] = {
        "initial_fraction": 0.1,
        "query_fraction": 0.0,
        "rounds": 2,
        "repeats": 1,
    }
    with pytest.raises(ValueError, match="greater than zero"):
        validate_config(config)
