from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ContextEncoder, ImageTransform, InsectImageDataset, LabelEncoder
from .models import ContextAwareClassifier, build_model
from .training.engine import class_weights, predict_loader, train_one_epoch
from .training.metrics import compute_metrics


@dataclass
class DatasetBundle:
    dataset: InsectImageDataset
    loader: DataLoader


@dataclass
class FitResult:
    model: ContextAwareClassifier
    history: pd.DataFrame
    best_epoch: int
    best_macro_f1: float


def build_transform(config: dict[str, Any], train: bool) -> ImageTransform:
    data_config = config["data"]
    augmentation = config.get("augmentation", {})
    return ImageTransform(
        image_size=int(data_config.get("image_size", 224)),
        train=train,
        horizontal_flip=float(augmentation.get("horizontal_flip", 0.0)) if train else 0.0,
        vertical_flip=float(augmentation.get("vertical_flip", 0.0)) if train else 0.0,
        brightness=float(augmentation.get("brightness", 0.0)) if train else 0.0,
        contrast=float(augmentation.get("contrast", 0.0)) if train else 0.0,
    )


def build_dataset_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    label_encoder: LabelEncoder,
    context_encoder: ContextEncoder,
    train: bool,
    shuffle: bool | None = None,
) -> DatasetBundle:
    data_config = config["data"]
    training_config = config["training"]
    context_columns = list(data_config.get("context_columns", []))
    dataset = InsectImageDataset(
        frame=frame,
        target_column=str(data_config.get("target_column", "species")),
        context_columns=context_columns,
        label_encoder=label_encoder,
        context_encoder=context_encoder,
        transform=build_transform(config, train=train),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=train if shuffle is None else shuffle,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(data_config.get("num_workers", 0)) > 0,
    )
    return DatasetBundle(dataset, loader)


def fit_model(
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    config: dict[str, Any],
    label_encoder: LabelEncoder,
    context_encoder: ContextEncoder,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> FitResult:
    model = build_model(
        config["model"],
        num_classes=len(label_encoder.class_to_idx),
        context_cardinalities=context_encoder.cardinalities(
            list(config["data"].get("context_columns", []))
        ),
    ).to(device)

    training_config = config["training"]
    weighted = bool(training_config.get("class_weighted_loss", True))
    weights = (
        class_weights(
            train_bundle.dataset.targets,
            len(label_encoder.class_to_idx),
            device,
        )
        if weighted
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf
    best_epoch = 0
    patience = int(training_config.get("patience", 5))
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    class_names = [
        label_encoder.idx_to_class[index] for index in range(len(label_encoder.class_to_idx))
    ]

    for epoch in range(1, int(training_config.get("epochs", 20)) + 1):
        train_loss = train_one_epoch(model, train_bundle.loader, optimizer, criterion, device)
        validation = predict_loader(model, val_bundle.loader, criterion, device)
        metrics = compute_metrics(validation.targets, validation.probabilities, class_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation.loss,
            "val_accuracy": metrics["accuracy"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_ece": metrics["expected_calibration_error"],
        }
        history.append(row)
        score = float(metrics["macro_f1"])
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    config,
                    label_encoder,
                    context_encoder,
                    epoch=epoch,
                    validation_macro_f1=score,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    return FitResult(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_macro_f1=float(best_score),
    )


def save_checkpoint(
    path: str | Path,
    model: ContextAwareClassifier,
    config: dict[str, Any],
    label_encoder: LabelEncoder,
    context_encoder: ContextEncoder,
    epoch: int,
    validation_macro_f1: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "class_to_idx": label_encoder.class_to_idx,
            "context_mappings": context_encoder.mappings,
            "epoch": epoch,
            "validation_macro_f1": validation_macro_f1,
        },
        output,
    )


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ContextAwareClassifier, dict[str, Any], LabelEncoder, ContextEncoder]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    label_encoder = LabelEncoder(dict(checkpoint["class_to_idx"]))
    context_encoder = ContextEncoder(dict(checkpoint["context_mappings"]))
    model_config = copy.deepcopy(config["model"])
    model_config["pretrained"] = False
    model = build_model(
        model_config,
        num_classes=len(label_encoder.class_to_idx),
        context_cardinalities=context_encoder.cardinalities(
            list(config["data"].get("context_columns", []))
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config, label_encoder, context_encoder
