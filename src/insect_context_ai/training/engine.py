from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    targets: np.ndarray
    probabilities: np.ndarray
    source_indices: np.ndarray
    image_paths: list[str]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        context = batch["context"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, context)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
) -> EpochResult:
    model.eval()
    losses: list[float] = []
    counts: list[int] = []
    targets_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    image_paths: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        context = batch["context"].to(device)
        logits = model(images, context)
        if criterion is not None:
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))
            counts.append(int(targets.shape[0]))
        probabilities = torch.softmax(logits, dim=1)
        targets_all.append(targets.detach().cpu().numpy())
        probs_all.append(probabilities.detach().cpu().numpy())
        source_indices.append(batch["source_index"].detach().cpu().numpy())
        image_paths.extend(list(batch["image_path"]))

    if not targets_all:
        raise ValueError("Prediction loader produced no batches.")
    weighted_loss = (
        float(np.average(losses, weights=counts)) if losses and sum(counts) > 0 else float("nan")
    )
    return EpochResult(
        loss=weighted_loss,
        targets=np.concatenate(targets_all),
        probabilities=np.concatenate(probs_all),
        source_indices=np.concatenate(source_indices),
        image_paths=image_paths,
    )


def class_weights(targets: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(
            f"At least one target class is absent from the training data: counts={counts}"
        )
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def entropy(probabilities: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)
