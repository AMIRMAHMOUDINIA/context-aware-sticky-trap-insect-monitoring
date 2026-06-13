from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    try:
        loss = float(log_loss(targets, probabilities, labels=labels))
    except ValueError:
        loss = math.nan
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            recall_score(
                targets,
                predictions,
                labels=np.unique(targets),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": loss,
        "expected_calibration_error": expected_calibration_error(targets, probabilities),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }
