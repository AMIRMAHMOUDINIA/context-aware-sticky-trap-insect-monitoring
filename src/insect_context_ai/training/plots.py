from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_reliability_diagram(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    accuracies: list[float] = []
    mean_confidences: list[float] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= lower) & (
            confidences <= upper if upper == 1 else confidences < upper
        )
        if mask.any():
            accuracies.append(float(correctness[mask].mean()))
            mean_confidences.append(float(confidences[mask].mean()))

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.plot(mean_confidences, accuracies, marker="o", label="Model")
    axis.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_learning_curve(frame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        frame.groupby(["strategy", "labelled_fraction"], as_index=False)["macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for strategy, group in summary.groupby("strategy"):
        axis.errorbar(
            group["labelled_fraction"],
            group["mean"],
            yerr=group["std"].fillna(0),
            marker="o",
            capsize=3,
            label=strategy,
        )
    axis.set(
        xlabel="Fraction of training pool labelled",
        ylabel="Test macro-F1",
        ylim=(0, 1),
        title="Active-learning simulation",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(frame, output_path: str | Path) -> None:
    """Plot training loss and validation macro-F1 from an epoch history table."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, loss_axis = plt.subplots(figsize=(7, 5))
    score_axis = loss_axis.twinx()
    loss_line = loss_axis.plot(
        frame["epoch"], frame["train_loss"], marker="o", label="Training loss"
    )
    score_line = score_axis.plot(
        frame["epoch"], frame["val_macro_f1"], marker="s", label="Validation macro-F1"
    )
    loss_axis.set(xlabel="Epoch", ylabel="Training loss")
    score_axis.set(ylabel="Validation macro-F1", ylim=(0, 1))
    loss_axis.set_title("Training history")
    lines = loss_line + score_line
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
