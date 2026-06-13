from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn

from .config import load_config, save_config
from .data import (
    ContextEncoder,
    LabelEncoder,
    assert_class_coverage,
    assert_disjoint,
    assert_group_disjoint,
    create_splits,
    load_manifest,
)
from .experiment import build_dataset_loader, fit_model
from .training.engine import entropy, predict_loader
from .training.metrics import compute_metrics
from .training.plots import plot_learning_curve
from .utils.reproducibility import resolve_device, set_global_seed


def _initial_stratified_indices(
    frame: pd.DataFrame,
    target_column: str,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    classes = sorted(frame[target_column].astype(str).unique().tolist())
    if count < len(classes):
        raise ValueError(
            f"Initial labelled count ({count}) must be at least the number of classes ({len(classes)})."
        )
    selected: list[int] = []
    for label in classes:
        candidates = frame.index[frame[target_column].astype(str) == label].to_numpy()
        if len(candidates) == 0:
            raise ValueError(f"No candidate image found for class {label!r}.")
        selected.append(int(rng.choice(candidates)))
    remaining = np.asarray([index for index in frame.index if index not in set(selected)])
    additional = min(count - len(selected), len(remaining))
    if additional > 0:
        selected.extend(rng.choice(remaining, size=additional, replace=False).astype(int).tolist())
    return selected


def run(config_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    if "active_learning" not in config:
        raise ValueError("Configuration is missing the active_learning section.")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config_resolved.yaml")

    base_seed = int(config["seed"])
    set_global_seed(base_seed)
    frame = load_manifest(config["data"]["manifest"], config["data"]["image_root"])
    splits = create_splits(frame, config["data"], base_seed)
    assert_disjoint(splits)
    if bool(config["data"].get("enforce_group_disjoint", True)):
        assert_group_disjoint(frame, splits, config["data"].get("group_column"))
    assert_class_coverage(
        frame,
        splits,
        str(config["data"].get("target_column", "species")),
        require_test=True,
    )
    if len(splits.test) == 0:
        raise ValueError("Active-learning evaluation requires a non-empty test split.")

    target_column = str(config["data"].get("target_column", "species"))
    context_columns = list(config["data"].get("context_columns", []))
    pool_frame = frame.loc[splits.train]
    val_frame = frame.loc[splits.val]
    test_frame = frame.loc[splits.test]

    label_encoder = LabelEncoder.fit(pool_frame[target_column])
    context_encoder = ContextEncoder.fit(pool_frame, context_columns)
    val_bundle = build_dataset_loader(
        val_frame, config, label_encoder, context_encoder, train=False
    )
    test_bundle = build_dataset_loader(
        test_frame, config, label_encoder, context_encoder, train=False
    )
    device = resolve_device(str(config["training"].get("device", "auto")))
    criterion = nn.CrossEntropyLoss()
    class_names = [
        label_encoder.idx_to_class[index] for index in range(len(label_encoder.class_to_idx))
    ]

    active = config["active_learning"]
    initial_fraction = float(active.get("initial_fraction", 0.1))
    query_fraction = float(active.get("query_fraction", 0.1))
    rounds = int(active.get("rounds", 5))
    repeats = int(active.get("repeats", 3))
    strategies = [str(value).lower() for value in active.get("strategies", ["random", "entropy"])]
    unsupported = set(strategies).difference({"random", "entropy"})
    if unsupported:
        raise ValueError(f"Unsupported active-learning strategies: {sorted(unsupported)}")

    initial_count = max(len(class_names), int(round(initial_fraction * len(pool_frame))))
    query_count = max(1, int(round(query_fraction * len(pool_frame))))
    records: list[dict] = []

    for repeat in range(repeats):
        repeat_seed = base_seed + repeat * 1000
        for strategy in strategies:
            rng = np.random.default_rng(repeat_seed)
            labelled = _initial_stratified_indices(pool_frame, target_column, initial_count, rng)
            for round_index in range(rounds + 1):
                # Use the same training seed for each strategy at a given repeat/round.
                # This keeps model initialization and augmentation noise paired, so the
                # comparison reflects the queried images rather than a seed confound.
                iteration_seed = repeat_seed + round_index
                set_global_seed(iteration_seed)
                labelled_frame = frame.loc[sorted(labelled)]
                train_bundle = build_dataset_loader(
                    labelled_frame,
                    config,
                    label_encoder,
                    context_encoder,
                    train=True,
                )
                fit_result = fit_model(
                    train_bundle,
                    val_bundle,
                    config,
                    label_encoder,
                    context_encoder,
                    device,
                    checkpoint_path=None,
                )
                evaluation = predict_loader(fit_result.model, test_bundle.loader, criterion, device)
                metrics = compute_metrics(evaluation.targets, evaluation.probabilities, class_names)
                records.append(
                    {
                        "repeat": repeat,
                        "round": round_index,
                        "strategy": strategy,
                        "labelled_count": len(labelled),
                        "labelled_fraction": len(labelled) / len(pool_frame),
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "accuracy": metrics["accuracy"],
                        "ece": metrics["expected_calibration_error"],
                        "best_epoch": fit_result.best_epoch,
                    }
                )

                if round_index == rounds or len(labelled) == len(pool_frame):
                    continue
                labelled_set = set(labelled)
                candidates = [index for index in pool_frame.index if index not in labelled_set]
                if not candidates:
                    continue
                amount = min(query_count, len(candidates))
                if strategy == "random":
                    chosen = rng.choice(candidates, size=amount, replace=False).astype(int).tolist()
                else:
                    candidate_bundle = build_dataset_loader(
                        frame.loc[candidates],
                        config,
                        label_encoder,
                        context_encoder,
                        train=False,
                        shuffle=False,
                    )
                    pool_predictions = predict_loader(
                        fit_result.model, candidate_bundle.loader, None, device
                    )
                    scores = entropy(pool_predictions.probabilities)
                    order = np.argsort(-scores)[:amount]
                    chosen = pool_predictions.source_indices[order].astype(int).tolist()
                labelled.extend(chosen)
                labelled = sorted(set(labelled))

    results = pd.DataFrame(records)
    results.to_csv(output_dir / "learning_curve.csv", index=False)
    plot_learning_curve(results, output_dir / "learning_curve.png")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate random and uncertainty sampling.")
    parser.add_argument("--config", required=True, help="Path to active-learning YAML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run(args.config)
    print(f"Completed {len(results)} active-learning evaluations.")


if __name__ == "__main__":
    main()
