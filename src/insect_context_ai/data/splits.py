from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class DataSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _normalise_fractions(train: float, val: float, test: float) -> tuple[float, float, float]:
    fractions = {"train": train, "validation": val, "test": test}
    negative = {name: value for name, value in fractions.items() if value < 0}
    if negative:
        raise ValueError(f"Split fractions must be non-negative; received {negative}.")
    total = train + val + test
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    normalised = (train / total, val / total, test / total)
    if normalised[0] <= 0:
        raise ValueError("The training fraction must be greater than zero.")
    if normalised[1] <= 0:
        raise ValueError("The validation fraction must be greater than zero.")
    return normalised


def _stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DataSplits:
    train_fraction, val_fraction, test_fraction = _normalise_fractions(
        train_fraction, val_fraction, test_fraction
    )
    if test_fraction == 0:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=labels,
        )
        return DataSplits(np.asarray(train_idx), np.asarray(val_idx), np.asarray([], dtype=int))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    label_lookup = {int(index): label for index, label in zip(indices, labels, strict=True)}
    remaining_labels = np.asarray([label_lookup[int(index)] for index in train_val_idx])
    relative_val = val_fraction / (train_fraction + val_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed + 1,
        stratify=remaining_labels,
    )
    return DataSplits(np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx))


def _group_split(
    frame: pd.DataFrame,
    group_column: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DataSplits:
    train_fraction, val_fraction, test_fraction = _normalise_fractions(
        train_fraction, val_fraction, test_fraction
    )
    indices = frame.index.to_numpy()
    groups = frame[group_column].astype(str).to_numpy()

    if test_fraction == 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_pos, val_pos = next(splitter.split(indices, groups=groups))
        return DataSplits(indices[train_pos], indices[val_pos], np.asarray([], dtype=int))

    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_pos, test_pos = next(first.split(indices, groups=groups))
    train_val_indices = indices[train_val_pos]
    train_val_groups = groups[train_val_pos]
    relative_val = val_fraction / (train_fraction + val_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_pos, val_pos = next(second.split(train_val_indices, groups=train_val_groups))
    return DataSplits(
        train_val_indices[train_pos],
        train_val_indices[val_pos],
        indices[test_pos],
    )


def create_splits(frame: pd.DataFrame, config: dict, seed: int) -> DataSplits:
    strategy = str(config.get("split_strategy", "stratified")).lower()
    target = str(config.get("target_column", "species"))
    group_column = config.get("group_column")
    train_fraction = float(config.get("train_fraction", 0.7))
    val_fraction = float(config.get("val_fraction", 0.15))
    test_fraction = float(config.get("test_fraction", 0.15))

    if target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")

    if strategy == "leave_one_domain_out":
        holdout_column = config.get("holdout_column")
        holdout_value = str(config.get("holdout_value", "")).strip().lower()
        if not holdout_column or holdout_column not in frame.columns or not holdout_value:
            raise ValueError("leave_one_domain_out requires holdout_column and holdout_value.")
        values = frame[holdout_column].astype(str).str.strip().str.lower()
        test_idx = frame.index[values == holdout_value].to_numpy()
        remainder = frame.loc[values != holdout_value]
        if len(test_idx) == 0:
            available = sorted(values.unique().tolist())
            raise ValueError(
                f"Holdout value {holdout_value!r} not found in {holdout_column}. "
                f"Available values: {available}"
            )
        if (
            group_column
            and group_column in remainder.columns
            and remainder[group_column].ne("").any()
        ):
            inner = _group_split(
                remainder,
                group_column,
                train_fraction,
                val_fraction,
                0.0,
                seed,
            )
        else:
            indices = remainder.index.to_numpy()
            labels = remainder[target].astype(str).to_numpy()
            inner = _stratified_split(
                indices,
                labels,
                train_fraction,
                val_fraction,
                0.0,
                seed,
            )
        return DataSplits(inner.train, inner.val, test_idx)

    if strategy == "predefined_or_grouped" and "split" in frame.columns:
        values = frame["split"].astype(str).str.lower().str.strip()
        recognised = values.isin(["train", "val", "validation", "test"])
        if recognised.all() and {"train", "test"}.issubset(set(values)):
            train_idx = frame.index[values == "train"].to_numpy()
            val_idx = frame.index[values.isin(["val", "validation"])].to_numpy()
            test_idx = frame.index[values == "test"].to_numpy()
            if len(val_idx) == 0:
                train_frame = frame.loc[train_idx]
                labels = train_frame[target].astype(str).to_numpy()
                inner = _stratified_split(
                    train_frame.index.to_numpy(), labels, 0.85, 0.15, 0.0, seed
                )
                train_idx, val_idx = inner.train, inner.val
            return DataSplits(train_idx, val_idx, test_idx)

    if strategy in {"grouped", "predefined_or_grouped"}:
        if group_column and group_column in frame.columns and frame[group_column].ne("").any():
            return _group_split(
                frame,
                group_column,
                train_fraction,
                val_fraction,
                test_fraction,
                seed,
            )

    indices = frame.index.to_numpy()
    labels = frame[target].astype(str).to_numpy()
    return _stratified_split(
        indices,
        labels,
        train_fraction,
        val_fraction,
        test_fraction,
        seed,
    )


def assert_disjoint(splits: DataSplits) -> None:
    train, val, test = map(set, [splits.train.tolist(), splits.val.tolist(), splits.test.tolist()])
    if train & val or train & test or val & test:
        raise AssertionError("Train, validation, and test indices are not disjoint.")


def assert_group_disjoint(
    frame: pd.DataFrame,
    splits: DataSplits,
    group_column: str | None,
) -> None:
    """Reject non-empty group identifiers shared across data splits."""
    if not group_column or group_column not in frame.columns:
        return

    def groups(indices: np.ndarray) -> set[str]:
        values = frame.loc[indices, group_column].fillna("").astype(str).str.strip()
        return {value for value in values if value}

    train_groups = groups(splits.train)
    val_groups = groups(splits.val)
    test_groups = groups(splits.test)
    overlaps = {
        "train/validation": train_groups & val_groups,
        "train/test": train_groups & test_groups,
        "validation/test": val_groups & test_groups,
    }
    present = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
    if present:
        raise AssertionError(
            "Non-empty group_id values cross data splits. This can leak repeated specimens or "
            f"traps across evaluation boundaries. Example overlaps: {present}"
        )


def assert_class_coverage(
    frame: pd.DataFrame,
    splits: DataSplits,
    target_column: str,
    require_test: bool = True,
) -> None:
    """Ensure evaluation splits contain every class learned from the training split."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    train_classes = set(frame.loc[splits.train, target_column].astype(str))
    if len(train_classes) < 2:
        raise ValueError(
            "The training split must contain at least two target classes; "
            f"found {sorted(train_classes)}."
        )

    checks = [("validation", splits.val)]
    if require_test and len(splits.test):
        checks.append(("test", splits.test))
    for name, indices in checks:
        observed = set(frame.loc[indices, target_column].astype(str))
        missing = sorted(train_classes.difference(observed))
        if missing:
            raise ValueError(
                f"The {name} split is missing training classes {missing}. "
                "Revise the split, grouping, or holdout design before comparing models."
            )
