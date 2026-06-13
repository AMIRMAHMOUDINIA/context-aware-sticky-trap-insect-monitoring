import pandas as pd

from insect_context_ai.data.splits import assert_disjoint, create_splits


def test_stratified_splits_are_disjoint_and_complete():
    frame = pd.DataFrame(
        {
            "species": ["a", "b"] * 30,
            "device": ["dslr"] * 60,
            "trap_color": ["yellow"] * 60,
        }
    )
    config = {
        "split_strategy": "stratified",
        "target_column": "species",
        "train_fraction": 0.6,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "group_column": None,
    }
    splits = create_splits(frame, config, seed=12)
    assert_disjoint(splits)
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(frame)
    assert set(frame.loc[splits.test, "species"]) == {"a", "b"}


def test_leave_one_domain_out_uses_requested_test_domain():
    frame = pd.DataFrame(
        {
            "species": ["a", "b"] * 20,
            "device": ["dslr"] * 20 + ["phone"] * 20,
            "trap_color": ["yellow"] * 40,
        }
    )
    config = {
        "split_strategy": "leave_one_domain_out",
        "target_column": "species",
        "train_fraction": 0.8,
        "val_fraction": 0.2,
        "test_fraction": 0,
        "group_column": None,
        "holdout_column": "device",
        "holdout_value": "phone",
    }
    splits = create_splits(frame, config, seed=12)
    assert set(frame.loc[splits.test, "device"]) == {"phone"}
    assert set(frame.loc[splits.train, "device"]) == {"dslr"}


def test_group_overlap_is_rejected():
    import numpy as np
    import pytest

    from insect_context_ai.data.splits import DataSplits, assert_group_disjoint

    frame = pd.DataFrame(
        {
            "species": ["a", "a", "b", "b"],
            "group_id": ["same", "same", "other", "third"],
        }
    )
    splits = DataSplits(
        train=np.asarray([0, 2]),
        val=np.asarray([1]),
        test=np.asarray([3]),
    )
    with pytest.raises(AssertionError, match="group_id"):
        assert_group_disjoint(frame, splits, "group_id")


def test_class_coverage_rejects_missing_validation_class():
    import numpy as np
    import pytest

    from insect_context_ai.data.splits import DataSplits, assert_class_coverage

    frame = pd.DataFrame({"species": ["a", "b", "a", "a", "a", "b"]})
    splits = DataSplits(
        train=np.asarray([0, 1]),
        val=np.asarray([2, 3]),
        test=np.asarray([4, 5]),
    )
    with pytest.raises(ValueError, match="validation split is missing"):
        assert_class_coverage(frame, splits, "species")


def test_negative_split_fraction_is_rejected():
    import pytest

    frame = pd.DataFrame({"species": ["a", "b"] * 10})
    config = {
        "split_strategy": "stratified",
        "target_column": "species",
        "train_fraction": 0.8,
        "val_fraction": 0.3,
        "test_fraction": -0.1,
    }
    with pytest.raises(ValueError, match="non-negative"):
        create_splits(frame, config, seed=1)
