import numpy as np

from insect_context_ai.training.metrics import compute_metrics


def test_macro_f1_uses_declared_class_set():
    targets = np.asarray([0, 0])
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2]])
    metrics = compute_metrics(targets, probabilities, ["class_a", "class_b"])
    assert metrics["macro_f1"] == 0.5
