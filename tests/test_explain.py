import numpy as np
import torch

from insect_context_ai.explain import generate_gradcam
from insect_context_ai.models.classifier import ContextAwareClassifier


def test_gradcam_returns_normalized_map_and_probabilities():
    model = ContextAwareClassifier(
        num_classes=2,
        context_cardinalities=[4, 5],
        backbone="tiny_cnn",
        pretrained=False,
        use_context=True,
        embedding_dim=2,
        hidden_dim=8,
        dropout=0.0,
    )
    model.eval()
    image = torch.randn(1, 3, 48, 48)
    context = torch.tensor([[1, 2]], dtype=torch.long)
    cam, probabilities = generate_gradcam(model, image, context)
    assert cam.shape == (48, 48)
    assert np.isfinite(cam).all()
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0
    assert probabilities.shape == (2,)
    assert np.isclose(probabilities.sum(), 1.0)
