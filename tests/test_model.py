import torch

from insect_context_ai.models.classifier import ContextAwareClassifier
from insect_context_ai.training.engine import entropy


def test_context_model_forward_shape():
    model = ContextAwareClassifier(
        num_classes=2,
        context_cardinalities=[4, 5],
        backbone="tiny_cnn",
        pretrained=False,
        use_context=True,
        embedding_dim=4,
        hidden_dim=16,
        dropout=0.0,
    )
    images = torch.randn(3, 3, 64, 64)
    context = torch.tensor([[1, 1], [2, 3], [0, 4]], dtype=torch.long)
    logits = model(images, context)
    assert logits.shape == (3, 2)


def test_entropy_ranks_uniform_prediction_highest():
    probabilities = torch.tensor([[0.99, 0.01], [0.5, 0.5]], dtype=torch.float32).numpy()
    scores = entropy(probabilities)
    assert scores[1] > scores[0]


def test_unknown_context_embedding_is_neutral_zero_vector():
    model = ContextAwareClassifier(
        num_classes=2,
        context_cardinalities=[4, 5],
        backbone="tiny_cnn",
        pretrained=False,
        use_context=True,
        embedding_dim=4,
        hidden_dim=16,
        dropout=0.0,
    )
    for embedding in model.context_embeddings:
        assert torch.equal(embedding.weight[0], torch.zeros_like(embedding.weight[0]))
        assert embedding.padding_idx == 0
