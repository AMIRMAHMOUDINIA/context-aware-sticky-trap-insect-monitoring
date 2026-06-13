from __future__ import annotations

import torch
from torch import nn

from .backbones import build_backbone


class ContextAwareClassifier(nn.Module):
    """Visual classifier with optional categorical context embeddings."""

    def __init__(
        self,
        num_classes: int,
        context_cardinalities: list[int],
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_context: bool = True,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        bundle = build_backbone(backbone, pretrained)
        self.image_encoder = bundle.module
        self.image_feature_dim = bundle.output_dim
        self.use_context = bool(use_context)
        self.context_cardinalities = list(context_cardinalities)

        if self.use_context and self.context_cardinalities:
            self.context_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, embedding_dim, padding_idx=0)
                    for cardinality in context_cardinalities
                ]
            )
            # Index 0 represents missing or unseen context. Keeping that row fixed at zero
            # makes an unseen domain contribute neutral context instead of random noise.
            for embedding in self.context_embeddings:
                with torch.no_grad():
                    embedding.weight[0].zero_()
            context_dim = embedding_dim * len(context_cardinalities)
        else:
            self.context_embeddings = nn.ModuleList()
            context_dim = 0

        input_dim = self.image_feature_dim + context_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.image_encoder(image)
        if image_features.ndim > 2:
            image_features = torch.flatten(image_features, 1)

        features = [image_features]
        if self.use_context and self.context_embeddings:
            if context is None:
                raise ValueError("Context tensor is required when use_context=True.")
            if context.ndim != 2 or context.shape[1] != len(self.context_embeddings):
                raise ValueError(
                    "Context tensor must have shape [batch, number_of_context_columns]."
                )
            embedded = [
                embedding(context[:, idx]) for idx, embedding in enumerate(self.context_embeddings)
            ]
            features.extend(embedded)
        return self.classifier(torch.cat(features, dim=1))


def build_model(
    model_config: dict,
    num_classes: int,
    context_cardinalities: list[int],
) -> ContextAwareClassifier:
    return ContextAwareClassifier(
        num_classes=num_classes,
        context_cardinalities=context_cardinalities,
        backbone=str(model_config.get("backbone", "resnet18")),
        pretrained=bool(model_config.get("pretrained", True)),
        use_context=bool(model_config.get("use_context", True)),
        embedding_dim=int(model_config.get("embedding_dim", 8)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        dropout=float(model_config.get("dropout", 0.3)),
    )
