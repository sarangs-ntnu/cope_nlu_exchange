from __future__ import annotations

import torch
import torch.nn as nn


class DualHeadClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        aspect_classes: int = 2,
        sentiment_classes: int = 2,
        hierarchical: bool = False,
    ) -> None:
        super().__init__()
        self.hierarchical = hierarchical
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.aspect_head = nn.Linear(hidden_dim, aspect_classes)
        sentiment_in = hidden_dim + aspect_classes if hierarchical else hidden_dim
        self.sentiment_head = nn.Linear(sentiment_in, sentiment_classes)

    def forward(self, bow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(bow)
        aspect_logits = self.aspect_head(encoded)
        if self.hierarchical:
            aspect_probs = torch.softmax(aspect_logits.detach(), dim=-1)
            sentiment_input = torch.cat([encoded, aspect_probs], dim=-1)
        else:
            sentiment_input = encoded
        sentiment_logits = self.sentiment_head(sentiment_input)
        return aspect_logits, sentiment_logits
