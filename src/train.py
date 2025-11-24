from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .model import DualHeadClassifier


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    aspect_weight: float = 1.0
    sentiment_weight: float = 1.0
    hierarchical: bool = False
    device: str = "cpu"


def _device_tensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device)


def train_joint(
    dataloader: DataLoader,
    input_dim: int,
    config: TrainConfig,
) -> DualHeadClassifier:
    model = DualHeadClassifier(input_dim=input_dim, hierarchical=config.hierarchical).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.epochs):
        for bow, aspect, sentiment in dataloader:
            bow = _device_tensor(bow, config.device)
            aspect = _device_tensor(aspect, config.device)
            sentiment = _device_tensor(sentiment, config.device)

            optimizer.zero_grad()
            aspect_logits, sentiment_logits = model(bow)
            aspect_loss = criterion(aspect_logits, aspect)
            sentiment_loss = criterion(sentiment_logits, sentiment)
            loss = config.aspect_weight * aspect_loss + config.sentiment_weight * sentiment_loss
            loss.backward()
            optimizer.step()
    return model


def train_single_task(
    dataloader: DataLoader,
    input_dim: int,
    task: str,
    epochs: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    assert task in {"aspect", "sentiment"}
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for bow, aspect, sentiment in dataloader:
            bow = _device_tensor(bow, device)
            target = _device_tensor(aspect if task == "aspect" else sentiment, device)
            optimizer.zero_grad()
            logits = model(bow)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model: DualHeadClassifier, dataloader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    model.eval()
    aspects: list[int] = []
    aspect_preds: list[int] = []
    sentiments: list[int] = []
    sentiment_preds: list[int] = []
    with torch.no_grad():
        for bow, aspect, sentiment in dataloader:
            bow = _device_tensor(bow, device)
            aspect_logits, sentiment_logits = model(bow)
            aspect_pred = aspect_logits.argmax(dim=-1).cpu()
            sentiment_pred = sentiment_logits.argmax(dim=-1).cpu()
            aspect_preds.extend(aspect_pred.tolist())
            sentiment_preds.extend(sentiment_pred.tolist())
            aspects.extend(aspect.tolist())
            sentiments.extend(sentiment.tolist())
    aspect_f1 = f1_score(aspects, aspect_preds, average="macro")
    sentiment_f1 = f1_score(sentiments, sentiment_preds, average="macro")
    return aspect_f1, sentiment_f1


def evaluate_single(model: nn.Module, dataloader: DataLoader, task: str, device: str = "cpu") -> float:
    assert task in {"aspect", "sentiment"}
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for bow, aspect, sentiment in dataloader:
            bow = _device_tensor(bow, device)
            logits = model(bow)
            preds = logits.argmax(dim=-1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend((aspect if task == "aspect" else sentiment).tolist())
    return f1_score(y_true, y_pred, average="macro")
