import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    tokens: List[str]
    aspect: int
    sentiment: int


class SyntheticReviewDataset(Dataset):
    """Synthetic dataset with controllable aspect noise.

    Aspects: 0=teacher, 1=course
    Sentiment: 0=negative, 1=positive
    """

    def __init__(
        self,
        num_examples: int = 400,
        vocab: Tuple[str, ...] | None = None,
        max_len: int = 8,
        aspect_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.vocab = vocab or (
            "teacher",
            "professor",
            "course",
            "syllabus",
            "great",
            "helpful",
            "boring",
            "confusing",
            "clear",
            "organized",
            "tough",
            "fair",
        )
        self.vocab_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.max_len = max_len
        self.aspect_noise = aspect_noise
        self.random = random.Random(seed)
        self.examples = self._build_examples(num_examples)

    def _build_examples(self, num_examples: int) -> List[Example]:
        examples: List[Example] = []
        for _ in range(num_examples):
            aspect = self.random.choice([0, 1])
            sentiment = self.random.choice([0, 1])
            tokens = self._sample_tokens(aspect, sentiment)
            noisy_aspect = aspect
            if self.aspect_noise > 0 and self.random.random() < self.aspect_noise:
                noisy_aspect = 1 - aspect
            examples.append(Example(tokens=tokens, aspect=noisy_aspect, sentiment=sentiment))
        return examples

    def _sample_tokens(self, aspect: int, sentiment: int) -> List[str]:
        base = ["teacher", "professor"] if aspect == 0 else ["course", "syllabus"]
        positive = ["great", "helpful", "clear", "organized", "fair"]
        negative = ["boring", "confusing", "tough"]
        sentiment_tokens = positive if sentiment == 1 else negative
        tokens = self.random.sample(base, k=1) + self.random.sample(sentiment_tokens, k=2)
        filler = self.random.choices(list(self.vocab), k=self.max_len - len(tokens))
        return tokens + filler

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        example = self.examples[idx]
        bow = torch.zeros(len(self.vocab), dtype=torch.float)
        for token in example.tokens:
            bow[self.vocab_to_idx[token]] += 1.0
        return bow, example.aspect, example.sentiment
