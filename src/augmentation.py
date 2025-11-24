from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

try:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
except Exception:  # pragma: no cover - optional dependency
    nac = None
    naw = None


def _find_aspect_tokens(text: str) -> List[str]:
    """Return bracketed aspect tokens such as ``[SERVICE]``.

    The tokens are used to ensure augmentation does not distort them.
    """

    return re.findall(r"\[[^\]]+\]", text)


def _protect_aspect_tokens(text: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to parts of text while preserving exact aspect tokens."""

    tokens = _find_aspect_tokens(text)
    if not tokens:
        return fn(text)

    escaped_tokens = {token: f"<<{i}>>" for i, token in enumerate(tokens)}
    replaced = text
    for token, placeholder in escaped_tokens.items():
        replaced = replaced.replace(token, placeholder)

    transformed = fn(replaced)

    for token, placeholder in escaped_tokens.items():
        transformed = transformed.replace(placeholder, token)
    return transformed


def simple_paraphrase(text: str, synonym_map: Optional[Dict[str, str]] = None) -> str:
    """Lightweight paraphrasing that swaps in synonyms.

    This fallback avoids network calls and keeps aspect tokens untouched.
    """

    synonym_map = synonym_map or {
        "kind": "friendly",
        "attentive": "thoughtful",
        "bland": "tasteless",
        "overpriced": "pricey",
        "cozy": "snug",
        "inviting": "welcoming",
        "flavorful": "tasty",
        "forgot": "missed",
        "loud": "noisy",
    }

    def _replace(text: str) -> str:
        tokens = text.split()
        swapped = [synonym_map.get(token.strip(",."), token) for token in tokens]
        return " ".join(swapped)

    return _protect_aspect_tokens(text, _replace)


def inject_noise(text: str, typo_prob: float = 0.08, punctuation_prob: float = 0.05) -> str:
    """Add light character noise while avoiding aspect tokens."""

    def _noisify(text: str) -> str:
        result: List[str] = []
        for token in text.split(" "):
            if token.startswith("<<") and token.endswith(">>"):
                result.append(token)
                continue

            noisy_token = []
            for ch in token:
                noisy_token.append(ch)
                if random.random() < typo_prob:
                    noisy_token.append(random.choice("abcdefghijklmnopqrstuvwxyz"))
            if random.random() < punctuation_prob:
                noisy_token.append(random.choice(["!", "?", ",", "."]))
            result.append("".join(noisy_token))
        return " ".join(result)

    return _protect_aspect_tokens(text, _noisify)


@dataclass
class AugmentationConfig:
    paraphrases_per_example: int = 1
    noise_prob: float = 0.1
    use_nlpaug: bool = False


class DataAugmenter:
    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg
        self.word_aug = None
        self.char_aug = None
        if cfg.use_nlpaug and naw and nac:
            self.word_aug = naw.SynonymAug(aug_src="wordnet")
            self.char_aug = nac.RandomCharAug(action="insert")

    def paraphrase(self, text: str) -> str:
        if self.word_aug:
            return _protect_aspect_tokens(text, self.word_aug.augment)
        return simple_paraphrase(text)

    def noisify(self, text: str) -> str:
        if self.char_aug:
            return _protect_aspect_tokens(text, self.char_aug.augment)
        return inject_noise(text, typo_prob=self.cfg.noise_prob / 2, punctuation_prob=self.cfg.noise_prob / 3)

    def augment_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        augmented_rows: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            base_text = row["text"]
            label = row["label"]
            for _ in range(self.cfg.paraphrases_per_example):
                paraphrased = self.paraphrase(base_text)
                augmented_rows.append({"text": paraphrased, "label": label})
                augmented_rows.append({"text": self.noisify(paraphrased), "label": label})
        augmented_df = pd.DataFrame(augmented_rows)
        return pd.concat([df, augmented_df], ignore_index=True)

    def noisify_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        noisy_rows = []
        for _, row in df.iterrows():
            noisy_rows.append({"text": self.noisify(row["text"]), "label": row["label"]})
        return pd.DataFrame(noisy_rows)


def load_dataset(train_path: str, val_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def prepare_datasets(train_path: str, val_path: str, cfg: AugmentationConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df = load_dataset(train_path, val_path)
    augmenter = DataAugmenter(cfg)
    augmented_train = augmenter.augment_frame(train_df)
    noisy_val = augmenter.noisify_frame(val_df)
    return augmented_train, val_df, noisy_val
