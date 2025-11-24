from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ModelConfig
from .utils import ensure_dir, set_seed


class TextClassifier:
    """Wrapper around a sklearn pipeline for aspect classification."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        set_seed(self.config.random_seed)
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
        )
        classifier = LogisticRegression(
            C=self.config.c,
            penalty=self.config.penalty,
            solver="liblinear",
            max_iter=1000,
            random_state=self.config.random_seed,
        )
        return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    def fit(self, df: pd.DataFrame, text_column: str, label_column: str) -> None:
        self.pipeline.fit(df[text_column], df[label_column])

    def predict(self, df: pd.DataFrame, text_column: str) -> List[str]:
        return self.pipeline.predict(df[text_column]).tolist()

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "TextClassifier":
        pipeline = joblib.load(path)
        dummy_config = ModelConfig()
        obj = cls(dummy_config)
        obj.pipeline = pipeline
        return obj
