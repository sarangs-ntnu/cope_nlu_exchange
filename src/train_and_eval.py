from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from augmentation import AugmentationConfig, prepare_datasets


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )


def fit_model(train_df: pd.DataFrame) -> Pipeline:
    model = build_model()
    model.fit(train_df["text"], train_df["label"])
    return model


def evaluate(model: Pipeline, df: pd.DataFrame) -> Dict[str, float]:
    preds = model.predict(df["text"])
    acc = accuracy_score(df["label"], preds)
    report = classification_report(df["label"], preds, output_dict=True)
    return {"accuracy": acc, "report": report}


def save_metrics(clean_metrics: Dict[str, float], noisy_metrics: Dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as fp:
        json.dump({"clean": clean_metrics, "noisy": noisy_metrics}, fp, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate with augmentation")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--val", default="data/val.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--paraphrases", type=int, default=1, help="Paraphrases per training example")
    parser.add_argument("--noise-prob", type=float, default=0.1, help="Noise probability for token corruption")
    parser.add_argument("--use-nlpaug", action="store_true", help="Use nlpaug for paraphrasing and noise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AugmentationConfig(
        paraphrases_per_example=args.paraphrases,
        noise_prob=args.noise_prob,
        use_nlpaug=args.use_nlpaug,
    )

    augmented_train, clean_val, noisy_val = prepare_datasets(args.train, args.val, cfg)

    model = fit_model(augmented_train)
    clean_metrics = evaluate(model, clean_val)
    noisy_metrics = evaluate(model, noisy_val)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "best_model.joblib")
    save_metrics(clean_metrics, noisy_metrics, output_dir)

    print("Clean validation accuracy:", clean_metrics["accuracy"])
    print("Noisy validation accuracy:", noisy_metrics["accuracy"])


if __name__ == "__main__":
    main()
