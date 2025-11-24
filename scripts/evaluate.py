"""Evaluate a trained model on held-out data."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlu.metrics import summarize
from nlu.model import TextClassifier
from nlu.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.csv"), help="CSV file containing evaluation data")
    parser.add_argument("--model-path", type=Path, default=Path("models/baseline.joblib"), help="Path to trained model")
    parser.add_argument("--text-column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label-column", type=str, default="aspect", help="Name of the label column")
    parser.add_argument("--output", type=Path, default=Path("runs/eval_metrics.json"), help="Where to store evaluation metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    classifier = TextClassifier.load(args.model_path)

    labels = sorted(df[args.label_column].unique())
    predictions = classifier.predict(df, text_column=args.text_column)
    metrics = summarize(df[args.label_column].tolist(), predictions, labels)

    save_json(args.output, metrics)

    print(f"Evaluation on {args.data}: macro F1={metrics['macro_f1']:.3f}")
    print(f"Metrics written to {args.output}")
    print(f"Per-label F1: {metrics['per_label_f1']}")


if __name__ == "__main__":
    main()
