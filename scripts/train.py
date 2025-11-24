"""Train a baseline text classifier with reproducible settings."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlu.config import ModelConfig, RunConfig, SplitConfig
from nlu.metrics import summarize
from nlu.model import TextClassifier
from nlu.tracking import CsvTracker
from nlu.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text classifier")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"), help="Path to the train split")
    parser.add_argument("--dev", type=Path, default=Path("data/processed/dev.csv"), help="Path to the dev/validation split")
    parser.add_argument("--model-path", type=Path, default=Path("models/baseline.joblib"), help="Where to store the trained model")
    parser.add_argument("--experiment-name", type=str, default="baseline", help="Name for experiment tracking")
    parser.add_argument("--tracker-file", type=Path, default=Path("runs/experiments.csv"), help="CSV log for runs")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for model initialization")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF maximum features")
    parser.add_argument("--c", type=float, default=1.5, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram size")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram size")
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"], help="Penalty for LogisticRegression")
    parser.add_argument("--text-column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label-column", type=str, default="aspect", help="Name of the label column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_config = SplitConfig(
        text_column=args.text_column,
        label_column=args.label_column,
        random_seed=args.seed,
    )
    model_config = ModelConfig(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        c=args.c,
        penalty=args.penalty,
        random_seed=args.seed,
    )
    run_config = RunConfig(
        experiment_name=args.experiment_name,
        output_dir=args.model_path.parent,
        tracker_file=args.tracker_file,
        split=split_config,
        model=model_config,
    )
    run_config.ensure_dirs()

    train_df = pd.read_csv(args.train)
    dev_df = pd.read_csv(args.dev)

    classifier = TextClassifier(model_config)
    classifier.fit(train_df, text_column=args.text_column, label_column=args.label_column)

    labels = sorted(train_df[args.label_column].unique())
    dev_predictions = classifier.predict(dev_df, text_column=args.text_column)
    metrics = summarize(dev_df[args.label_column].tolist(), dev_predictions, labels)

    classifier.save(args.model_path)
    metrics_path = args.model_path.with_suffix(".metrics.json")
    save_json(metrics_path, metrics)

    tracker = CsvTracker(args.tracker_file)
    tracker.log(run_config.to_dict(), {"macro_f1": metrics["macro_f1"]})

    print(f"Saved model to {args.model_path}")
    print(f"Macro F1 on dev: {metrics['macro_f1']:.3f}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
