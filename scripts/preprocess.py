"""Preprocess raw reviews and create deterministic splits."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlu.config import SplitConfig
from nlu.data import Splitter
from nlu.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible dataset splits")
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Where to store the splits")
    parser.add_argument("--train-size", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--dev-size", type=float, default=0.15, help="Proportion of data for development")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
    parser.add_argument("--text-column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label-column", type=str, default="aspect", help="Name of the label column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        train_size=args.train_size,
        dev_size=args.dev_size,
        random_seed=args.seed,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    splitter = Splitter(config)
    df = splitter.load(args.input)
    train_df, dev_df, test_df = splitter.split(df)
    splitter.save_splits(train_df, dev_df, test_df, args.output_dir)

    metadata = {
        "train_size": config.train_size,
        "dev_size": config.dev_size,
        "random_seed": config.random_seed,
        "text_column": config.text_column,
        "label_column": config.label_column,
        "source": str(args.input),
    }
    save_json(args.output_dir / "split_metadata.json", metadata)
    print(f"Saved splits to {args.output_dir}")


if __name__ == "__main__":
    main()
