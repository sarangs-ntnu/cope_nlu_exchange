"""
Utilities for training and evaluating sentiment models with aspect information.

This module can be used as a script. Example usage:
    python sentiment_experiment.py --data path/to/data.csv --output results/

The script loads tabular data containing `aspect`, `sentiment`, and `comments`
columns, builds TF-IDF features (word and character n-grams), trains multiple
linear text classifiers with hyperparameter tuning, and reports metrics per
aspect including macro F1 and confusion matrices. Misclassified examples are
also written to disk for error analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load a CSV/TSV file containing aspect, sentiment, and comments columns.

    Args:
        data_path: Path to the input file.

    Returns:
        DataFrame with the required columns.
    """
    if data_path.suffix.lower() == ".tsv":
        sep = "\t"
    else:
        sep = ","
    df = pd.read_csv(data_path, sep=sep)
    required_columns = {"aspect", "sentiment", "comments"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return df[list(required_columns)].copy()


def stratified_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and validation sets stratified by sentiment and aspect.
    """
    stratify_labels = df["sentiment"].astype(str) + "__" + df["aspect"].astype(str)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_text_union(word_ngrams: Tuple[int, int] = (1, 2), char_ngrams: Tuple[int, int] = (3, 5)) -> FeatureUnion:
    """Create a feature union of word and character TF-IDF vectorizers."""
    word_vectorizer = TfidfVectorizer(ngram_range=word_ngrams, lowercase=True, min_df=2)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=char_ngrams, min_df=2)
    return FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer),
    ])


def joint_feature_transformer(text_union: FeatureUnion) -> ColumnTransformer:
    """Combine text features with one-hot encoded aspects."""
    return ColumnTransformer(
        [
            ("text", text_union, "comments"),
            ("aspect", OneHotEncoder(handle_unknown="ignore"), ["aspect"]),
        ]
    )


def model_grids() -> Dict[str, Tuple[object, Dict[str, List[float]]]]:
    """
    Define estimators and their hyperparameter grids.

    Returns:
        Mapping of model name to (estimator, grid) tuple.
    """
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"clf__C": [0.5, 1.0, 2.0]},
        ),
        "linear_svm": (
            LinearSVC(),
            {"clf__C": [0.5, 1.0, 2.0]},
        ),
        "naive_bayes": (
            ComplementNB(),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
    }


def run_grid_search(pipeline: Pipeline, grid: Dict[str, Iterable[float]], X, y) -> GridSearchCV:
    """Run cross-validated grid search with macro F1 scoring."""
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=3,
        n_jobs=-1,
        scoring="f1_macro",
    )
    search.fit(X, y)
    return search


def evaluate_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
) -> Tuple[float, np.ndarray]:
    """Compute macro F1 and confusion matrix."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return macro_f1, matrix


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_misclassified(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def evaluate_joint_models(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path) -> List[dict]:
    text_union = build_text_union()
    transformer = joint_feature_transformer(text_union)
    labels = sorted(train_df["sentiment"].unique())
    model_results = []

    for model_name, (estimator, grid) in model_grids().items():
        pipeline = Pipeline([
            ("features", transformer),
            ("clf", estimator),
        ])
        search = run_grid_search(pipeline, grid, train_df[["comments", "aspect"]], train_df["sentiment"])
        val_pred = search.predict(val_df[["comments", "aspect"]])
        macro_f1, matrix = evaluate_predictions(val_df["sentiment"], val_pred, labels)

        aspect_metrics = {}
        misclassified_rows = []
        for aspect, group in val_df.assign(prediction=val_pred).groupby("aspect"):
            aspect_f1, aspect_matrix = evaluate_predictions(
                group["sentiment"], group["prediction"], labels
            )
            aspect_metrics[aspect] = {
                "macro_f1": aspect_f1,
                "confusion_matrix": aspect_matrix.tolist(),
            }
            misclassified = group[group["sentiment"] != group["prediction"]][
                ["aspect", "sentiment", "prediction", "comments"]
            ]
            misclassified_rows.append(misclassified)

        misclassified_df = pd.concat(misclassified_rows, ignore_index=True) if misclassified_rows else pd.DataFrame()
        save_misclassified(
            misclassified_df,
            output_dir / f"joint_{model_name}_misclassified.csv",
        )

        metrics = {
            "mode": "joint",
            "model": model_name,
            "best_params": search.best_params_,
            "overall_macro_f1": macro_f1,
            "confusion_matrix": matrix.tolist(),
            "per_aspect": aspect_metrics,
        }
        save_json(metrics, output_dir / f"joint_{model_name}_metrics.json")
        model_results.append(metrics)

    return model_results


def evaluate_aspect_specific_models(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path) -> List[dict]:
    text_union = build_text_union()
    labels = sorted(train_df["sentiment"].unique())
    aspects = sorted(train_df["aspect"].unique())
    overall_results = []

    for model_name, (estimator, grid) in model_grids().items():
        aspect_predictions = []
        aspect_truths = []
        aspect_details = {}
        misclassified_rows = []

        for aspect in aspects:
            train_subset = train_df[train_df["aspect"] == aspect]
            val_subset = val_df[val_df["aspect"] == aspect]
            pipeline = Pipeline([
                ("features", text_union),
                ("clf", estimator),
            ])
            search = run_grid_search(pipeline, grid, train_subset["comments"], train_subset["sentiment"])
            val_pred = search.predict(val_subset["comments"])

            aspect_truths.extend(val_subset["sentiment"].tolist())
            aspect_predictions.extend(val_pred.tolist())

            aspect_f1, aspect_matrix = evaluate_predictions(val_subset["sentiment"], val_pred, labels)
            aspect_details[aspect] = {
                "macro_f1": aspect_f1,
                "confusion_matrix": aspect_matrix.tolist(),
                "best_params": search.best_params_,
            }
            misclassified = val_subset.assign(prediction=val_pred)
            misclassified = misclassified[misclassified["sentiment"] != misclassified["prediction"]][
                ["aspect", "sentiment", "prediction", "comments"]
            ]
            misclassified_rows.append(misclassified)

        macro_f1, matrix = evaluate_predictions(aspect_truths, aspect_predictions, labels)
        misclassified_df = pd.concat(misclassified_rows, ignore_index=True) if misclassified_rows else pd.DataFrame()
        save_misclassified(
            misclassified_df,
            output_dir / f"aspect_specific_{model_name}_misclassified.csv",
        )

        metrics = {
            "mode": "aspect_specific",
            "model": model_name,
            "overall_macro_f1": macro_f1,
            "confusion_matrix": matrix.tolist(),
            "per_aspect": aspect_details,
        }
        save_json(metrics, output_dir / f"aspect_specific_{model_name}_metrics.json")
        overall_results.append(metrics)

    return overall_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate aspect-aware sentiment models.")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV/TSV file containing aspect, sentiment, and comments columns.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write metrics and misclassified rows.")
    parser.add_argument(
        "--mode",
        choices=["joint", "aspect"],
        default="joint",
        help="Use a joint vocabulary with aspect one-hot features or train separate models per aspect.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size. Default: 0.2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    train_df, val_df = stratified_split(df, test_size=args.test_size)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "joint":
        metrics = evaluate_joint_models(train_df, val_df, args.output)
    else:
        metrics = evaluate_aspect_specific_models(train_df, val_df, args.output)

    save_json({"mode": args.mode, "results": metrics}, args.output / "summary.json")


if __name__ == "__main__":
    main()
