# COPE NLU Exchange

This repository provides a small, reproducible pipeline for aspect-based text classification experiments. It includes deterministic train/dev/test splits, macro and per-aspect F1 reporting, confusion matrices, and a lightweight CSV experiment tracker.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

A toy dataset is included at `data/raw/sample_reviews.csv`. Each row contains a `text` field and an `aspect` label. You can replace this file with your own data as long as those columns are present.

## Preprocessing

Create deterministic splits (70/15/15 by default) and record the split metadata:

```bash
python scripts/preprocess.py --input data/raw/sample_reviews.csv --output-dir data/processed --seed 1337
```

Outputs:
- `data/processed/train.csv`, `dev.csv`, `test.csv`
- `data/processed/split_metadata.json` describing the split configuration

## Training

Train a TF-IDF + Logistic Regression classifier, log the run configuration and macro F1 to `runs/experiments.csv`, and save metrics (including per-aspect F1 and a confusion matrix) alongside the model artifact:

```bash
python scripts/train.py --train data/processed/train.csv --dev data/processed/dev.csv --model-path models/baseline.joblib --seed 1337
```

Outputs:
- `models/baseline.joblib` (trained model)
- `models/baseline.metrics.json` (macro F1, per-aspect F1, confusion matrix)
- `runs/experiments.csv` (experiment tracker with configs and seeds)

Key hyperparameters (ngram range, regularization, and max features) are exposed as CLI flags.

## Evaluation

Evaluate a saved model on any split and store the full metric bundle:

```bash
python scripts/evaluate.py --data data/processed/test.csv --model-path models/baseline.joblib --output runs/eval_metrics.json
```

Outputs:
- `runs/eval_metrics.json` containing macro F1, per-aspect F1, confusion matrix, and the classification report

## Reproducibility

- All scripts accept `--seed` (or use the default `1337`) to fix randomness.
- Splits are stratified on the label column to ensure consistency across runs.
- The experiment tracker captures configuration values and seeds alongside metrics for transparent comparisons.
