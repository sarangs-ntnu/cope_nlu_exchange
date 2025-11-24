# Aspect-aware sentiment experiments

This repository contains a script for training and evaluating sentiment models that account for multiple aspects in tabular feedback data.

## Requirements

Install dependencies (network access required):

```bash
pip install numpy pandas scikit-learn
```

## Usage

Provide a CSV or TSV file containing `aspect`, `sentiment`, and `comments` columns.

```bash
python sentiment_experiment.py --data path/to/data.csv --output results/
```

Key options:

- `--mode joint` (default): build a joint TF-IDF vocabulary with an aspect one-hot feature and train a single model.
- `--mode aspect`: train separate TF-IDF models per aspect and combine their validation predictions for evaluation.
- `--test-size`: validation split size (default `0.2`).

The script tunes Logistic Regression, Linear SVM, and Naive Bayes with cross-validation, then reports macro F1 and confusion matrices per aspect. Misclassified validation examples are written to CSV files for error analysis.
