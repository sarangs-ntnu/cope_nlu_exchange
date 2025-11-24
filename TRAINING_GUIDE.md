# Aspect Classification Training Guide

## Input formatting
- Represent each example as `Aspect: teacher | Text: <utterance>` so the model can focus on the target aspect.
- If a course identifier or teacher token exists, prepend a special token (e.g., `[TEACHER]`, `[COURSE]`) to highlight it.

## Model choice
- Start with a lightweight encoder such as `distilbert-base-uncased` to keep training economical and fast.

## Fine-tuning setup
- Train for 3–5 epochs with early stopping based on validation macro F1 to prevent overfitting.
- If label distribution is imbalanced, enable class weighting or focal loss to stabilize learning.
- Use a small batch size initially and apply gradient accumulation if GPU memory is limited.

## Evaluation and tracking
- Track validation macro F1 plus per-aspect F1 so rare classes are visible.
- Log loss and F1 curves for train/validation; keep the checkpoint with the best validation macro F1.
- After training, run inference on a held-out set and inspect misclassified examples per aspect to guide future data collection.

## Outputs to save
- Best model checkpoint.
- Training curves (loss, macro F1, per-aspect F1).
- A short report summarizing class balance, hyperparameters, and key metrics.
