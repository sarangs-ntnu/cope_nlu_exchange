# Dual-head aspect and sentiment classifier

This repository provides a compact PyTorch example for training a shared encoder with two heads:

* **Aspect classifier** – predicts whether a review is about a *teacher* or *course*.
* **Sentiment classifier** – predicts positive vs. negative sentiment.

The script also compares multi-task training against single-task fine-tuning and can inject noisy aspect labels to measure robustness.

## Running experiments

Install dependencies (PyTorch and scikit-learn must be available in your environment) and run:

```bash
python -m src.experiments --noise 0.3 --hierarchical --aspect-weight 1.2
```

Flags:

* `--noise` – probability of flipping aspect labels in the synthetic training data.
* `--hierarchical` – when set, the sentiment head consumes predicted aspect probabilities.
* `--aspect-weight` – weight assigned to the aspect loss when training jointly.

The command prints macro F1 for both heads and for single-task baselines, making it easy to compare hierarchical vs. flat multi-task training and their robustness to noisy aspect annotations.
