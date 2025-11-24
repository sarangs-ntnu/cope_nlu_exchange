# COPE NLU Exchange

This repository demonstrates a lightweight data augmentation and retraining pipeline for aspect-aware NLU tasks.

## Data
Example labeled data lives in `data/` with bracketed aspect tokens that must be preserved during augmentation.

## Usage
1. (Optional) Install dependencies into a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Run augmentation, training, and evaluation in a single step:
   ```bash
   python src/train_and_eval.py --paraphrases 1 --noise-prob 0.1
   ```
   Add `--use-nlpaug` to leverage `nlpaug` synonym and character insertion augmenters when available.

Artifacts (the trained model and evaluation metrics) are saved to `artifacts/`.
