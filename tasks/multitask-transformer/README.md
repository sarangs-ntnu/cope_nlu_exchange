# Multi-task transformer for aspect and sentiment

## Objective
Train a shared encoder with dual heads to jointly predict aspect (teacher vs course) and sentiment, testing whether shared supervision improves robustness.

## Dataset assumptions
- Columns: `comments`, `teacher/course` (aspect), `sentiment`.
- Aspect labels may contain noise; consider smoothing/cleaning.

## Experiment steps
1. **Model architecture**: Shared transformer encoder with two classifiers: (a) aspect head, (b) sentiment head. Optionally condition the sentiment head on the predicted aspect distribution.
2. **Loss design**: Weighted sum of aspect loss and sentiment loss. Sweep loss weights to balance tasks.
3. **Training**: Use AdamW with linear warmup. Train 3–6 epochs with early stopping on sentiment macro F1.
4. **Ablations**: Compare joint training against single-task sentiment fine-tuning and a hierarchical variant (predict aspect then sentiment).
5. **Evaluation**: Report macro F1 for sentiment overall and per aspect. Track aspect accuracy to see transfer effects.
6. **Artifacts**: Save best checkpoint, tokenizer, and a config JSON documenting loss weights and hyperparameters.

## Reporting
Highlight whether multi-task learning improves sentiment performance or stability under noisy aspect labels, including qualitative examples.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the provided dual-head experiment script (synthetic data with controllable aspect noise):
   ```bash
   python -m src.experiments --noise 0.3 --hierarchical --aspect-weight 1.2
   ```
   - Omit `--hierarchical` to compare flat multi-task training.
   - Adjust `--noise` to simulate different aspect corruption rates.
3. Replace the synthetic dataset with your real CSV/TSV by adapting `src/data.py` (or writing a small loader) to emit real examples, then rerun the command above to evaluate on your data.
4. Log results (macro F1 for both heads) and compare against single-task baselines printed by the script.
