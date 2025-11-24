# Data augmentation and robustness checks

## Objective
Increase training data diversity and measure robustness of sentiment models to noise and paraphrasing while preserving aspect cues.

## Dataset assumptions
- Labeled data with `comments`, `teacher/course`, `sentiment`.
- Augmentations must keep the aspect label intact.

## Experiment steps
1. **Augmentation methods**:
   - Back-translation or paraphrase generation while retaining aspect markers in text/prompt.
   - Light noise: typos, punctuation changes, and synonym swaps using tools like `nlpaug`.
2. **Dataset construction**: Blend original and augmented samples; cap augmentation ratio to avoid overpowering the original distribution.
3. **Retraining**: Use the best-performing supervised model (e.g., embeddings or transformer) and retrain on the augmented dataset.
4. **Robustness evaluation**: Evaluate on clean validation data and on a noised validation split. Track macro F1 deltas and degradation.
5. **Diagnostics**: Inspect whether augmentation harms aspect disambiguation or label consistency.

## Reporting
Summarize augmentation recipes, sizes, and their impact on metrics. Provide scripts/configs to reproduce augmented datasets.

## How to run from the command line
1. Install augmentation dependencies (example uses `nlpaug`):
   ```bash
   pip install -r requirements.txt
   pip install nlpaug
   ```
2. Generate augmented samples that preserve aspect cues:
   ```bash
   python - <<'PY'
   import nlpaug.augmenter.word as naw
   import pandas as pd

   df = pd.read_csv('data/teacher_course.csv')
   aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
   df['augmented'] = [aug.augment(text) for text in df['comments']]
   df[['aspect', 'sentiment', 'augmented']].rename(columns={'augmented': 'comments'}).to_csv('outputs/data_augmentation/augmented.csv', index=False)
   PY
   ```
3. Merge original and augmented rows, then retrain your best model (e.g., the n-gram baseline):
   ```bash
   python - <<'PY'
   import pandas as pd

   orig = pd.read_csv('data/teacher_course.csv')
   aug = pd.read_csv('outputs/data_augmentation/augmented.csv')
   combined = pd.concat([orig, aug], ignore_index=True)
   combined.to_csv('outputs/data_augmentation/combined.csv', index=False)
   PY

   python sentiment_experiment.py --data outputs/data_augmentation/combined.csv --output outputs/data_augmentation/run --mode joint
   ```
4. Evaluate robustness by running the same model on a noised validation split (e.g., add typos via `nlpaug` before inference) and compare macro F1 to the clean validation metrics.
