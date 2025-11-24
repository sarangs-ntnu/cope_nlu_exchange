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
