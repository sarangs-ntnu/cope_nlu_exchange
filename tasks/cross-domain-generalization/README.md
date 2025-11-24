# Cross-domain/generalization between teacher and course aspects

## Objective
Quantify transferability between teacher-focused and course-focused sentiment data and assess whether shared training helps generalization.

## Dataset assumptions
- Same columns as prior tasks; aspects may be imbalanced.

## Experiment steps
1. **Single-aspect training**: Train on teacher examples only and evaluate on course, then reverse. Record performance gaps.
2. **Combined training with aspect tokens**: Train on the merged dataset with explicit aspect prompts/tokens. Evaluate per aspect to measure gains.
3. **Domain adaptation variants**: Try simple fine-tuning from single-aspect model to the other aspect, and compare to training from scratch.
4. **Metrics**: Macro F1 per aspect; track degradation when transferring across aspects vs. in-domain baselines.
5. **Analysis**: Identify error categories unique to cross-domain testing (e.g., vocabulary shifts, pedagogical vs. curriculum language).

## Reporting
Highlight whether shared modeling narrows the gap between aspects and provide recommendations on when to train specialized vs. unified models.
