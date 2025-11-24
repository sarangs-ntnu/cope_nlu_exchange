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

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train on a single aspect and evaluate on the other using the TF-IDF baseline:
   ```bash
   python - <<'PY'
   import pandas as pd
   from pathlib import Path
   from sentiment_experiment import load_dataset, stratified_split, evaluate_aspect_specific_models, evaluate_joint_models

   df = load_dataset(Path('data/teacher_course.csv'))
   teacher_df = df[df['aspect'].str.lower() == 'teacher']
   course_df = df[df['aspect'].str.lower() == 'course']

   # train on teacher, test on course
   train_df, _ = stratified_split(teacher_df, test_size=0.0)
   evaluate_aspect_specific_models(train_df, course_df, Path('outputs/cross_domain/teacher_to_course'))

   # train on course, test on teacher
   train_df, _ = stratified_split(course_df, test_size=0.0)
   evaluate_aspect_specific_models(train_df, teacher_df, Path('outputs/cross_domain/course_to_teacher'))
   PY
   ```
3. Train on combined data with aspect tokens and evaluate per aspect:
   ```bash
   python sentiment_experiment.py --data data/teacher_course.csv --output outputs/cross_domain/combined --mode joint
   ```
4. Compare macro F1 across the cross-domain and combined runs to quantify transfer gaps.
