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

## How to run from the command line
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train on **teacher only**, evaluate on **course**:
   ```bash
   python - <<'PY'
   from pathlib import Path
   import pandas as pd
   from sentiment_experiment import load_dataset, evaluate_aspect_specific_models

   df = load_dataset(Path('data/teacher_course.csv'))
   teacher_df = df[df['aspect'].str.lower() == 'teacher']
   course_df = df[df['aspect'].str.lower() == 'course']

   evaluate_aspect_specific_models(teacher_df, course_df, Path('outputs/cross_domain/teacher_to_course'))
   PY
   ```
3. Train on **course only**, evaluate on **teacher**:
   ```bash
   python - <<'PY'
   from pathlib import Path
   import pandas as pd
   from sentiment_experiment import load_dataset, evaluate_aspect_specific_models

   df = load_dataset(Path('data/teacher_course.csv'))
   teacher_df = df[df['aspect'].str.lower() == 'teacher']
   course_df = df[df['aspect'].str.lower() == 'course']

   evaluate_aspect_specific_models(course_df, teacher_df, Path('outputs/cross_domain/course_to_teacher'))
   PY
   ```
4. Train a **combined joint model** to measure the benefit of shared training:
   ```bash
   python sentiment_experiment.py --data data/teacher_course.csv --output outputs/cross_domain/combined --mode joint
   ```
5. Compare macro F1 and confusion matrices across `outputs/cross_domain/*` to quantify transfer gaps.
