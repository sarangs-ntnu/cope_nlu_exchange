# Baseline n-gram sentiment classifier (teacher vs course)

## Objective
Establish a transparent baseline for aspect-aware sentiment analysis using bag-of-ngrams with linear models, enabling quick comparisons with embedding and transformer approaches.

## Dataset assumptions
- Columns: `teacher/course` (aspect flag), `comments` (text), `sentiment` (label), optional `aspect` normalized value.
- Ensure consistent label casing (e.g., `positive`, `negative`, `neutral`).

## Experiment steps
1. **Data loading**: Read the tabular file (CSV/TSV). Normalize aspect names to `teacher` or `course` and clean whitespace.
2. **Split strategy**: Stratify train/validation by both `sentiment` and aspect to preserve distribution. Fix random seeds for reproducibility.
3. **Feature extraction**:
   - Word n-grams: 1–2 or 1–3 with min_df filtering.
   - Character n-grams: 3–5 for robustness to typos.
   - Optional: concatenate a one-hot aspect feature.
4. **Models to compare**: Logistic Regression (liblinear or saga), Linear SVM, Multinomial Naive Bayes (word-level only). Tune C/smoothing via cross-validation.
5. **Evaluation**: Report macro F1 overall and per aspect. Include confusion matrices and top misclassified examples.
6. **Artifacts**: Save TF-IDF vectorizers, model weights, and metrics JSON/CSV for later reuse.

## Reporting
Summarize best-performing configuration, runtime, and error patterns (e.g., negation, aspect confusion). Provide reproducible commands or notebook cells.
