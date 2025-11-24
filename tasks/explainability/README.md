# Explainability and error analysis

## Objective
Provide interpretable insights into model decisions and diagnose common failure modes across aspects.

## Dataset assumptions
- Access to trained models from prior experiments (linear, embedding-based, transformer).

## Experiment steps
1. **Linear models**: Extract top weighted n-grams per sentiment class and aspect from Logistic Regression/SVM to surface indicative phrases.
2. **Transformer attributions**: Apply techniques like Integrated Gradients or attention rollout on representative teacher and course examples.
3. **Counterfactuals**: Use minimal edits (e.g., swap aspect tokens or key words) to probe model sensitivity.
4. **Error logging**: Collect misclassified examples with predicted vs. gold labels and confidence; categorize by phenomenon (negation, sarcasm, aspect confusion).
5. **Visualization**: Generate small tables/plots summarizing influential tokens and frequent error categories.

## Reporting
Summarize interpretability findings and actionable data/model improvements (e.g., add negation examples, strengthen aspect prompts).

## How to run from the command line
1. Install visualization and interpretability helpers:
   ```bash
   pip install -r requirements.txt
   pip install captum matplotlib seaborn
   ```
2. Inspect linear model weights from the TF-IDF baseline:
   ```bash
   python - <<'PY'
   import numpy as np
   from sentiment_experiment import load_dataset, stratified_split, build_text_union, joint_feature_transformer, model_grids
   from pathlib import Path
   from sklearn.pipeline import Pipeline

   df = load_dataset(Path('data/teacher_course.csv'))
   train_df, _ = stratified_split(df, test_size=0.2)
   text_union = build_text_union()
   transformer = joint_feature_transformer(text_union)
   clf, _ = list(model_grids().values())[0]

   pipeline = Pipeline([('features', transformer), ('clf', clf[0])])
   pipeline.fit(train_df[['comments', 'aspect']], train_df['sentiment'])
   vocab = pipeline.named_steps['features'].transformers_[0][1].transformer_list[0][1].get_feature_names_out()
   coefs = pipeline.named_steps['clf'].coef_
   top = np.argsort(coefs, axis=1)[:, -10:]
   print('Top word features per class:')
   for idx, class_name in enumerate(pipeline.named_steps['clf'].classes_):
       print(class_name, [vocab[i] for i in top[idx]])
   PY
   ```
3. For transformer attributions, load your fine-tuned model (from the transformer tasks) and run Captum Integrated Gradients over a few teacher/course examples, then visualize token importances with matplotlib (e.g., matplotlib bar plots) to share in reports.
4. Collect misclassified rows from earlier runs (e.g., `*_misclassified.csv` in `outputs/`) and group them by error type for qualitative notes.
