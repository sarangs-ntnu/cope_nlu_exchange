# Sentence embedding classifier (Sentence-BERT / E5)

## Objective
Leverage modern sentence embeddings to improve aspect-aware sentiment classification with minimal fine-tuning overhead.

## Dataset assumptions
- Same columns as baseline: `teacher/course`, `comments`, `sentiment`.
- Optional aspect prompt added to text (e.g., "[teacher] comment").

## Experiment steps
1. **Model selection**: Start with `sentence-transformers/all-mpnet-base-v2` and `intfloat/e5-base-v2`; note embedding sizes and token limits.
2. **Embedding generation**:
   - Prepend or append aspect tokens/prompts to the comment before encoding.
   - Batch inference with GPU if available; persist embeddings to disk (e.g., NumPy/Parquet) for reuse.
3. **Classifier training**: Train Logistic Regression or Linear SVM on frozen embeddings. Sweep C, class weights, and optionally apply feature scaling.
4. **Evaluation**: Macro F1 overall and per aspect. Compare against the n-gram baseline and record runtime and memory use.
5. **Ablations**: Test with/without aspect prompt, pooled vs. first-token embeddings, and reduced-dimension embeddings via PCA.

## Reporting
Document the best model/configuration, hyperparameters, and notable errors. Include scripts/notebooks and random seeds for reproducibility.

## How to run from the command line
1. Install the core and sentence-transformer dependencies:
   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers
   ```
2. Generate embeddings for your dataset (example with E5 and an aspect prompt):
   ```bash
   python - <<'PY'
   from sentence_transformers import SentenceTransformer
   import pandas as pd
   from pathlib import Path

   df = pd.read_csv('data/teacher_course.csv')
   df['prompted'] = df['aspect'].str.lower().radd('Aspect: ').str.cat(df['comments'], sep=' | Text: ')

   model = SentenceTransformer('intfloat/e5-base-v2')
   embeddings = model.encode(df['prompted'].tolist(), batch_size=64, convert_to_numpy=True)

   out_dir = Path('outputs/sentence_embeddings')
   out_dir.mkdir(parents=True, exist_ok=True)
   pd.DataFrame(embeddings).to_parquet(out_dir / 'e5_embeddings.parquet')
   df[['sentiment']].to_csv(out_dir / 'labels.csv', index=False)
   PY
   ```
3. Train a lightweight classifier (e.g., Logistic Regression) on the saved embeddings:
   ```bash
   python - <<'PY'
   import pandas as pd
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report

   X = pd.read_parquet('outputs/sentence_embeddings/e5_embeddings.parquet')
   y = pd.read_csv('outputs/sentence_embeddings/labels.csv')['sentiment']
   X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
   clf = LogisticRegression(max_iter=2000)
   clf.fit(X_train, y_train)
   print(classification_report(y_val, clf.predict(X_val)))
   PY
   ```
