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
