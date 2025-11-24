# Aspect-level Comment Classification

This repository demonstrates a lightweight experimentation loop for aspect-level
sentiment classification when network access to large model weights is
unavailable. It includes a small synthetic dataset of course-related comments
(`data/comments.csv`) and a standalone experiment script that emulates
state-of-the-art embedding backbones via deterministic hashing so that
comparisons between n-gram features and modern sentence embeddings remain
possible offline.

## Running experiments

```
python src/experiment.py --data data/comments.csv --output-dir outputs
```

The script will:

- Build a TF–IDF bigram baseline with a logistic regression classifier.
- Generate hashed embeddings that mimic
  `sentence-transformers/all-mpnet-base-v2` and `intfloat/e5-base-v2`, testing
  both aspect prompts (`[aspect] comment`) and concatenated aspect vectors.
- Sweep a few values of `C` and optional class balancing for the logistic
  regression head.
- Report macro F1 per aspect, embedding dimensionality, and end-to-end runtime
  for each configuration.

Outputs are written to `outputs/embedding_results.json` and
`outputs/embedding_report.md`.
