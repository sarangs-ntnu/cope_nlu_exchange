| model | type | dim | runtime (s) | macro F1 | best params |
|---|---|---|---|---|---|
| TF-IDF bigrams | n-gram baseline | 387 | 1.42 | 0.231 | {'C': '0.5', 'class_weight': 'none'} |
| sentence-transformers/all-mpnet-base-v2 (prompt) | logistic regression | 768 | 2.91 | 0.231 | {'C': '0.25', 'class_weight': 'none'} |
| sentence-transformers/all-mpnet-base-v2 (concat aspect) | logistic regression | 1536 | 5.77 | 0.231 | {'C': '0.25', 'class_weight': 'none'} |
| intfloat/e5-base-v2 (prompt) | logistic regression | 768 | 2.83 | 0.231 | {'C': '0.25', 'class_weight': 'none'} |
| intfloat/e5-base-v2 (concat aspect) | logistic regression | 1536 | 5.73 | 0.231 | {'C': '0.25', 'class_weight': 'none'} |

### Aspect macro F1 scores
* **TF-IDF bigrams** — materials: 0.333, schedule: 0.000, teacher: 0.167
* **sentence-transformers/all-mpnet-base-v2 (prompt)** — materials: 0.333, schedule: 0.000, teacher: 0.167
* **sentence-transformers/all-mpnet-base-v2 (concat aspect)** — materials: 0.333, schedule: 0.000, teacher: 0.167
* **intfloat/e5-base-v2 (prompt)** — materials: 0.333, schedule: 0.000, teacher: 0.167
* **intfloat/e5-base-v2 (concat aspect)** — materials: 0.333, schedule: 0.000, teacher: 0.167