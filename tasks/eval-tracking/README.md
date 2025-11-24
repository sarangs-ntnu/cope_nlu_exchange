# Evaluation protocol and experiment tracking

## Objective
Standardize metrics, splits, and logging to ensure reproducible comparisons across all sentiment experiments.

## Dataset assumptions
- Stable dataset versioning; record hashes or timestamps for each run.

## Experiment steps
1. **Splits**: Define train/validation/test splits with fixed random seeds and aspect-aware stratification. Store split indices to disk.
2. **Metrics**: Use macro F1 as the primary metric; also track per-aspect F1, accuracy, confusion matrices, and class distribution.
3. **Experiment logging**: Configure MLflow/W&B or a structured CSV/JSON logger to capture hyperparameters, seeds, dataset version, runtime, and metrics. Save model checkpoints and preprocessing artifacts.
4. **Reproducibility**: Set global seeds, log library versions, and capture command-line invocations or notebook versions.
5. **Visualization**: Plot training curves and metric tables; maintain a changelog of best results per task.

## Reporting
Provide instructions for replaying experiments using logged configs and note any deviations (e.g., hardware differences) that may affect results.

## How to run
1. Install tracking tools (example uses MLflow):
   ```bash
   pip install -r requirements.txt
   pip install mlflow
   ```
2. Create or reuse a split file to ensure consistent evaluation:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   import pandas as pd
   from sklearn.model_selection import train_test_split

   df = pd.read_csv('data/teacher_course.csv')
   strat = df['sentiment'].astype(str) + '__' + df['aspect'].astype(str)
   train_idx, val_idx = train_test_split(df.index, test_size=0.2, stratify=strat, random_state=42)
   Path('outputs/splits').mkdir(parents=True, exist_ok=True)
   Path('outputs/splits/default.json').write_text(
       json.dumps({'train': train_idx.tolist(), 'val': val_idx.tolist()}, indent=2)
   )
   PY
   ```
3. Log a baseline run with MLflow, capturing metrics and parameters:
   ```bash
   mlflow run . -e sentiment_experiment --env-manager=local -P data=data/teacher_course.csv -P output=outputs/mlflow/baseline
   ```
   (Alternatively, wrap your custom training script with `mlflow.start_run()` and log params/metrics manually.)
4. Use the logged split file and parameters to replay experiments; compare macro F1 across runs via `mlflow ui` for a local dashboard.
