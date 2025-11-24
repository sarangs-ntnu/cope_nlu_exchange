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
