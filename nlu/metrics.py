from typing import Dict, List

from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    return f1_score(y_true, y_pred, average="macro")


def per_label_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=labels, average=None)
    return {label: float(score) for label, score in zip(labels, scores)}


def build_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, List[int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {label: row.tolist() for label, row in zip(labels, matrix)}


def summarize(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    """Return macro F1, per-aspect F1, and confusion matrix summary."""

    macro = macro_f1(y_true, y_pred)
    per_label = per_label_f1(y_true, y_pred, labels)
    matrix = build_confusion_matrix(y_true, y_pred, labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "macro_f1": macro,
        "per_label_f1": per_label,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
