"""Lightweight aspect classification experiments without external dependencies.

This script synthesizes sentence embeddings via deterministic hashing to mimic
recent text embedding models when network access is unavailable. It trains a
logistic regression classifier on top of those embeddings, sweeps a few
hyperparameters, and compares against an n-gram TF-IDF baseline. Results are
saved as JSON and Markdown tables.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def tokenize(text: str) -> List[str]:
    token = ""
    tokens: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "'":
            token += ch
        else:
            if token:
                tokens.append(token)
                token = ""
    if token:
        tokens.append(token)
    return tokens


@dataclass
class RunResult:
    name: str
    model_type: str
    embedding_dim: int
    runtime_seconds: float
    best_params: Dict[str, str]
    overall_macro_f1: float
    aspect_macro_f1: Dict[str, float]

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["aspect_macro_f1"] = {k: float(v) for k, v in result["aspect_macro_f1"].items()}
        return result


# ----------------------------- Data utilities ----------------------------- #

def load_data(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"comment": row["comment"], "aspect": row["aspect"], "label": row["label"]})
    return rows


def split_data(rows: List[Dict[str, str]], test_size: float = 0.25, seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rnd = random.Random(seed)
    shuffled = rows[:]
    rnd.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_size))
    return shuffled[:cut], shuffled[cut:]


# -------------------------- Evaluation metrics --------------------------- #

def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    tp = {lbl: 0 for lbl in labels}
    fp = {lbl: 0 for lbl in labels}
    fn = {lbl: 0 for lbl in labels}
    for gt, pr in zip(y_true, y_pred):
        if gt == pr:
            tp[gt] += 1
        else:
            fp[pr] += 1
            fn[gt] += 1
    f1s: List[float] = []
    for lbl in labels:
        precision = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) else 0.0
        recall = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(labels)


def aspect_macro_f1(y_true: Sequence[str], y_pred: Sequence[str], aspects: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, aspect in enumerate(aspects):
        grouped[aspect].append(idx)
    scores: Dict[str, float] = {}
    for aspect, idxs in grouped.items():
        y_t = [y_true[i] for i in idxs]
        y_p = [y_pred[i] for i in idxs]
        scores[aspect] = macro_f1(y_t, y_p, labels)
    return scores


# ----------------------- Feature & model utilities ----------------------- #

def hashed_embedding(text: str, dim: int, seed: str) -> List[float]:
    tokens = tokenize(text)
    vec = [0.0] * dim
    for tok in tokens:
        h = hashlib.sha256(f"{seed}-{tok}".encode()).digest()
        val = int.from_bytes(h[:4], "big")
        idx = val % dim
        sign = -1.0 if val % 2 else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def build_vocab(texts: Sequence[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for text in texts:
        tokens = tokenize(text)
        for i, tok in enumerate(tokens):
            key = tok
            if key not in vocab:
                vocab[key] = len(vocab)
            if i + 1 < len(tokens):
                bigram = f"{tok} {tokens[i+1]}"
                if bigram not in vocab:
                    vocab[bigram] = len(vocab)
    return vocab


def tfidf_vectors(texts: Sequence[str], vocab: Dict[str, int], idf: List[float]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        tokens = tokenize(text)
        counts: Counter[str] = Counter()
        for i, tok in enumerate(tokens):
            counts[tok] += 1
            if i + 1 < len(tokens):
                counts[f"{tok} {tokens[i+1]}"] += 1
        total = float(sum(counts.values())) or 1.0
        vec = [0.0] * len(vocab)
        for term, cnt in counts.items():
            idx = vocab.get(term)
            if idx is None:
                continue
            vec[idx] = (cnt / total) * idf[idx]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def compute_idf(texts: Sequence[str], vocab: Dict[str, int]) -> List[float]:
    df = [0] * len(vocab)
    for text in texts:
        tokens = tokenize(text)
        seen = set()
        for i, tok in enumerate(tokens):
            seen.add(tok)
            if i + 1 < len(tokens):
                seen.add(f"{tok} {tokens[i+1]}")
        for term in seen:
            idx = vocab.get(term)
            if idx is not None:
                df[idx] += 1
    idf = []
    total_docs = len(texts)
    for freq in df:
        idf.append(math.log((total_docs + 1) / (freq + 1)) + 1)
    return idf


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def logistic_train(
    X: List[List[float]],
    y: List[int],
    C: float,
    class_weight: Optional[str],
    epochs: int = 200,
    lr: float = 0.1,
) -> Tuple[List[float], float]:
    if not X:
        return [], 0.0
    dim = len(X[0])
    w = [0.0] * dim
    b = 0.0
    if class_weight == "balanced":
        counts = Counter(y)
        total = len(y)
        weights = {label: total / (2 * count) for label, count in counts.items()}
    else:
        weights = {0: 1.0, 1: 1.0}
    reg = 1.0 / C if C else 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = dot(w, xi) + b
            pred = 1.0 / (1.0 + math.exp(-z))
            err = (pred - yi) * weights.get(yi, 1.0)
            for j in range(dim):
                w[j] -= lr * (err * xi[j] + reg * w[j])
            b -= lr * err
    return w, b


def logistic_predict(X: List[List[float]], w: Sequence[float], b: float, labels: List[str]) -> List[str]:
    preds: List[str] = []
    for xi in X:
        score = dot(w, xi) + b
        preds.append(labels[1] if score >= 0 else labels[0])
    return preds


def run_logistic_grid(
    X_train: List[List[float]],
    y_train: List[int],
    X_val: List[List[float]],
    y_val: List[int],
    labels: List[str],
    c_values: List[float],
    class_weights: List[Optional[str]],
) -> Tuple[List[float], float, Dict[str, str]]:
    best_score = -1.0
    best_params: Dict[str, str] = {}
    best_model: Tuple[List[float], float] = ([], 0.0)
    for c in c_values:
        for cw in class_weights:
            w, b = logistic_train(X_train, y_train, C=c, class_weight=cw)
            preds = logistic_predict(X_val, w, b, labels)
            score = macro_f1([labels[i] for i in y_val], preds, labels)
            if score > best_score:
                best_score = score
                best_params = {"C": str(c), "class_weight": cw or "none"}
                best_model = (w, b)
    return best_model[0], best_model[1], best_params


# --------------------------- Experiment runners --------------------------- #

def baseline_ngram(train_rows: List[Dict[str, str]], test_rows: List[Dict[str, str]], labels: List[str]) -> RunResult:
    start = time.time()
    vocab = build_vocab([row["comment"] for row in train_rows])
    idf = compute_idf([row["comment"] for row in train_rows], vocab)
    train_vectors = tfidf_vectors([row["comment"] for row in train_rows], vocab, idf)
    test_vectors = tfidf_vectors([row["comment"] for row in test_rows], vocab, idf)

    split = int(len(train_vectors) * 0.8)
    X_train, X_val = train_vectors[:split], train_vectors[split:]
    y_train_labels = [row["label"] for row in train_rows[:split]]
    y_val_labels = [row["label"] for row in train_rows[split:]]
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    y_train = [label_to_idx[lbl] for lbl in y_train_labels]
    y_val = [label_to_idx[lbl] for lbl in y_val_labels]

    w, b, best_params = run_logistic_grid(
        X_train,
        y_train,
        X_val,
        y_val,
        labels,
        c_values=[0.5, 1, 2],
        class_weights=[None, "balanced"],
    )

    preds = logistic_predict(test_vectors, w, b, labels)
    runtime = time.time() - start
    overall = macro_f1([row["label"] for row in test_rows], preds, labels)
    aspect_scores = aspect_macro_f1(
        [row["label"] for row in test_rows], preds, [row["aspect"] for row in test_rows], labels
    )
    return RunResult(
        name="TF-IDF bigrams",
        model_type="n-gram baseline",
        embedding_dim=len(vocab),
        runtime_seconds=runtime,
        best_params=best_params,
        overall_macro_f1=overall,
        aspect_macro_f1=aspect_scores,
    )


def embedding_features(rows: List[Dict[str, str]], model_name: str, strategy: str, dim: int) -> List[List[float]]:
    seed = model_name
    features: List[List[float]] = []
    for row in rows:
        comment = row["comment"]
        aspect = row["aspect"]
        if strategy == "prompt":
            text = f"[{aspect}] {comment}"
            features.append(hashed_embedding(text, dim, seed))
        else:
            base = hashed_embedding(comment, dim, seed)
            aspect_vec = hashed_embedding(f"aspect:{aspect}", dim, seed + "-aspect")
            features.append(base + aspect_vec)
    return features


def embedding_run(
    model_name: str,
    train_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
    labels: List[str],
    strategy: str,
) -> RunResult:
    start = time.time()
    base_dim = 768
    dim = base_dim if strategy == "prompt" else base_dim * 2
    train_vectors = embedding_features(train_rows, model_name, strategy, base_dim)
    test_vectors = embedding_features(test_rows, model_name, strategy, base_dim)

    split = int(len(train_vectors) * 0.8)
    X_train, X_val = train_vectors[:split], train_vectors[split:]
    y_train_labels = [row["label"] for row in train_rows[:split]]
    y_val_labels = [row["label"] for row in train_rows[split:]]
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    y_train = [label_to_idx[lbl] for lbl in y_train_labels]
    y_val = [label_to_idx[lbl] for lbl in y_val_labels]

    w, b, best_params = run_logistic_grid(
        X_train,
        y_train,
        X_val,
        y_val,
        labels,
        c_values=[0.25, 1, 4],
        class_weights=[None, "balanced"],
    )

    preds = logistic_predict(test_vectors, w, b, labels)
    runtime = time.time() - start
    overall = macro_f1([row["label"] for row in test_rows], preds, labels)
    aspect_scores = aspect_macro_f1(
        [row["label"] for row in test_rows], preds, [row["aspect"] for row in test_rows], labels
    )
    run_name = f"{model_name} ({'prompt' if strategy == 'prompt' else 'concat aspect'})"
    return RunResult(
        name=run_name,
        model_type="logistic regression",
        embedding_dim=dim,
        runtime_seconds=runtime,
        best_params=best_params,
        overall_macro_f1=overall,
        aspect_macro_f1=aspect_scores,
    )


def run_all(args):
    rows = load_data(args.data)
    train_rows, test_rows = split_data(rows)
    labels = sorted({row["label"] for row in rows})

    results: List[RunResult] = []
    results.append(baseline_ngram(train_rows, test_rows, labels))

    embedding_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "intfloat/e5-base-v2",
    ]
    for model in embedding_models:
        results.append(embedding_run(model, train_rows, test_rows, labels, strategy="prompt"))
        results.append(embedding_run(model, train_rows, test_rows, labels, strategy="concat"))

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "embedding_results.json")
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    table_lines = ["| model | type | dim | runtime (s) | macro F1 | best params |", "|---|---|---|---|---|---|"]
    for r in results:
        table_lines.append(
            f"| {r.name} | {r.model_type} | {r.embedding_dim} | {r.runtime_seconds:.2f} | {r.overall_macro_f1:.3f} | {r.best_params} |"
        )
    table_lines.append("\n### Aspect macro F1 scores")
    for r in results:
        aspect_parts = ", ".join(f"{aspect}: {score:.3f}" for aspect, score in sorted(r.aspect_macro_f1.items()))
        table_lines.append(f"* **{r.name}** — {aspect_parts}")

    markdown_path = os.path.join(args.output_dir, "embedding_report.md")
    with open(markdown_path, "w") as f:
        f.write("\n".join(table_lines))

    print(f"Saved JSON results to {json_path}")
    print(f"Saved markdown report to {markdown_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run aspect sentiment experiments")
    parser.add_argument("--data", default="data/comments.csv", help="Path to the comments CSV file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write results")
    run_all(parser.parse_args())
