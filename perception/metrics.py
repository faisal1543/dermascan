"""Evaluation metrics (numpy only, so CI can test them without PyTorch).

Ground rule 6: never accuracy alone. Report balanced accuracy and per-class recall with n and a
95% Wilson confidence interval.
"""
from __future__ import annotations

import math

import numpy as np

from perception.labels import CLASSES


def confusion_matrix(y_true, y_pred, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Returns (nan, nan) when n == 0."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_class_report(y_true, y_pred, classes=CLASSES) -> list[dict]:
    """Recall (with n and Wilson CI) and precision for every class."""
    cm = confusion_matrix(y_true, y_pred, len(classes))
    rows = []
    for i, name in enumerate(classes):
        n = int(cm[i].sum())
        tp = int(cm[i, i])
        predicted = int(cm[:, i].sum())
        lo, hi = wilson_interval(tp, n)
        rows.append({
            "class": name,
            "n": n,
            "recall": tp / n if n else math.nan,
            "recall_ci_low": lo,
            "recall_ci_high": hi,
            "precision": tp / predicted if predicted else math.nan,
        })
    return rows


def balanced_accuracy(y_true, y_pred, classes=CLASSES) -> float:
    """Mean recall over classes that appear in y_true."""
    recalls = [r["recall"] for r in per_class_report(y_true, y_pred, classes) if r["n"] > 0]
    return float(np.mean(recalls)) if recalls else math.nan


def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row divided by log(K): 0 = certain, 1 = uniform."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / math.log(p.shape[1])


def summarise(y_true, probs: np.ndarray, classes=CLASSES) -> dict:
    """Headline numbers for one split from true labels and predicted probabilities."""
    y_pred = np.asarray(probs).argmax(axis=1)
    report = per_class_report(y_true, y_pred, classes)
    mel = next(r for r in report if r["class"] == "mel")
    return {
        "n": int(len(y_true)),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, classes),
        "mel_recall": mel["recall"],
        "mel_recall_ci": [mel["recall_ci_low"], mel["recall_ci_high"]],
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(classes)).tolist(),
    }
