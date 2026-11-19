"""Contract C1 as a CSV: one row per image, for tuning the reasoning layer offline.

Columns: image_id, dx (true label, for evaluation only), age, sex, localization,
p_akiec ... p_vasc (fixed class order), p_malignant (= mel + bcc + akiec),
entropy (normalised, 0-1), model_version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from perception.labels import CLASSES, MALIGNANT
from perception.metrics import normalised_entropy


def c1_frame(meta: pd.DataFrame, probs: np.ndarray, model_version: str) -> pd.DataFrame:
    """meta: rows aligned with probs, with image_id, dx, age, sex, localization."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(meta), len(CLASSES)):
        raise ValueError(f"probs shape {probs.shape} does not match {len(meta)} rows x {len(CLASSES)} classes")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each row of probs must sum to 1")

    out = meta[["image_id", "dx", "age", "sex", "localization"]].reset_index(drop=True).copy()
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i].round(6)
    out["p_malignant"] = sum(out[f"p_{c}"] for c in MALIGNANT).round(6)
    out["entropy"] = normalised_entropy(probs).round(6)
    out["model_version"] = model_version
    return out
