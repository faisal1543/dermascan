"""Mocked perception for the API skeleton (DERM-13). Replaced by the real CNN in DERM-28.

It returns contract C1 without running a model: the image's SHA-256 picks one of the 10 made-up
cases in reasoning/fake_cases.json (DERM-12). The same image always gets the same output, and the
10 cases cover every urgency, including an out-of-distribution case that must give cannot_assess.
These are NOT model predictions and must never be shown as such outside development.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from app.config import MOCK_MODEL_VERSION

FAKE_CASES = Path(__file__).resolve().parents[1] / "reasoning" / "fake_cases.json"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # Ground rule 4
MALIGNANT = ("mel", "bcc", "akiec")

_CASES = json.loads(FAKE_CASES.read_text())["cases"]


def pick_case(image_sha256: str) -> dict:
    """Deterministic: the same hash always picks the same fake case."""
    return _CASES[int(image_sha256[:8], 16) % len(_CASES)]


def mock_c1(image_sha256: str) -> dict:
    """Contract C1 (live form) for an image, from the fake case its hash picks."""
    case = pick_case(image_sha256)
    probs = {c: float(case["c1"]["probs"][c]) for c in CLASSES}
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0) / math.log(len(CLASSES))
    return {
        "image_id": f"sha256:{image_sha256}",
        "probs": probs,
        "p_malignant": round(sum(probs[c] for c in MALIGNANT), 4),
        "entropy": round(entropy, 4),
        "in_distribution": bool(case["c1"].get("in_distribution", True)),
        "ood_score": None,  # set by the Mahalanobis check in DERM-28
        "heatmap_png_b64": None,  # Grad-CAM arrives with the real model in DERM-28
        "model_version": MOCK_MODEL_VERSION,
        "mock_case": case["case"],  # development only: which fake case was used
    }
