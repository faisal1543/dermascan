import math

import numpy as np
import pandas as pd
import pytest

from perception.labels import CLASSES
from perception.metrics import (
    balanced_accuracy,
    confusion_matrix,
    normalised_entropy,
    per_class_report,
    summarise,
    wilson_interval,
)
from perception.predictions import c1_frame


def test_balanced_accuracy_is_mean_recall():
    y_true = [0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 1, 0]
    # recall class 0 = 1/2, class 1 = 3/4
    assert balanced_accuracy(y_true, y_pred, ["a", "b"]) == pytest.approx(0.625)


def test_always_majority_class_scores_low():
    y_true = [5] * 67 + [4] * 11 + [2] * 11 + [1] * 5 + [0] * 3 + [6] * 2 + [3] * 1
    y_pred = [5] * len(y_true)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(1 / 7)


def test_confusion_matrix_rows_are_true_class():
    cm = confusion_matrix([4, 4, 5], [5, 4, 5])
    assert cm[4, 5] == 1 and cm[4, 4] == 1 and cm[5, 5] == 1


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(25, 45)  # summer pilot: mel recall 0.556
    assert lo == pytest.approx(0.412, abs=0.002)
    assert hi == pytest.approx(0.691, abs=0.002)
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


def test_per_class_report_handles_missing_class():
    rows = per_class_report([0, 0], [0, 1])
    assert rows[0]["n"] == 2 and rows[0]["recall"] == 0.5
    assert rows[3]["n"] == 0 and math.isnan(rows[3]["recall"])


def test_entropy_bounds():
    certain = np.eye(7)[[0]]
    uniform = np.full((1, 7), 1 / 7)
    assert normalised_entropy(certain)[0] == pytest.approx(0, abs=1e-6)
    assert normalised_entropy(uniform)[0] == pytest.approx(1)


def test_summarise_reports_melanoma():
    probs = np.eye(7)[[4, 4, 5, 5]]
    s = summarise([4, 5, 5, 5], probs)
    assert s["mel_recall"] == 1.0
    assert s["n"] == 4


def test_c1_frame_columns_and_malignant_sum():
    meta = pd.DataFrame({"image_id": ["ISIC_1"], "dx": ["mel"], "age": [60.0], "sex": ["male"],
                         "localization": ["back"]})
    probs = np.array([[0.1, 0.2, 0.1, 0.0, 0.3, 0.3, 0.0]])
    df = c1_frame(meta, probs, "cnn-test")
    assert [f"p_{c}" for c in CLASSES] == [c for c in df.columns if c.startswith("p_") and c != "p_malignant"]
    assert df.loc[0, "p_malignant"] == pytest.approx(0.6)
    assert 0 < df.loc[0, "entropy"] < 1


def test_c1_frame_rejects_bad_probs():
    meta = pd.DataFrame({"image_id": ["a"], "dx": ["nv"], "age": [1], "sex": ["f"], "localization": ["x"]})
    with pytest.raises(ValueError):
        c1_frame(meta, np.array([[0.5] * 7]), "v")
