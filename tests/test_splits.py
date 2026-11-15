"""Ground rule 1: no lesion may appear in more than one split.

Activated once splits/splits.csv was committed (DERM-10).
"""
from pathlib import Path

import pytest

SPLITS = Path(__file__).resolve().parents[1] / "splits" / "splits.csv"


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.csv not committed yet (W2)")
def test_no_lesion_spans_two_splits():
    import pandas as pd

    df = pd.read_csv(SPLITS)
    per_lesion = df.groupby("lesion_id")["split"].nunique()
    leaking = per_lesion[per_lesion > 1]
    assert leaking.empty, f"{len(leaking)} lesions appear in more than one split"


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.csv not committed yet (W2)")
def test_split_proportions_roughly_70_15_15():
    import pandas as pd

    share = pd.read_csv(SPLITS)["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.05
    assert abs(share["val"] - 0.15) < 0.05
    assert abs(share["test"] - 0.15) < 0.05


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.csv not committed yet (W2)")
def test_split_covers_full_dataset_once():
    import pandas as pd

    df = pd.read_csv(SPLITS)
    assert len(df) == 10015
    assert df["image_id"].is_unique
    assert set(df["split"]) == {"train", "val", "test"}


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.csv not committed yet (W2)")
def test_every_class_in_every_split():
    import pandas as pd

    df = pd.read_csv(SPLITS)
    classes = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    for name, part in df.groupby("split"):
        assert set(part["dx"]) == classes, f"{name} is missing a class"


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.csv not committed yet (W2)")
def test_split_file_unchanged():
    """Ground rule 1: the frozen split must never change."""
    import hashlib

    digest = hashlib.sha256(SPLITS.read_bytes()).hexdigest()
    assert digest == "98485423d758cae1ed8b7adf64c3e92d31a6afd4d48de7a26e89355238e34b38"
