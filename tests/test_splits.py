"""Ground rule 1: no lesion may appear in more than one split.

Skipped until splits/splits.csv is committed in W2 (DERM-10).
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
