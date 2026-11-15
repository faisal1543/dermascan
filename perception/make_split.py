"""DERM-10: create the frozen lesion-grouped train/val/test split (Ground rule 1).

Usage:  python -m perception.make_split --metadata data/HAM10000_metadata.csv
Run ONCE. The output splits/splits.csv is committed and never regenerated.

Method: StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42) on lesion_id,
stratified by dx. Folds 0-2 -> test (15%), folds 3-5 -> val (15%), folds 6-19 -> train (70%).
All images of one lesion stay in one split.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 20
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}


def make_split(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta[["image_id", "lesion_id", "dx"]].sort_values("image_id").reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["split"] = "train"
    for fold, (_, idx) in enumerate(sgkf.split(df, df.dx, groups=df.lesion_id)):
        if fold in TEST_FOLDS:
            df.loc[idx, "split"] = "test"
        elif fold in VAL_FOLDS:
            df.loc[idx, "split"] = "val"
    return df


def main(metadata, out):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists. The split is frozen; never regenerate it.")
    df = make_split(pd.read_csv(metadata))
    assert df.groupby("lesion_id").split.nunique().max() == 1, "lesion leakage"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.split.value_counts(normalize=True).round(3).to_string())
    print(pd.crosstab(df.dx, df.split).to_string())
    print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    p.add_argument("--out", default="splits/splits.csv")
    a = p.parse_args()
    main(a.metadata, a.out)
