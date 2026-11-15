# Splits — FROZEN on 16 Nov 2026 (DERM-10)

`splits.csv` (columns: `image_id, lesion_id, dx, split`) was created once by
`python -m perception.make_split` — StratifiedGroupKFold(20 folds, shuffle, seed 42) on
`lesion_id`, stratified by `dx`; 3 folds test, 3 folds val, 14 folds train.

| Split | Images | Lesions | Share |
|---|---:|---:|---:|
| train | 7,009 | 5,230 | 70% |
| val | 1,503 | 1,122 | 15% |
| test | 1,503 | 1,118 | 15% |

Class shares are within 0.1 percentage points of each other across the three splits.
SHA-256: `98485423d758cae1ed8b7adf64c3e92d31a6afd4d48de7a26e89355238e34b38`

**Never regenerate or edit this file.** `tests/test_splits.py` checks no lesion spans two
splits, every class is in every split, and the file hash is unchanged.
The test split stays sealed until W8 and is run once (DERM-32).
