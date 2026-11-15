# Splits

`splits.csv` (columns: `image_id, lesion_id, dx, split`) is created ONCE in W2 (DERM-10):
70/15/15, grouped by `lesion_id`, stratified by class, seed 42.

**Never regenerate it.** `tests/test_splits.py` checks that no lesion spans two splits.
The test split stays sealed until W8 and is run once.
