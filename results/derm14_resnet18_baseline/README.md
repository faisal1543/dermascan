# DERM-14 — ResNet-18 baseline (validation only)

Kaggle notebook "DERM-14 ResNet-18 baseline", version 1, 19 Nov 2026, GPU T4.
Code: `perception/train.py` at commit 5cf655a. Weights are not in git (Ground rule 9).

| Metric (validation, n = 1,503, best epoch 12 of 18) | Value |
|---|---|
| Balanced accuracy | **0.796** |
| Melanoma recall (argmax, no threshold tuning) | **0.801** (95% CI 0.73–0.85, n = 166) |
| Melanoma precision | 0.433 |

| Class | n | Recall [95% CI] | Precision |
|---|---|---|---|
| akiec | 50 | 0.580 [0.44–0.71] | 0.617 |
| bcc | 77 | 0.883 [0.79–0.94] | 0.701 |
| bkl | 165 | 0.521 [0.45–0.60] | 0.768 |
| df | 18 | 1.000 [0.82–1.00] | 0.529 |
| mel | 166 | 0.801 [0.73–0.85] | 0.433 |
| nv | 1006 | 0.832 [0.81–0.85] | 0.950 |
| vasc | 21 | 0.952 [0.77–0.99] | 0.800 |

Files:
- `val_predictions.csv` — contract C1, one row per validation image (probabilities, `p_malignant`,
  normalised entropy, age, sex, site, true `dx`). Input for the reasoning layer (DERM-15).
- `metrics_val.json` — the numbers above plus the confusion matrix.
- `history.csv` — per-epoch training loss and validation metrics.

Notes: validation peaked at epoch 12 and then plateaued while training loss kept falling
(mild overfitting). The test set has not been read (Ground rule 2).
