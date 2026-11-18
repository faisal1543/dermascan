# Ground rules

Decided early, never broken later. Each rule is locked in the week shown. Breaking one later
means redoing every result that came after it. Jira items refer to these by number
("Ground rule 1").

| # | Rule | Locked in | What goes wrong if we break it |
|---|---|---|---|
| 1 | Split by lesion ID, never by image: 70/15/15, stratified by class, seed 42. `splits/splits.csv` is committed once and never regenerated (`tests/test_splits.py` checks its hash). | W2 | The same lesion lands in training and test; scores look great and are fake. |
| 2 | The test set is sealed until W8. It is run once, by the perception lead, and the date is logged in `CONTRIBUTIONS.md`. All tuning uses validation. | W2 | Any peek at test turns it into a second validation set and the final numbers are no longer honest. |
| 3 | Statistics come from training data only. Class weights from train; fuzzy membership functions and thresholds tuned on validation. | W2 | Information leaks from evaluation data into the model. |
| 4 | Class order is fixed: `akiec, bcc, bkl, df, mel, nv, vasc`. | W2 | A saved model's outputs get read as the wrong classes. |
| 5 | Interface contracts (`docs/contracts.md`) are agreed before parallel work starts. Changes need all three members to agree. | W2 | Components built separately do not fit together at integration. |
| 6 | No accuracy-only reporting. Always balanced accuracy and per-class recall with sample size and 95% CI. | W3 | Results mislead on unbalanced data. |
| 7 | Every experiment is logged in `experiments/log.csv` (date, who, machine, config, validation metrics). No silent reruns. | W3 | We cannot explain or reproduce the numbers in the report. |
| 8 | Augmentation on training data only, and no colour jitter: lesion colour is diagnostic. | W3 | Evaluation becomes random, or the model loses a melanoma cue. |
| 9 | No data or model weights in git. Data stays on Kaggle or local disk; weights go in GitHub Releases. | W1 | The repository bloats or breaks the dataset licence. |
| 10 | No direct commits to `main`. Branch, pull request, an approving review from a member other than the author, tests pass. No admin bypass. | W1 | Untraceable contributions; broken builds before the demo. |
| 11 | The design report is the contract. Any change after 21 Dec goes in `docs/CHANGELOG.md` with the reason. | W6 | The final report contradicts the approved design without explanation. |
| 12 | Model freeze 3 Jan (end of W7); feature freeze Mon 11 Jan (W9). After that, bug fixes only. | W7 / W9 | Late changes invalidate the evaluation and break the demo. |
| 13 | Logs are updated every week: `CONTRIBUTIONS.md` and `docs/ai-use-log.md` (tool, purpose, who checked it). | W1 | We cannot evidence individual contributions or AI use at the end. |

## Compute

- **Training:** GPU, on Kaggle Notebooks (free weekly quota) with Google Colab or a member's local
  NVIDIA GPU (RTX 5050 laptop; needs PyTorch >= 2.7 with CUDA 12.8) as backup. Record the machine in `experiments/log.csv`.
- **Inference and the demo:** CPU. The 3-second latency target (objective O6) is measured on the
  demo laptop's CPU.
