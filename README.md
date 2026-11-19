# DermaScan — Human-in-the-Loop Cognitive Triage Assistant

SCSE3103 Cognitive Computing · Group Bravo · Semester 1, 2026/2027

DermaScan turns a dermoscopic image into a triage recommendation (urgent / routine / monitor)
with a visible explanation. A CNN perceives the lesion, a fuzzy-logic layer reasons over the
CNN's malignancy evidence, its uncertainty and patient risk factors, and a clinician confirms
or overrides every decision, which is logged for audit.

> **Not a medical device.** Educational prototype only. Not for diagnosis or patient self-screening.

## Team

| Member | Matric | Lead role |
|---|---|---|
| Alfaisal Khalid Abdulaziz Ahmed | A24CS0004 | Project manager; perception lead |
| Abdelrahman Zakaria Mohamed | A24CS4001 | Architecture and reasoning lead |
| Maged Abdulsalam Al-Shami | A24CS4020 | Application and quality lead |

## Milestones

| Deliverable | Due | Tag |
|---|---|---|
| Proposal | 16 Nov 2026 | `v0.1-proposal` |
| Design | 21 Dec 2026 | `v0.2-design` |
| Prototype + final report | 21 Jan 2027 | `v1.0-final` |
| Presentation + demo | 21 Jan 2027 | — |

## Data

HAM10000 (Tschandl, Rosendahl & Kittler, 2018), CC BY-NC 4.0:
https://doi.org/10.7910/DVN/DBW86T — download it locally into `data/`. It is not committed.

## Project tracking

Jira: https://graduate-team-az9eb22g.atlassian.net/jira/software/projects/DERM/boards/34
(private project DERM). Put the Jira key in every branch, commit and pull request, e.g. `DERM-12`.

## Repository layout

```
perception/   dataset, CNN models, training, evaluation      (Alfaisal)
reasoning/    fuzzy variables, rules, explanations           (Abdelrahman)
app/          FastAPI backend, review screen, case log       (Maged)
tests/        pytest: split checks, rules, API, validation
splits/       frozen lesion-grouped split (W2)
experiments/  log.csv — every training run
results/      final tables and figures (W8)
notebooks/    exported Kaggle notebooks
docs/         contracts, meetings/, CHANGELOG.md, ai-use-log.md
data/         NOT committed — see data/README.md
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest -q
```

## Run the API (DERM-13)

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/ for the review screen (DERM-16), or http://127.0.0.1:8000/docs
to try the API directly. Until the real CNN is
wired in (DERM-28), perception is **mocked**: the image's hash picks one of the 10 made-up cases in
`reasoning/fake_cases.json`, and the real fuzzy layer turns it into a recommendation. The outputs
are not model predictions.

Every assessment and decision is written to a SQLite case log, `caselog.db` in the folder you run
from (set `DERMASCAN_DB` to use another path). It stores the image hash, never the image. Check or
delete it (proposal Section 6: deleted at project end):

```
python -m app.caselog --db caselog.db             # number of cases
python -m app.caselog --db caselog.db --delete --yes
```

See `CONTRIBUTING.md` for the team workflow and `CONTRIBUTIONS.md` for the contribution log.
