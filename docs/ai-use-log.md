# Generative AI use log

Required by the project brief (Section 8). Add a row whenever an AI tool is used.

| Date | Member | Tool | Purpose | What was generated or changed | How it was verified |
|---|---|---|---|---|---|
| 2026-11-14 | Alfaisal | Claude (Anthropic) | Proposal structure and draft, related-work search, team playbook, repo starter files, Jira setup | First drafts, edited by the team | Citations checked against sources; plan reviewed by all members |
| 2026-11-16 | Alfaisal | Claude (Anthropic) | DERM-7 compute check notebook, DERM-8 EDA script, DERM-10 split script and tests | Code drafts | Ran on Kaggle/locally; outputs checked against metadata; split reproducibility and leakage tests pass |
| 2026-11-19 | Abdelrahman | Claude (Anthropic) | DERM-9 fuzzy variables and candidate rules; DERM-12 fuzzy prototype and DERM-52 rule file; DERM-11 contracts v1 and architecture sketch; DERM-46 review notes for PR #3 | Drafts of `docs/fuzzy-design.md`, `reasoning/`, `docs/contracts.md`, `docs/architecture.md`, tests | NICE NG12 rec. numbers checked against the April 2026 PDF; 10 fake cases and safety tests pass locally; scikit-fuzzy cross-check runs in CI; rules to be reviewed at the W2 meeting |
| 2026-11-19 | Maged | Claude (Anthropic) | DERM-13 API skeleton and DERM-16 case log and review screen | Code, tests and README drafts for `app/` and `tests/` | Core tests run locally; API tests run in CI; endpoints and error codes checked against `docs/contracts.md` (C3, C4) |
