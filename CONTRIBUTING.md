# Team workflow

1. **Every task is a Jira work item in project DERM** (owner, label, epic per phase). Move your own
   items To Do → In Progress → Done; the lecturer reads this history.
2. **One branch per issue:** `feature/DERM-12-short-name` or `fix/DERM-12-short-name`.
   Never commit directly to `main` (it is protected).
3. **Commit messages reference the issue:** `DERM-12: Add fuzzy membership functions`.
4. **Pull requests** need one approving review from another member, and tests must pass.
   The reviewer checks the code runs, has tests where relevant, and has no secrets or data files.
5. **Releases:** tag each deliverable — `v0.1-proposal`, `v0.2-design`, `v1.0-final`.
6. **Meetings:** weekly on [day, time] via [platform]; written stand-up in the group chat every
   Monday and Thursday. Notes go in `docs/meetings/YYYY-MM-DD.md`.
7. **Joint responsibility:** everyone must be able to explain and demo every component.

## Data and evaluation rules

- Split by `lesion_id`, never by image. A check must confirm no lesion spans two splits.
- Class weights, normalisation stats and fuzzy membership functions come from train/validation only.
- The test set is read once, at the end. Record the date in the contribution log when it happens.
