# Interface contracts v1 (DERM-11)

**Status: proposed for agreement at the W2 meeting** (Ground rule 5). After agreement, any change
needs all three members to agree. After 21 Dec, changes also go in `docs/CHANGELOG.md` (Ground rule 11).
Architecture and data flow: `docs/architecture.md`. Changes from the W1 draft are listed at the end.

| Contract | Between | Owner | Consumer |
|---|---|---|---|
| C1 perception output | CNN module → reasoning layer, API | Alfaisal | Abdelrahman, Maged |
| C2 reasoning output | Fuzzy layer → API, review screen | Abdelrahman | Maged |
| C3 API endpoints | Backend ↔ review screen | Maged | Maged (screen), demo |
| C4 case-log table | Backend → SQLite | Maged | Audit, evaluation |

Class order is fixed everywhere: `akiec, bcc, bkl, df, mel, nv, vasc` (Ground rule 4).

## C1: Perception output (CNN → reasoning)

Live form (one image, inside the API):

```json
{
  "image_id": "sha256:3f5a…",
  "probs": {"akiec": 0.02, "bcc": 0.05, "bkl": 0.10, "df": 0.01, "mel": 0.31, "nv": 0.50, "vasc": 0.01},
  "p_malignant": 0.38,
  "entropy": 0.61,
  "in_distribution": true,
  "ood_score": 12.4,
  "heatmap_png_b64": "iVBORw0…",
  "model_version": "cnn-resnet18-v1"
}
```

| Field | Type | Rule |
|---|---|---|
| `probs` | 7 floats | Softmax after temperature scaling (O5); sum to 1 ± 0.001 |
| `p_malignant` | float 0–1 | `mel + bcc + akiec` |
| `entropy` | float 0–1 | Shannon entropy ÷ ln 7 (0 = certain, 1 = uniform) |
| `in_distribution` | bool | `false` when the Mahalanobis test on CNN features fails (proposal §3). The reasoning layer then returns `cannot_assess` |
| `ood_score` | float | Mahalanobis distance; threshold set on validation images plus non-dermoscopic photos |
| `heatmap_png_b64` | string | Grad-CAM for the top class, PNG, same aspect as the input |
| `model_version` | string | Matches the GitHub Release holding the weights |

Offline form (CSV, one row per validation or test image, for tuning the reasoning layer and for evaluation):

```
image_id,lesion_id,dx,prob_akiec,prob_bcc,prob_bkl,prob_df,prob_mel,prob_nv,prob_vasc,p_malignant,entropy,model_version
```

`lesion_id` and `dx` exist only in the offline CSV (for grouped confidence intervals and evaluation),
never in the live API. Files: `results/val_predictions/<run>.csv`. The test file is written once, in DERM-32.

## C2: Reasoning output (fuzzy layer → API)

Produced by `reasoning.fuzzy.TriageEngine.assess(c1, age, site)` (DERM-12). Example: fake case 6 in
`results/reasoning/fake_cases_c2.json` (a third fired rule, F1, is left out here for space).

```json
{
  "urgency": "urgent",
  "score": 65.0,
  "inputs": {"p_malignant": 0.4, "p_mel": 0.35, "entropy": 0.6624, "age": 72.0, "site": "lower extremity"},
  "fired_rules": [
    {"id": "F3", "kind": "floor", "text": "IF melanoma is medium AND age is older THEN urgency is at least urgent",
     "explanation": "Some melanoma evidence in an older patient: suspected-cancer referral.", "strength": 1.0, "applied": true},
    {"id": "R2", "kind": "score", "text": "IF melanoma is medium THEN routine",
     "explanation": "Some melanoma evidence: specialist review recommended.", "strength": 1.0}
  ],
  "rules_version": "fuzzy-v0.1-draft"
}
```

| Field | Type | Rule |
|---|---|---|
| `urgency` | enum | `urgent`, `routine`, `monitor` or `cannot_assess` |
| `score` | float 0–100 or `null` | `null` only for `cannot_assess`; always inside the band of `urgency` (monitor < 35 ≤ routine < 65 ≤ urgent) |
| `inputs` | object | Crisp inputs actually used; unknown age is `null`, unknown site is `"unknown"` |
| `fired_rules` | list | Every rule with strength > 0, strongest first. `applied` (floor rules only) is true when strength ≥ 0.5 |
| `rules_version` | string | `version` field of the rule file in `reasoning/rules/` |

The screen shows `explanation`, not `text`. `text` is for the audit log and the report.

## C3: API endpoints (backend ↔ review screen)

| Method | Path | Request | Success |
|---|---|---|---|
| GET | `/health` | none | `200 {"status": "ok", "model_version": "...", "rules_version": "..."}` |
| POST | `/api/assess` | multipart: `image` (JPEG/PNG, ≤ 10 MB), `age` (integer 0–120, optional), `site` (see below, optional), `clinician` (string, required) | `200 {"case_id", "c1", "c2", "notice": "Decision support only, not a diagnosis."}` |
| POST | `/api/cases/{case_id}/decision` | JSON `{"final_decision", "override_reason", "clinician"}` | `200` full case record (C4 row) |
| GET | `/api/cases/{case_id}` | none | `200` full case record |

- `site`: one of the 15 HAM10000 values: `abdomen, acral, back, chest, ear, face, foot, genital, hand,
  lower extremity, neck, scalp, trunk, upper extremity, unknown`. A missing value means `unknown`.
- `final_decision`: `urgent`, `routine` or `monitor`. `override_reason` is required (non-empty) when
  `final_decision` differs from `c2.urgency`, and always required when `c2.urgency` is `cannot_assess` (O3).
- `clinician`: the reviewer ID picked at session start (no login; a stated limitation).
- A failed image check skips the CNN and returns `200` with `c2.urgency = "cannot_assess"`, so every
  path still produces a case to confirm (proposal Figure 1). Malformed requests return errors.

Errors always use one shape: `{"error": "<code>", "message": "<human-readable>"}`.

| HTTP | `error` | When |
|---|---|---|
| 400 | `invalid_image` | Not a decodable JPEG or PNG |
| 413 | `image_too_large` | Over 10 MB |
| 422 | `invalid_age` / `invalid_site` | Out of range or not in the list |
| 422 | `override_reason_required` | Override or `cannot_assess` with no reason |
| 404 | `case_not_found` | Unknown `case_id` |
| 409 | `decision_already_recorded` | A case is decided once; a later change is a new case |

## C4: Case-log table (SQLite)

The image itself is never stored. Only its hash is (proposal §6, Privacy).

| Column | Type | Notes |
|---|---|---|
| `case_id` | TEXT PK | UUID4 |
| `created_at` | TEXT | ISO 8601, UTC |
| `image_sha256` | TEXT | Hash of the uploaded bytes |
| `age` | INTEGER NULL | |
| `site` | TEXT | One of the C3 values |
| `probs_json` | TEXT | C1 `probs` |
| `in_distribution` | INTEGER | 0/1 |
| `recommended_urgency` | TEXT | C2 `urgency` |
| `score` | REAL NULL | C2 `score` |
| `fired_rules_json` | TEXT | C2 `fired_rules` |
| `final_decision` | TEXT NULL | Set by the decision endpoint |
| `override_reason` | TEXT NULL | Required on override |
| `clinician` | TEXT | Reviewer ID |
| `decided_at` | TEXT NULL | ISO 8601, UTC |
| `model_version` | TEXT | |
| `rules_version` | TEXT | |

An override is a row where `final_decision != recommended_urgency`. The log stays on the demo
machine and is deleted at project end (proposal §6).

## Changes from the W1 draft (for the W2 meeting)

1. C1: added `in_distribution` and `ood_score` (the proposal's out-of-distribution check needed a place),
   defined `entropy` as normalised, and defined the offline CSV columns (`lesion_id`, `dx` for grouped CIs).
2. C2: added `cannot_assess`, `p_mel` in `inputs` (see `docs/fuzzy-design.md`, decision 1), and `kind`,
   `explanation`, `applied` on each fired rule.
3. C3: listed the allowed `site` values, made `clinician` a required field on `/api/assess`, required a reason
   for `cannot_assess` decisions, added error codes and HTTP statuses, added the "not a diagnosis" notice.
4. C4: renamed `urgency` → `recommended_urgency` (so overrides are a simple comparison), renamed `timestamp`
   → `created_at`, added `decided_at` and `in_distribution`, and gave types.
