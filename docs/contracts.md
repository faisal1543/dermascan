# Interface contracts (agree in W2 — DERM-11)

Changes need all three members to agree. Draft fields below; confirm at the W2 meeting.

## C1 — Perception output (CNN → reasoning)
Also saved as one CSV row per validation/test image so the reasoning layer can be tuned offline.

```json
{
  "image_id": "ISIC_0024306",
  "probs": {"akiec": 0.02, "bcc": 0.05, "bkl": 0.10, "df": 0.01, "mel": 0.31, "nv": 0.50, "vasc": 0.01},
  "p_malignant": 0.38,
  "entropy": 0.61,
  "heatmap_png_b64": "...",
  "model_version": "cnn-v1.0"
}
```
Class order is fixed: akiec, bcc, bkl, df, mel, nv, vasc. `p_malignant` = mel + bcc + akiec.

## C2 — Reasoning output (fuzzy layer → API)

```json
{
  "urgency": "urgent | routine | monitor | cannot_assess",
  "score": 72.5,
  "inputs": {"p_malignant": 0.38, "entropy": 0.61, "age": 64, "site": "back"},
  "fired_rules": [
    {"id": "R4", "text": "IF uncertainty is high THEN urgency is at least routine", "strength": 0.71}
  ],
  "rules_version": "fuzzy-v1.0"
}
```

## C3 — API endpoints (backend ↔ review screen)

| Method | Path | Sends | Returns |
|---|---|---|---|
| GET | `/health` | — | Status, model and rules versions |
| POST | `/api/assess` | Image (JPEG/PNG, ≤ 10 MB), age, body site | `case_id`, C1, C2; or an error |
| POST | `/api/cases/{case_id}/decision` | Final decision, reason (required if overriding), clinician | Saved case |
| GET | `/api/cases/{case_id}` | — | Full case record |

Errors: `{"error": "invalid_image", "message": "File is not a JPEG or PNG"}`

## C4 — Case log table (SQLite)

`case_id, timestamp, image_sha256, age, site, probs_json, urgency, score, fired_rules_json,
final_decision, override_reason, clinician, model_version, rules_version`

The image itself is not stored.
