# Architecture sketch v1 (DERM-11)

Owner: Abdelrahman. For agreement at the W2 meeting together with `docs/contracts.md`.
The final diagrams for the design report (DERM-20) are drawn from this sketch.

DermaScan is a perception → reasoning → action pipeline (proposal §3). Three parts are built
in parallel against the contracts and joined at the proof of concept (DERM-17, W3).

## 1. Components and owners

```mermaid
flowchart LR
    subgraph Browser["Review screen (HTML/JS) · Maged"]
        UI["Upload image, age, body site<br/>Show urgency, reasons, heatmap<br/>Confirm or override + reason"]
    end
    subgraph API["FastAPI backend (CPU) · Maged"]
        V["Input checks<br/>type, size, decode"]
        ORCH["/api/assess orchestration"]
        LOG[("SQLite case log<br/>C4")]
    end
    subgraph P["Perception · Alfaisal"]
        CNN["CNN + temperature scaling<br/>Grad-CAM"]
        OOD["Mahalanobis OOD check<br/>on CNN features"]
    end
    subgraph R["Reasoning · Abdelrahman"]
        FZ["Fuzzy triage engine<br/>reasoning/fuzzy.py"]
        RF[["Versioned rule file<br/>reasoning/rules/rules_v1.json"]]
    end
    UI -- "C3 POST /api/assess" --> V
    V -- valid image --> ORCH
    V -- "invalid: cannot_assess" --> LOG
    ORCH --> CNN
    CNN --> OOD
    OOD -- "C1" --> FZ
    RF --> FZ
    FZ -- "C2" --> ORCH
    ORCH -- "C1 + C2" --> UI
    ORCH --> LOG
    UI -- "C3 POST /decision" --> LOG
```

| Component | Responsibility | Code | Contract out |
|---|---|---|---|
| Review screen | Collect inputs, show the recommendation and reasons, record the decision | `app/static/` | C3 requests |
| Backend | Validate, orchestrate, log; never stores the image | `app/` | C3 responses, C4 rows |
| Perception | Class probabilities, uncertainty, OOD flag, heatmap | `perception/` | C1 |
| Reasoning | Urgency, score and fired rules from C1 + context | `reasoning/` | C2 |
| Case log | Audit trail of every recommendation and decision | SQLite file | C4 |

## 2. One assessment, step by step

```mermaid
sequenceDiagram
    actor GP as Clinician
    participant S as Review screen
    participant A as Backend
    participant C as CNN + OOD
    participant F as Fuzzy engine
    participant D as Case log
    GP->>S: image, age, site
    S->>A: POST /api/assess
    alt file is not a decodable JPEG/PNG, or over 10 MB
        A-->>S: 400 / 413 error
    else image decodes
        A->>C: image tensor
        C-->>A: C1 (probs, entropy, in_distribution, heatmap)
        A->>F: C1, age, site
        F-->>A: C2 (urgency, score, fired rules)
        A->>D: insert case (hash, C1, C2)
        A-->>S: case_id, C1, C2, "not a diagnosis"
    end
    GP->>S: confirm, or override with reason
    S->>A: POST /api/cases/{id}/decision
    A->>D: final decision, reason, clinician
```

If the OOD check fails, the fuzzy engine's gate G1 returns `cannot_assess`. The case is still logged,
and the clinician still records a decision, with a reason (O3).

## 3. Offline (training and tuning) vs online (demo)

```mermaid
flowchart LR
    subgraph Offline["Offline: GPU (Daeta's laptop, RTX 5050; Kaggle as backup)"]
        H[("HAM10000<br/>not in git")] --> SP["Frozen split<br/>splits/splits.csv"]
        SP --> TR["perception: train<br/>on train split"]
        TR --> VC["Validation C1 CSV<br/>results/val_predictions/"]
        VC --> TU["reasoning: tune breakpoints<br/>on validation only"]
        TR --> W["Weights → GitHub Release"]
        TU --> RF2["rules_vN.json in git"]
    end
    subgraph Online["Online: demo laptop CPU, under 3 s per case (O6)"]
        APP["FastAPI + screen + SQLite"]
    end
    W --> APP
    RF2 --> APP
```

The test split is never used for training or tuning. It is read once, in W8 (DERM-32 and DERM-33),
after the model and rules freezes (Ground rules 2 and 12).

## 4. Design decisions recorded so far

| Decision | Why | Alternative rejected |
|---|---|---|
| Fuzzy layer after the CNN, not before | Turns uncertain perception into a readable, editable triage decision (proposal §4 gap) | Learned meta-classifier: clinicians cannot read or edit it |
| Floor rules for safety ("at least routine") | Centroid averaging cannot lower a safety rule's effect; context can only raise urgency | All-Mamdani rules: adding a weak "urgent" rule can lower the score |
| Rules in a versioned JSON file with a source per rule | Provenance for every rule (brief; proposal §3); `rules_version` in every log row | Rules hard-coded in Python |
| OOD check before reasoning, separate from entropy | Entropy alone does not catch a non-skin photo (proposal §3) | Entropy threshold only |
| Image hashed, never stored | Privacy (proposal §6) | Storing uploads for later review |
| CPU inference, GPU training | Demo runs on a laptop; training needs a GPU | GPU at demo time |

## 5. Open points for W2

1. Agree the C1–C4 changes listed at the end of `docs/contracts.md`.
2. Where the OOD threshold lives: in the model release (with the weights) or in a config file in the repo.
3. Whether `/api/assess` calls the CNN in-process (simplest; current sketch) or as a separate worker.
