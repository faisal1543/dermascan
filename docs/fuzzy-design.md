# Fuzzy reasoning layer: variables and candidate rules (DERM-9)

Draft for review at the W2 weekly meeting. Owner: Abdelrahman (reasoning lead).
The machine-readable version, with a source on every rule, is `reasoning/rules/rules_v1.json`
(DERM-12, DERM-52). If this page and the rule file disagree, the rule file is what runs.

**Status:** all breakpoints are placeholders. They are tuned on the **validation** split only, in
DERM-27 (Ground rule 3), once the ResNet-18 baseline's validation outputs exist (DERM-14).

## 1. What the layer does

It turns one CNN output (contract C1) plus patient context into an urgency level with reasons
(contract C2):

| Urgency | Meaning (proposal, Section 2) |
|---|---|
| `urgent` | Suspected-cancer referral |
| `routine` | Non-urgent dermatology referral |
| `monitor` | GP follow-up |
| `cannot_assess` | Invalid or out-of-distribution image. Refer, no label shown |

It is a **Mamdani** system (min for AND, max for aggregation, centroid defuzzification) with
two kinds of rule:

- **Score rules** map image evidence to an urgency term. Their defuzzified output is a score
  from 0 to 100, which sets the base category.
- **Floor rules** say "urgency is *at least* X". A floor applies when its firing strength is
  at least 0.5, and it can only raise the category, never lower it. This is how the proposal's
  "IF uncertainty is high THEN at least routine" is implemented, and it is what guarantees
  that patient context and missing values never lower urgency (Section 5).

Why floors and not more Mamdani rules: with centroid defuzzification, adding a rule that fires
"urgent" weakly can still pull the score *down* when the other rules already point higher. A floor
cannot do that, so the safety rules stay safe whatever the breakpoints end up being.

## 2. Input variables

| Variable | From C1 | Universe | Terms (placeholder breakpoints) |
|---|---|---|---|
| Melanoma evidence | `prob_mel` | 0–1 | low trap(0, 0, 0.10, 0.25) · medium tri(0.15, 0.35, 0.55) · high trap(0.40, 0.60, 1, 1) |
| Malignancy evidence | `p_malignant` = mel + bcc + akiec | 0–1 | low trap(0, 0, 0.15, 0.35) · medium tri(0.20, 0.45, 0.70) · high trap(0.55, 0.80, 1, 1) |
| Uncertainty | `entropy` (normalised, 0 = certain, 1 = uniform) | 0–1 | low trap(0, 0, 0.25, 0.45) · medium tri(0.30, 0.50, 0.70) · high trap(0.55, 0.75, 1, 1) |
| Age | request field | 0–120 years | young trap(0, 0, 30, 45) · middle tri(35, 50, 65) · older trap(55, 70, 120, 120) · **unknown** |
| Body site | request field | categorical | **head_neck** = face, ear, scalp, neck · **other** = the other 10 HAM10000 sites · **unknown** |

`trap(a, b, c, d)` is a trapezoid (0 before a, rising to 1 at b, flat to c, back to 0 at d);
`tri(a, b, c)` is a triangle. These are the same shapes as `skfuzzy.trapmf` / `skfuzzy.trimf`.

**Why melanoma evidence is a separate input.** The proposal's single malignancy input sums
mel + bcc + akiec. NICE NG12 treats these cancers differently: suspected melanoma gets a
suspected-cancer referral (1.7.2), suspected basal cell carcinoma a non-urgent one (1.7.5).
A single sum cannot tell a 0.6 melanoma from a 0.6 BCC, so the layer also reads `prob_mel`,
which is already in C1. **Decision needed at W2:** agree to the extra input. It changes no
contract, but it does refine what the proposal describes.

**Why only "older" age is used in rules.** Malignant classes are older in HAM10000 (median age:
akiec 70, bcc 70, mel 60, against nv 45; `results/eda/summary.md`). Young and middle are
defined for plots and completeness, but no rule lowers urgency for a younger patient.

## 3. Output variable

Urgency score, 0–100: monitor trap(0, 0, 20, 40) · routine tri(30, 50, 70) · urgent trap(60, 80, 100, 100).
Category from the score: **monitor < 35 ≤ routine < 65 ≤ urgent** (placeholder cut-offs).
When a floor raises the category, the score is lifted to that category's lower bound (35 or 65),
so the number shown on screen never contradicts the label.

## 4. Candidate rules (10)

| ID | Kind | Rule | Source |
|---|---|---|---|
| G1 | gate | IF the image failed validation or the out-of-distribution check THEN `cannot_assess` (no rules run) | Proposal §3 (OOD check, Lee et al., 2018); objective O3 |
| R1 | score | IF melanoma is high THEN urgent | NICE NG12 1.7.2: refer on a suspected cancer pathway if dermoscopy suggests melanoma |
| R2 | score | IF melanoma is medium THEN routine | NG12 1.7.2 applied to borderline evidence. Escalated by F3. **Tuning knob for O2** |
| R3 | score | IF malignancy is high AND melanoma is low THEN routine | NG12 1.7.5 (BCC: non-urgent referral) and 1.7.4 (SCC: consider suspected cancer pathway, via F4/F1) |
| R4 | score | IF malignancy is medium AND melanoma is low THEN routine | NG12 1.7.5: a lesion that "raises the suspicion" of BCC is referred, not watched |
| R5 | score | IF malignancy is low THEN monitor | Proposal §2 (monitor = GP follow-up); benign classes |
| F1 | floor | IF uncertainty is high THEN at least routine | Proposal §3; Tschandl et al. (2020): unreliable AI output misleads clinicians, so uncertain cases go to a specialist |
| F2 | floor | IF melanoma is high THEN at least urgent | NG12 1.7.2; objective O2 (≥ 85% of melanomas urgent). Nothing may average a melanoma down |
| F3 | floor | IF melanoma is medium AND age is older THEN at least urgent | NG12 1.7.2 plus age as a risk factor (HAM10000 EDA: median melanoma age 60 vs nevus 45). To confirm on the train split |
| F4 | floor | IF malignancy is high AND site is head_neck THEN at least urgent | NG12 1.7.6: suspected-cancer referral for BCC only when delay may have a significant impact, for example because of lesion site |

Each rule also carries a one-line, plain-language explanation for the review screen (in the rule
file). The final wording is part of DERM-30.

## 5. Missing values: "unknown never lowers urgency"

HAM10000 has 57 images with no age and 234 with body site "unknown" (proposal §3).

- `unknown` has membership 0 in every age or site term, so it fires no context rule.
- Context (age, site) appears **only in floor rules**, and floors only raise.

So for any image, the urgency with an unknown age or site equals the urgency from the image
evidence alone, and is never lower than it. The test `tests/test_reasoning.py` checks this over a
grid of inputs, including that a known non-escalating value (age 30, site "back") gives the same
result as unknown.

## 6. How it will be checked

- **DERM-12:** 10 hand-made cases run end to end (C1 in, C2 out); a unit test fixes the expected urgency
  for each.
- **DERM-15:** on the baseline's validation CSV, compare with (a) the argmax label and (b) a single
  P(melanoma) threshold. Measures: melanomas urgent (target ≥ 85%), malignant lesions urgent or routine
  (target ≥ 95%), benign lesions sent urgent (lower than both baselines at equal melanoma recall), objective O2.
- **DERM-27:** tune breakpoints on validation only; save membership-function plots for the design report.

## 7. Open questions for the W2 meeting

1. Agree to melanoma evidence as a separate input (Section 2).
2. Which sites count as "delay is risky" for BCC (F4)? The draft uses face, ear, scalp, neck. Should acral or genital be added?
3. akiec in HAM10000 mixes actinic keratosis with intraepidermal carcinoma (Bowen's disease).
   Should a confident akiec result be routine (R3, current draft) or urgent (NG12 1.7.4, SCC)?
4. Timescale wording in the design report: the proposal says "seen within 2 weeks". NG12 was
   last updated on 15 April 2026 and now says "suspected cancer pathway referral" without a 2-week
   figure, and NICE quality standard QS130 uses a 28-day faster-diagnosis standard. Suggest the
   design report defines `urgent` as "suspected cancer pathway referral (NG12)" and drops the 2 weeks.

## References

- NICE (2015, last updated 15 April 2026). *Suspected cancer: recognition and referral* (NG12),
  recommendations 1.7.1–1.7.7. https://www.nice.org.uk/guidance/ng12
- NICE (2016, updated 2024). *Skin cancer* quality standard (QS130), statement 2.
  https://www.nice.org.uk/guidance/qs130
- Tschandl, P., et al. (2020). Human–computer collaboration for skin cancer recognition. *Nature Medicine*, 26, 1229–1234.
- Lee, K., et al. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
- Chatterjee, S., et al. (2021). Dermatological expert system implementing the ABCD rule of dermoscopy. *Expert Systems with Applications*, 167, 114204.
  ABCD-style expert systems use the same pattern: graded features, combined by explicit rules a clinician can read.
