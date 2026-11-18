"""DERM-12 / DERM-52: fuzzy triage layer (rule file, inference, safety properties)."""
from __future__ import annotations

import copy
import itertools
import json

import numpy as np
import pytest

from reasoning.fuzzy import CATEGORY_ORDER, RULES_V1, TriageEngine, membership, validate_spec
from reasoning.run_fake_cases import run

ENGINE = TriageEngine()
RANK = {c: i for i, c in enumerate(CATEGORY_ORDER)}


def c1(mel: float, other_malignant: float = 0.0, entropy: float | None = None) -> dict:
    """C1 in CSV-row form: mel and bcc as given, the rest on nv."""
    probs = {"akiec": 0.0, "bcc": other_malignant, "bkl": 0.0, "df": 0.0, "mel": mel,
             "nv": 1.0 - mel - other_malignant, "vasc": 0.0}
    row = {f"prob_{k}": v for k, v in probs.items()}
    if entropy is not None:
        row["entropy"] = entropy
    return row


# --- the rule file (DERM-52) -------------------------------------------------------------

def test_rule_file_is_valid_and_versioned():
    spec = json.loads(RULES_V1.read_text())
    validate_spec(spec)
    assert spec["version"].startswith("fuzzy-v")
    assert 8 <= len(spec["rules"]) <= 12  # DERM-9: 8-12 candidate rules


def test_every_rule_cites_a_source():
    for rule in ENGINE.rules:
        assert len(rule["source"]) > 10, rule["id"]
        assert rule["explanation"].strip(), rule["id"]


def test_rule_without_source_is_rejected():
    spec = json.loads(RULES_V1.read_text())
    bad = copy.deepcopy(spec)
    bad["rules"][1]["source"] = ""
    with pytest.raises(ValueError, match="source"):
        validate_spec(bad)
    bad = copy.deepcopy(spec)
    bad["rules"][1]["if"] = [["melanoma", "huge"]]
    with pytest.raises(ValueError, match="no term"):
        validate_spec(bad)


def test_age_and_site_only_appear_in_floor_rules():
    """Context may only raise urgency (docs/fuzzy-design.md, Section 5)."""
    for rule in ENGINE.rules:
        if rule["kind"] == "score":
            assert not {v for v, _ in rule["if"]} & {"age", "site"}, rule["id"]


# --- membership functions ----------------------------------------------------------------

@pytest.mark.parametrize("x, expected", [(0.0, 0.0), (0.15, 0.0), (0.25, 0.5), (0.35, 1.0), (0.45, 0.5), (0.6, 0.0)])
def test_triangle(x, expected):
    assert membership("tri", [0.15, 0.35, 0.55], x) == pytest.approx(expected)


@pytest.mark.parametrize("x, expected", [(0.0, 1.0), (0.10, 1.0), (0.175, 0.5), (0.25, 0.0), (1.0, 0.0)])
def test_left_shoulder_trapezoid(x, expected):
    assert membership("trap", [0, 0, 0.10, 0.25], x) == pytest.approx(expected)


def test_terms_cover_every_input_value():
    """No gap in the universe where no term fires (otherwise no rule could fire)."""
    for name in ("melanoma", "malignancy", "uncertainty"):
        terms = ENGINE.spec["inputs"][name]["terms"].values()
        for x in np.linspace(0, 1, 201):
            assert max(membership(t["shape"], t["params"], x) for t in terms) > 0, (name, x)


def test_matches_scikit_fuzzy():
    """Same shapes and defuzzification as scikit-fuzzy (runs in CI, where it is installed)."""
    fuzz = pytest.importorskip("skfuzzy")
    xs = np.linspace(0, 1, 101)
    for var in ENGINE.spec["inputs"].values():
        if var.get("categorical") or var["universe"] != [0, 1]:
            continue
        for t in var["terms"].values():
            ref = (fuzz.trimf if t["shape"] == "tri" else fuzz.trapmf)(xs, t["params"])
            ours = [membership(t["shape"], t["params"], x) for x in xs]
            np.testing.assert_allclose(ours, ref, atol=1e-9)
    for strengths in ({"R1": 1.0}, {"R2": 0.6, "R1": 0.3}, {"R5": 0.9, "R4": 0.2}):
        agg = ENGINE.aggregate(strengths)
        ours = float((ENGINE.xs * agg).sum() / agg.sum())
        assert ours == pytest.approx(fuzz.defuzz(ENGINE.xs, agg, "centroid"), abs=0.2)


# --- DERM-12 done-when: 10 fake cases ----------------------------------------------------

def test_ten_fake_cases_give_expected_urgency_score_and_rules():
    results = run(ENGINE)
    assert len(results) == 10
    for r in results:
        c2 = r["c2"]
        assert c2["urgency"] == r["expected"], r["case"]
        assert c2["fired_rules"], r["case"]
        if c2["urgency"] != "cannot_assess":
            assert 0 <= c2["score"] <= 100


def test_c2_shape():
    c2 = ENGINE.assess({"probs": {"akiec": 0.02, "bcc": 0.05, "bkl": 0.10, "df": 0.01,
                                  "mel": 0.31, "nv": 0.50, "vasc": 0.01}}, age=64, site="back")
    assert set(c2) == {"urgency", "score", "inputs", "fired_rules", "rules_version"}
    assert c2["urgency"] in CATEGORY_ORDER
    assert c2["inputs"]["p_malignant"] == pytest.approx(0.38)
    assert c2["rules_version"] == ENGINE.version
    strengths = [f["strength"] for f in c2["fired_rules"]]
    assert strengths == sorted(strengths, reverse=True)
    for f in c2["fired_rules"]:
        assert {"id", "kind", "text", "explanation", "strength"} <= set(f)


def test_label_and_score_agree():
    """The score shown on screen always falls inside the band of the urgency label."""
    for mel, other, unc, age, site in itertools.product(
            [0, 0.2, 0.35, 0.5, 0.8], [0, 0.3, 0.6], [0.1, 0.5, 0.9], [None, 30, 75], ["back", "face"]):
        if mel + other > 1:
            continue
        c2 = ENGINE.assess(c1(mel, other, unc), age=age, site=site)
        assert ENGINE.category(c2["score"]) == c2["urgency"]


# --- safety properties -------------------------------------------------------------------

GRID = [(m, o, u) for m in np.linspace(0, 1, 11) for o in np.linspace(0, 1, 11)
        for u in np.linspace(0, 1, 6) if m + o <= 1 + 1e-9]


def test_unknown_age_or_site_never_lowers_urgency():
    """Proposal Section 3 and DERM-52: missing age/site maps to a neutral term that can never
    lower urgency. Unknown = image evidence alone = a known non-escalating value."""
    for m, o, u in GRID:
        x = c1(m, o, u)
        unknown = ENGINE.assess(x, age=None, site="unknown")
        neutral = ENGINE.assess(x, age=30, site="back")
        assert unknown["urgency"] == neutral["urgency"] and unknown["score"] == neutral["score"]
        for age, site in itertools.product([None, 20, 50, 80], ["unknown", "back", "face", "acral"]):
            known = ENGINE.assess(x, age=age, site=site)
            assert RANK[known["urgency"]] >= RANK[unknown["urgency"]], (m, o, u, age, site)


def test_partially_missing_context_never_lowers_urgency():
    for m, o, u in GRID[::7]:
        x = c1(m, o, u)
        both = ENGINE.assess(x, age=80, site="face")["urgency"]
        assert RANK[ENGINE.assess(x, age=None, site="face")["urgency"]] <= RANK[both]
        assert RANK[ENGINE.assess(x, age=80, site=None)["urgency"]] <= RANK[both]
        base = ENGINE.assess(x, age=None, site=None)["urgency"]
        assert RANK[ENGINE.assess(x, age=None, site="face")["urgency"]] >= RANK[base]
        assert RANK[ENGINE.assess(x, age=80, site=None)["urgency"]] >= RANK[base]


def test_strong_melanoma_evidence_is_always_urgent():
    """F2 / objective O2: nothing averages a melanoma down."""
    for mel in (0.6, 0.75, 0.9, 1.0):
        for unc in (0.0, 0.5, 1.0):
            for age, site in itertools.product([None, 20, 80], ["unknown", "back", "face"]):
                assert ENGINE.assess(c1(mel, 0, unc), age=age, site=site)["urgency"] == "urgent"


def test_more_melanoma_evidence_never_lowers_urgency():
    for other in (0.0, 0.1, 0.3):
        for unc in (0.1, 0.5, 0.9):
            for age, site in [(None, "unknown"), (30, "back"), (75, "face")]:
                ranks = [RANK[ENGINE.assess(c1(m, other, unc), age=age, site=site)["urgency"]]
                         for m in np.linspace(0, 1 - other, 41)]
                assert ranks == sorted(ranks), (other, unc, age, site)


def test_high_uncertainty_is_never_monitor():
    """F1: an unsure model never sends a case to GP follow-up alone."""
    for m, o, _ in GRID:
        assert ENGINE.assess(c1(m, o, 0.9))["urgency"] != "monitor"


def test_out_of_distribution_and_errors_cannot_be_assessed():
    probs = c1(0.9, 0, 0.1)
    for bad in ({**probs, "in_distribution": False}, {**probs, "error": "invalid_image"}):
        c2 = ENGINE.assess(bad, age=60, site="back")
        assert c2["urgency"] == "cannot_assess" and c2["score"] is None
        assert [f["id"] for f in c2["fired_rules"]] == ["G1"]


# --- input handling ----------------------------------------------------------------------

def test_json_and_csv_forms_agree_and_derived_fields_computed():
    probs = {"akiec": 0.02, "bcc": 0.05, "bkl": 0.10, "df": 0.01, "mel": 0.31, "nv": 0.50, "vasc": 0.01}
    a = ENGINE.assess({"probs": probs}, age=50, site="Back ")
    b = ENGINE.assess({f"prob_{k}": v for k, v in probs.items()}, age=50, site="back")
    assert a == b
    assert 0 < a["inputs"]["entropy"] < 1


def test_bad_inputs_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        ENGINE.assess(c1(0.5) | {"prob_nv": 0.9})
    with pytest.raises(ValueError, match="body site"):
        ENGINE.assess(c1(0.1), site="elbow")
    with pytest.raises(ValueError, match="age"):
        ENGINE.assess(c1(0.1), age=150)


def test_missing_value_spellings_are_unknown():
    for age in (None, float("nan"), ""):
        for site in (None, float("nan"), "", "unknown", "UNKNOWN"):
            c2 = ENGINE.assess(c1(0.35, 0, 0.5), age=age, site=site)
            assert c2["inputs"]["age"] is None and c2["inputs"]["site"] == "unknown"
