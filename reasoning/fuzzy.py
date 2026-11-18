"""DERM-12: Mamdani fuzzy triage layer. Reads contract C1, returns contract C2.

Rules, terms and sources live in a versioned rule file (reasoning/rules/rules_v1.json, DERM-52);
this module contains no medical knowledge of its own. Design: docs/fuzzy-design.md.

    from reasoning.fuzzy import TriageEngine
    engine = TriageEngine()
    engine.assess(c1, age=64, site="back")   # -> contract C2 dict

Inference: min for AND, max for aggregation, centroid defuzzification on 0-100 ("score" rules),
then "floor" rules that can only raise the category ("IF uncertainty is high THEN at least
routine"). Membership shapes match skfuzzy.trimf / skfuzzy.trapmf (tests cross-check this when
scikit-fuzzy is installed, as in CI).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RULES_V1 = Path(__file__).resolve().parent / "rules" / "rules_v1.json"
CATEGORY_ORDER = ["monitor", "routine", "urgent"]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # Ground rule 4
MALIGNANT = ("mel", "bcc", "akiec")
UNKNOWN = "unknown"


# --- membership functions ----------------------------------------------------------------

def membership(shape: str, params, x: float) -> float:
    """Degree of membership of scalar x in a 'tri' (a, b, c) or 'trap' (a, b, c, d) set."""
    if shape == "tri":
        a, b, c = params
        a, b, c, d = a, b, b, c
    elif shape == "trap":
        a, b, c, d = params
    else:
        raise ValueError(f"unknown shape {shape!r}")
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def membership_curve(shape: str, params, xs) -> np.ndarray:
    return np.array([membership(shape, params, float(x)) for x in xs])


# --- engine ------------------------------------------------------------------------------

class TriageEngine:
    def __init__(self, rules_path: Path | str = RULES_V1, resolution: int = 1001):
        self.spec = json.loads(Path(rules_path).read_text())
        validate_spec(self.spec)
        self.version = self.spec["version"]
        self.rules = self.spec["rules"]
        self.by_id = {r["id"]: r for r in self.rules}
        out = self.spec["output"]
        self.xs = np.linspace(*out["universe"], resolution)
        self.out_curves = {t: membership_curve(s["shape"], s["params"], self.xs)
                           for t, s in out["terms"].items()}
        self.min_score = {c["name"]: float(c["min_score"]) for c in out["categories"]}
        self.floor_threshold = float(out["floor_threshold"])
        site = self.spec["inputs"]["site"]["groups"]
        self.site_group = {s: g for g, sites in site.items() for s in sites}

    # inputs ------------------------------------------------------------------------------

    def fuzzify(self, crisp: dict) -> dict[str, dict[str, float]]:
        """Membership of each crisp input in each of its terms. Unknown values get all zeros."""
        degrees: dict[str, dict[str, float]] = {}
        for name, var in self.spec["inputs"].items():
            value = crisp.get(name)
            if var.get("categorical"):
                group = UNKNOWN if value == UNKNOWN else self.site_group[value]
                degrees[name] = {g: float(g == group) for g in var["groups"]}
            elif value is None:
                degrees[name] = {t: 0.0 for t in var["terms"]}
            else:
                degrees[name] = {t: membership(s["shape"], s["params"], value)
                                 for t, s in var["terms"].items()}
        return degrees

    def strength(self, rule: dict, degrees) -> float:
        return min(degrees[var][term] for var, term in rule["if"])

    def aggregate(self, score_strengths: dict[str, float]) -> np.ndarray:
        """Max of each score rule's output set clipped at its strength (Mamdani)."""
        agg = np.zeros_like(self.xs)
        for rule_id, s in score_strengths.items():
            if s > 0:
                agg = np.maximum(agg, np.minimum(s, self.out_curves[self.by_id[rule_id]["then"]]))
        return agg

    def category(self, score: float) -> str:
        return max((c for c in CATEGORY_ORDER if score >= self.min_score[c]),
                   key=CATEGORY_ORDER.index)

    # main entry point --------------------------------------------------------------------

    def assess(self, c1: dict, age=None, site=None) -> dict:
        """Contract C1 (JSON or CSV-row form) plus request context -> contract C2."""
        crisp = read_c1(c1)
        crisp["age"] = clean_age(age)
        crisp["site"] = self.clean_site(site)
        inputs = {"p_malignant": round(crisp["malignancy"], 4), "p_mel": round(crisp["melanoma"], 4),
                  "entropy": round(crisp["uncertainty"], 4), "age": crisp["age"],
                  "site": crisp["site"]}

        if not crisp["assessable"]:
            gate = self.by_id["G1"]
            return {"urgency": "cannot_assess", "score": None, "inputs": inputs,
                    "fired_rules": [_fired(gate, 1.0, applied=True)], "rules_version": self.version}

        degrees = self.fuzzify(crisp)
        score_rules = [r for r in self.rules if r["kind"] == "score"]
        floor_rules = [r for r in self.rules if r["kind"] == "floor"]
        score_strengths = {r["id"]: self.strength(r, degrees) for r in score_rules}
        agg = self.aggregate(score_strengths)

        fired = [_fired(r, s) for r in score_rules if (s := score_strengths[r["id"]]) > 0]
        if agg.sum() == 0:  # no score rule fired: should not happen with covering terms
            score, urgency = self.min_score["routine"], "routine"
            fired.append({"id": "R0", "kind": "fallback", "text": "No rule fired: default to routine",
                          "explanation": "No rule matched; referred as a precaution.",
                          "strength": 1.0})
        else:
            score = float((self.xs * agg).sum() / agg.sum())
            urgency = self.category(score)

        for r in floor_rules:
            s = self.strength(r, degrees)
            if s <= 0:
                continue
            applied = s >= self.floor_threshold
            fired.append(_fired(r, s, applied=applied))
            target = r["at_least"]
            if applied and CATEGORY_ORDER.index(target) > CATEGORY_ORDER.index(urgency):
                urgency = target
                score = max(score, self.min_score[target])  # keep the number consistent with the label

        fired.sort(key=lambda f: (-f["strength"], f["id"]))
        return {"urgency": urgency, "score": round(score, 1), "inputs": inputs,
                "fired_rules": fired, "rules_version": self.version}

    def clean_site(self, site) -> str:
        if site is None or (isinstance(site, float) and math.isnan(site)):
            return UNKNOWN
        s = str(site).strip().lower()
        if s in ("", UNKNOWN):
            return UNKNOWN
        if s not in self.site_group:
            raise ValueError(f"unknown body site {site!r}; expected one of {sorted(self.site_group)} or 'unknown'")
        return s


# --- helpers -----------------------------------------------------------------------------

def _fired(rule: dict, strength: float, applied: bool | None = None) -> dict:
    out = {"id": rule["id"], "kind": rule["kind"], "text": rule["text"],
           "explanation": rule["explanation"], "strength": round(float(strength), 3)}
    if rule["kind"] == "floor":
        out["applied"] = bool(applied)
    return out


def clean_age(age):
    """None/NaN/'' -> None (unknown). Out-of-range ages are an input error, not 'unknown'."""
    if age is None or age == "" or (isinstance(age, float) and math.isnan(age)):
        return None
    age = float(age)
    if not 0 <= age <= 120:
        raise ValueError(f"age must be between 0 and 120, got {age}")
    return age


def read_c1(c1: dict) -> dict:
    """Accept C1 as JSON ({"probs": {...}, ...}) or as a CSV row (prob_mel, ...)."""
    if "probs" in c1:
        probs = {c: float(c1["probs"][c]) for c in CLASSES}
    else:
        probs = {c: float(c1[f"prob_{c}"]) for c in CLASSES}
    total = sum(probs.values())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError(f"C1 probabilities must sum to 1, got {total:.4f}")
    p = np.clip(np.array([probs[c] for c in CLASSES]), 1e-12, 1.0)
    entropy = c1.get("entropy")
    if entropy is None:
        entropy = float(-(p * np.log(p)).sum() / np.log(len(CLASSES)))
    p_mal = c1.get("p_malignant")
    if p_mal is None:
        p_mal = sum(probs[c] for c in MALIGNANT)
    return {
        "melanoma": probs["mel"],
        "malignancy": float(p_mal),
        "uncertainty": float(entropy),
        "assessable": bool(c1.get("in_distribution", True)) and "error" not in c1,
    }


def validate_spec(spec: dict) -> None:
    """Fail fast on a malformed rule file (every rule needs an id, text, explanation and source)."""
    inputs, out = spec["inputs"], spec["output"]
    ids = [r["id"] for r in spec["rules"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rule ids")
    if "G1" not in ids:
        raise ValueError("the cannot-assess gate G1 is required")
    names = [c["name"] for c in out["categories"]]
    if names != CATEGORY_ORDER or [c["min_score"] for c in out["categories"]] != sorted(
            c["min_score"] for c in out["categories"]):
        raise ValueError("output categories must be monitor < routine < urgent")
    for r in spec["rules"]:
        for key in ("id", "kind", "text", "explanation", "source"):
            if not str(r.get(key, "")).strip():
                raise ValueError(f"rule {r.get('id')} has no {key}")
        if r["kind"] == "gate":
            continue
        if r["kind"] not in ("score", "floor"):
            raise ValueError(f"rule {r['id']}: unknown kind {r['kind']!r}")
        for var, term in r["if"]:
            spec_var = inputs.get(var)
            choices = (spec_var or {}).get("groups" if (spec_var or {}).get("categorical") else "terms", {})
            if term not in choices:
                raise ValueError(f"rule {r['id']}: {var} has no term {term!r}")
        if r["kind"] == "score" and r["then"] not in out["terms"]:
            raise ValueError(f"rule {r['id']}: unknown output term {r['then']!r}")
        if r["kind"] == "floor" and r["at_least"] not in CATEGORY_ORDER:
            raise ValueError(f"rule {r['id']}: unknown category {r['at_least']!r}")
