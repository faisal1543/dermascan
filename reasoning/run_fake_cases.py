"""DERM-12: run the fuzzy prototype on the 10 fake C1 cases and save the C2 outputs.

Usage:  python -m reasoning.run_fake_cases [--out results/reasoning/fake_cases_c2.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoning.fuzzy import TriageEngine

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).resolve().parent / "fake_cases.json"


def run(engine: TriageEngine | None = None) -> list[dict]:
    engine = engine or TriageEngine()
    results = []
    for case in json.loads(CASES.read_text())["cases"]:
        c2 = engine.assess(case["c1"], age=case["age"], site=case["site"])
        results.append({"case": case["case"], "expected": case["expected"], "c2": c2})
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "reasoning" / "fake_cases_c2.json")
    args = p.parse_args(argv)
    results = run()
    print(f"{'case':<42} {'urgency':<14} {'score':>5}  rules applied (strength)")
    for r in results:
        c2 = r["c2"]
        rules = ", ".join(f"{f['id']}({f['strength']:.2f})" for f in c2["fired_rules"]
                          if f.get("applied", True))
        flag = "" if c2["urgency"] == r["expected"] else f"   <-- expected {r['expected']}"
        score = "-" if c2["score"] is None else f"{c2['score']:.1f}"
        print(f"{r['case']:<42} {c2['urgency']:<14} {score:>5}  {rules}{flag}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nC2 outputs written to {args.out}")
    return 0 if all(r["c2"]["urgency"] == r["expected"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
