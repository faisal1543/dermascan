"""SQLite case log (DERM-16, contract C4): one row per assessment, plus the reviewer's decision.

The image itself is never stored, only its SHA-256 (proposal Section 6, Privacy). A case is decided
once; a later change of mind is a new case (409 decision_already_recorded). The log stays on the
demo machine and is deleted at project end:

    python -m app.caselog --delete --db caselog.db --yes
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.config import DECISIONS
from app.errors import ApiError
from app.validation import check_clinician

COLUMNS = (
    "case_id", "created_at", "image_sha256", "age", "site", "probs_json", "in_distribution",
    "recommended_urgency", "score", "fired_rules_json", "final_decision", "override_reason",
    "clinician", "decided_at", "model_version", "rules_version",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    age INTEGER,
    site TEXT NOT NULL,
    probs_json TEXT NOT NULL,
    in_distribution INTEGER NOT NULL,
    recommended_urgency TEXT NOT NULL,
    score REAL,
    fired_rules_json TEXT NOT NULL,
    final_decision TEXT,
    override_reason TEXT,
    clinician TEXT NOT NULL,
    decided_at TEXT,
    model_version TEXT NOT NULL,
    rules_version TEXT NOT NULL
)
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        """Short-lived connection per call (safe across FastAPI worker threads).

        Commits on success, rolls back on error, and always closes. The table is created on
        first use, so importing the app writes nothing to disk.
        """
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute(SCHEMA)
                yield conn
        finally:
            conn.close()

    def insert(self, assessment: dict) -> dict:
        """Store a new case from AssessmentService.assess() output. Returns the C4 row."""
        log, c1, c2 = assessment["_log"], assessment["c1"], assessment["c2"]
        row = {
            "case_id": assessment["case_id"],
            "created_at": now_utc(),
            "image_sha256": log["image_sha256"],
            "age": log["age"],
            "site": log["site"],
            "probs_json": json.dumps(c1["probs"]),
            "in_distribution": int(bool(c1["in_distribution"])),
            "recommended_urgency": c2["urgency"],
            "score": c2["score"],
            "fired_rules_json": json.dumps(c2["fired_rules"]),
            "final_decision": None,
            "override_reason": None,
            "clinician": log["clinician"],
            "decided_at": None,
            "model_version": c1["model_version"],
            "rules_version": c2["rules_version"],
        }
        with self._connect() as conn:
            conn.execute(f"INSERT INTO cases ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                         [row[c] for c in COLUMNS])
        return row

    def get(self, case_id: str) -> dict:
        with self._connect() as conn:
            found = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if found is None:
            raise ApiError(404, "case_not_found", f"No case with id {case_id}")
        return dict(found)

    def decide(self, case_id: str, final_decision, override_reason, clinician) -> dict:
        """Record the reviewer's decision once. Overrides and cannot_assess cases need a reason (O3)."""
        case = self.get(case_id)
        if case["final_decision"] is not None:
            raise ApiError(409, "decision_already_recorded",
                           "This case already has a decision. Start a new case to change it.")
        if final_decision not in DECISIONS:
            raise ApiError(422, "invalid_decision", f"Decision must be one of: {', '.join(DECISIONS)}")
        clinician = check_clinician(clinician)
        reason = (override_reason or "").strip() or None
        overriding = final_decision != case["recommended_urgency"]  # always true for cannot_assess
        if overriding and reason is None:
            raise ApiError(422, "override_reason_required",
                           "Give a reason when the decision differs from the recommendation")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE cases SET final_decision = ?, override_reason = ?, clinician = ?, decided_at = ? "
                "WHERE case_id = ? AND final_decision IS NULL",
                (final_decision, reason, clinician, now_utc(), case_id))
            if cur.rowcount != 1:  # another request decided it first
                raise ApiError(409, "decision_already_recorded",
                               "This case already has a decision. Start a new case to change it.")
        return self.get(case_id)

    def count(self) -> int:
        with self._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DermaScan case log maintenance")
    ap.add_argument("--db", default="caselog.db", type=Path)
    ap.add_argument("--delete", action="store_true", help="delete the case log file (project end)")
    ap.add_argument("--yes", action="store_true", help="confirm the delete")
    args = ap.parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} does not exist")
        return 0
    if not args.delete:
        print(f"{args.db}: {CaseLog(args.db).count()} cases")
        return 0
    if not args.yes:
        print("Refusing to delete without --yes")
        return 1
    args.db.unlink()
    print(f"deleted {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
