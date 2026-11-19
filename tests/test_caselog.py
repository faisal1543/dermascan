"""DERM-16: the SQLite case log (contract C4), tested without FastAPI."""
from __future__ import annotations

import io
import json
import sqlite3

import pytest
from PIL import Image

from app.caselog import COLUMNS, CaseLog, main
from app.errors import ApiError
from app.service import AssessmentService


@pytest.fixture
def service(tmp_path):
    return AssessmentService(caselog=CaseLog(tmp_path / "caselog.db"))


def png(color=(180, 120, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return buf.getvalue()


def code(fn, *args):
    with pytest.raises(ApiError) as info:
        fn(*args)
    return info.value.status, info.value.code


def other_than(urgency: str) -> str:
    return "monitor" if urgency != "monitor" else "urgent"


def test_table_has_the_c4_columns(service, tmp_path):
    service.assess(png(), clinician="dr-a")
    with sqlite3.connect(tmp_path / "caselog.db") as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(cases)")]
    assert tuple(cols) == COLUMNS


def test_assessment_is_logged_with_hash_not_image(service):
    out = service.assess(png(), age="70", site="face", clinician="dr-a")
    row = service.get_case(out["case_id"])
    assert row["image_sha256"] == out["_log"]["image_sha256"]
    assert row["age"] == 70 and row["site"] == "face"
    assert json.loads(row["probs_json"]) == out["c1"]["probs"]
    assert json.loads(row["fired_rules_json"]) == out["c2"]["fired_rules"]
    assert row["recommended_urgency"] == out["c2"]["urgency"]
    assert row["final_decision"] is None and row["decided_at"] is None
    assert row["created_at"].endswith("+00:00")


def test_confirm_override_and_once_only(service):
    out = service.assess(png(), clinician="dr-a")
    rec = out["c2"]["urgency"]
    other = other_than(rec)
    assert code(service.decide, out["case_id"], other, "  ", "dr-b") == (422, "override_reason_required")
    row = service.decide(out["case_id"], other, " Changed in size ", "dr-b")
    assert row["final_decision"] == other and row["override_reason"] == "Changed in size"
    assert row["clinician"] == "dr-b" and row["decided_at"]
    assert code(service.decide, out["case_id"], other, "again", "dr-b") == (409, "decision_already_recorded")


def test_confirming_the_recommendation_needs_no_reason(service):
    for color in range(0, 250, 10):  # find an assessable case
        out = service.assess(png((color, 50, 50)), clinician="dr-a")
        if out["c2"]["urgency"] != "cannot_assess":
            break
    row = service.decide(out["case_id"], out["c2"]["urgency"], None, "dr-a")
    assert row["override_reason"] is None


def test_bad_decisions(service):
    out = service.assess(png(), clinician="dr-a")
    assert code(service.decide, out["case_id"], "biopsy", "x", "dr-a") == (422, "invalid_decision")
    assert code(service.decide, out["case_id"], out["c2"]["urgency"], None, "") == (422, "invalid_clinician")
    assert code(service.get_case, "missing") == (404, "case_not_found")
    assert code(service.decide, "missing", "urgent", "x", "dr-a") == (404, "case_not_found")


def test_cannot_assess_always_needs_a_reason(tmp_path):
    log = CaseLog(tmp_path / "c.db")
    fake = {"case_id": "c1", "_log": {"image_sha256": "0" * 64, "age": None, "site": "unknown",
                                       "clinician": "dr-a"},
            "c1": {"probs": {}, "in_distribution": False, "model_version": "mock-cnn-v0"},
            "c2": {"urgency": "cannot_assess", "score": None, "fired_rules": [], "rules_version": "fuzzy-v0"}}
    log.insert(fake)
    assert code(log.decide, "c1", "routine", "", "dr-a") == (422, "override_reason_required")
    assert log.decide("c1", "routine", "Image too blurred to judge", "dr-a")["final_decision"] == "routine"


def test_importing_creates_no_file(tmp_path):
    CaseLog(tmp_path / "never.db")
    assert not (tmp_path / "never.db").exists()


def test_delete_command_needs_confirmation(service, tmp_path):
    service.assess(png(), clinician="dr-a")
    db = tmp_path / "caselog.db"
    assert main(["--db", str(db), "--delete"]) == 1 and db.exists()
    assert main(["--db", str(db), "--delete", "--yes"]) == 0 and not db.exists()
