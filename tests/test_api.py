"""DERM-13 / DERM-16: the HTTP API (contract C3) and case log (C4). Runs in CI, where FastAPI is installed."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.config import MAX_IMAGE_BYTES, NOTICE  # noqa: E402
from app.main import create_app  # noqa: E402

DB = Path(tempfile.mkdtemp()) / "caselog.db"
client = TestClient(create_app(db_path=DB))


def png(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (180, 120, 90)).save(buf, "PNG")
    return buf.getvalue()


def assess(data: bytes = None, name="lesion.png", **form):
    form.setdefault("clinician", "dr-a")
    return client.post("/api/assess", files={"image": (name, png() if data is None else data, "image/png")},
                       data={k: v for k, v in form.items() if v is not None})


def test_health_reports_versions():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("mock-")
    assert body["rules_version"].startswith("fuzzy-v")


def test_assess_returns_contract_shape():
    r = assess(age="64", site="back")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"case_id", "c1", "c2", "notice"}
    assert body["notice"] == NOTICE
    assert set(body["c1"]["probs"]) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert body["c2"]["urgency"] in {"urgent", "routine", "monitor", "cannot_assess"}
    assert body["c2"]["inputs"]["age"] == 64


def test_assess_accepts_jpeg_without_age_or_site():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "JPEG")
    r = assess(buf.getvalue(), name="lesion.jpg")
    assert r.status_code == 200, r.text
    assert r.json()["c2"]["inputs"]["site"] == "unknown"


@pytest.mark.parametrize("data, status, code", [
    (b"not an image", 400, "invalid_image"),
    (b"\0" * (MAX_IMAGE_BYTES + 1), 413, "image_too_large"),
])
def test_bad_files_use_the_error_shape(data, status, code):
    r = assess(data)
    assert r.status_code == status
    assert r.json() == {"error": code, "message": r.json()["message"]}


def test_renamed_non_image_is_still_rejected():
    r = assess(b"GIF89a fake", name="lesion.png")
    assert r.status_code == 400 and r.json()["error"] == "invalid_image"


@pytest.mark.parametrize("form, code", [
    ({"age": "121"}, "invalid_age"),
    ({"age": "old"}, "invalid_age"),
    ({"site": "elbow"}, "invalid_site"),
    ({"clinician": ""}, "invalid_clinician"),
])
def test_bad_fields_return_422(form, code):
    r = assess(**form)
    assert r.status_code == 422
    assert r.json()["error"] == code


def test_missing_image_is_a_malformed_request():
    r = client.post("/api/assess", data={"clinician": "dr-a"})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


# --- DERM-16: decisions and the case log -----------------------------------------------------

def new_case(**form) -> dict:
    r = assess(**form)
    assert r.status_code == 200, r.text
    return r.json()


def decide(case_id, final, reason=None, clinician="dr-a"):
    return client.post(f"/api/cases/{case_id}/decision",
                       json={"final_decision": final, "override_reason": reason, "clinician": clinician})


def test_every_assessment_is_logged_without_the_image():
    case = new_case(age="50", site="trunk")
    r = client.get(f"/api/cases/{case['case_id']}")
    assert r.status_code == 200
    row = r.json()
    assert row["recommended_urgency"] == case["c2"]["urgency"]
    assert row["final_decision"] is None and row["clinician"] == "dr-a"
    assert len(row["image_sha256"]) == 64
    assert "image" not in row and all(not str(v).startswith("iVBOR") for v in row.values())


def test_confirming_needs_no_reason():
    case = new_case()
    rec = case["c2"]["urgency"]
    if rec == "cannot_assess":
        pytest.skip("this test image picked the cannot-assess case")
    r = decide(case["case_id"], rec)
    assert r.status_code == 200, r.text
    assert r.json()["final_decision"] == rec and r.json()["decided_at"]


def test_override_needs_a_reason_and_is_logged():
    case = new_case()
    other = "monitor" if case["c2"]["urgency"] != "monitor" else "urgent"
    r = decide(case["case_id"], other)
    assert r.status_code == 422 and r.json()["error"] == "override_reason_required"
    r = decide(case["case_id"], other, reason="Lesion changed over 3 months")
    assert r.status_code == 200
    assert r.json()["override_reason"] == "Lesion changed over 3 months"


def test_a_case_is_decided_once():
    case = new_case()
    other = "monitor" if case["c2"]["urgency"] != "monitor" else "urgent"
    assert decide(case["case_id"], other, reason="first").status_code == 200
    r = decide(case["case_id"], other, reason="second")
    assert r.status_code == 409 and r.json()["error"] == "decision_already_recorded"


def test_unknown_case_and_bad_decision():
    assert client.get("/api/cases/nope").json()["error"] == "case_not_found"
    assert decide("nope", "urgent", "x").status_code == 404
    case = new_case()
    r = decide(case["case_id"], "biopsy", reason="x")
    assert r.status_code == 422 and r.json()["error"] == "invalid_decision"
    r = client.post(f"/api/cases/{case['case_id']}/decision", json={"final_decision": "urgent"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_review_screen_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "DermaScan" in r.text and "not a diagnosis" in r.text
    assert client.get("/app.js").status_code == 200
