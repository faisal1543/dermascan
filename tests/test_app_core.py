"""DERM-13: input checks, mocked perception and the assessment flow (no FastAPI needed)."""
from __future__ import annotations

import hashlib
import io
import math

import pytest
from PIL import Image

from app import validation
from app.config import MAX_IMAGE_BYTES, NOTICE, SITES
from app.errors import ApiError
from app.mock_perception import CLASSES, FAKE_CASES, mock_c1, pick_case
from app.service import AssessmentService

SERVICE = AssessmentService()


def image_bytes(fmt: str = "JPEG", size=(64, 48), color=(180, 120, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def image_for_case(case_name: str) -> bytes:
    """A small valid PNG whose hash picks the named fake case (search over colours)."""
    for r in range(256):
        for g in range(0, 256, 16):
            data = image_bytes("PNG", (16, 16), (r, g, 90))
            if pick_case(hashlib.sha256(data).hexdigest())["case"] == case_name:
                return data
    raise AssertionError(f"no test image found for {case_name}")


def error_code(fn, *args) -> tuple[int, str]:
    with pytest.raises(ApiError) as info:
        fn(*args)
    return info.value.status, info.value.code


# --- image checks -----------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_jpeg_and_png_accepted(fmt):
    img = validation.check_image(image_bytes(fmt))
    assert img.mode == "RGB" and img.size == (64, 48)


def test_other_formats_rejected_even_if_valid_images():
    assert error_code(validation.check_image, image_bytes("GIF")) == (400, "invalid_image")
    assert error_code(validation.check_image, image_bytes("BMP")) == (400, "invalid_image")


def test_non_image_and_empty_files_rejected():
    assert error_code(validation.check_image, b"not an image") == (400, "invalid_image")
    assert error_code(validation.check_image, b"") == (400, "invalid_image")


def test_truncated_image_rejected():
    data = image_bytes("JPEG", (200, 200))
    assert error_code(validation.check_image, data[: len(data) // 3]) == (400, "invalid_image")


def test_oversized_file_rejected_before_decoding():
    assert error_code(validation.check_image, b"\0" * (MAX_IMAGE_BYTES + 1)) == (413, "image_too_large")


# --- age, site, reviewer ----------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("  ", None), ("0", 0),
                                             ("64", 64), (" 120 ", 120), (45, 45)])
def test_valid_ages(value, expected):
    assert validation.check_age(value) == expected


@pytest.mark.parametrize("value", ["-1", "121", "abc", "45.5", "1e2"])
def test_invalid_ages(value):
    assert error_code(validation.check_age, value) == (422, "invalid_age")


def test_sites_match_contract_and_default_to_unknown():
    assert len(SITES) == 15
    assert validation.check_site(None) == "unknown"
    assert validation.check_site("") == "unknown"
    assert validation.check_site(" Lower Extremity ") == "lower extremity"
    assert error_code(validation.check_site, "elbow") == (422, "invalid_site")


def test_sites_match_the_fuzzy_layer():
    """Every C3 site except 'unknown' must be a known site in the rule file."""
    assert set(SITES) - {"unknown"} == set(SERVICE.engine.site_group)


def test_reviewer_id_required():
    assert validation.check_clinician(" dr-a ") == "dr-a"
    assert error_code(validation.check_clinician, None) == (422, "invalid_clinician")
    assert error_code(validation.check_clinician, "   ") == (422, "invalid_clinician")
    assert error_code(validation.check_clinician, "x" * 65) == (422, "invalid_clinician")


# --- mocked perception (C1) -------------------------------------------------------------

def test_mock_c1_follows_contract():
    c1 = mock_c1(hashlib.sha256(b"abc").hexdigest())
    assert list(c1["probs"]) == CLASSES
    assert math.isclose(sum(c1["probs"].values()), 1.0, abs_tol=1e-3)
    assert c1["p_malignant"] == pytest.approx(c1["probs"]["mel"] + c1["probs"]["bcc"]
                                              + c1["probs"]["akiec"], abs=1e-4)
    assert 0.0 <= c1["entropy"] <= 1.0
    assert c1["image_id"].startswith("sha256:")
    assert c1["model_version"].startswith("mock-")


def test_mock_c1_is_deterministic_and_covers_every_case():
    sha = hashlib.sha256(b"same image").hexdigest()
    assert mock_c1(sha) == mock_c1(sha)
    import json
    names = {c["case"] for c in json.loads(FAKE_CASES.read_text())["cases"]}
    picked = {pick_case(f"{i:08x}" + "0" * 56)["case"] for i in range(len(names))}
    assert picked == names


# --- the whole flow ---------------------------------------------------------------------

def test_assess_returns_c1_c2_and_notice():
    out = SERVICE.assess(image_bytes(), age="64", site="back", clinician="dr-a")
    assert set(out) == {"case_id", "c1", "c2", "notice", "_log"}
    assert out["notice"] == NOTICE
    assert out["c2"]["urgency"] in {"urgent", "routine", "monitor", "cannot_assess"}
    assert out["c2"]["inputs"]["age"] == 64 and out["c2"]["inputs"]["site"] == "back"
    assert out["_log"]["image_sha256"] == hashlib.sha256(image_bytes()).hexdigest()


def test_each_request_gets_a_new_case_id():
    a = SERVICE.assess(image_bytes(), clinician="dr-a")
    b = SERVICE.assess(image_bytes(), clinician="dr-a")
    assert a["case_id"] != b["case_id"]
    assert a["c2"] == b["c2"]  # same image, same inputs -> same recommendation


def test_missing_age_and_site_are_unknown():
    out = SERVICE.assess(image_bytes(), clinician="dr-a")
    assert out["c2"]["inputs"]["age"] is None and out["c2"]["inputs"]["site"] == "unknown"


def test_out_of_distribution_image_cannot_be_assessed():
    out = SERVICE.assess(image_for_case("out_of_distribution_photo"), clinician="dr-a")
    assert out["c1"]["in_distribution"] is False
    assert out["c2"]["urgency"] == "cannot_assess" and out["c2"]["score"] is None


def test_clear_melanoma_case_is_urgent():
    out = SERVICE.assess(image_for_case("clear_melanoma"), age="64", site="back", clinician="dr-a")
    assert out["c2"]["urgency"] == "urgent"


def test_bad_input_raises_before_any_assessment():
    assert error_code(SERVICE.assess, b"junk", None, None, "dr-a") == (400, "invalid_image")
    assert error_code(SERVICE.assess, image_bytes(), "200", None, "dr-a") == (422, "invalid_age")
    assert error_code(SERVICE.assess, image_bytes(), None, None, "") == (422, "invalid_clinician")
