"""Input checks for /api/assess (DERM-13, contract C3).

Every check raises ApiError with the C3 error code, so the API layer only has to turn it into JSON.
Images are checked by decoding them, never by trusting the file name or the content type.
"""
from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from app.config import ALLOWED_FORMATS, MAX_IMAGE_BYTES, SITES
from app.errors import ApiError


def check_image(data: bytes) -> Image.Image:
    """Return the decoded RGB image, or raise 413 image_too_large / 400 invalid_image."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(413, "image_too_large", f"Image is over {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    if not data:
        raise ApiError(400, "invalid_image", "The uploaded file is empty")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.load()  # full decode: truncated or corrupt files fail here
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ApiError(400, "invalid_image", "File is not a JPEG or PNG") from None
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(400, "invalid_image", "File is not a JPEG or PNG")
    if rgb.width == 0 or rgb.height == 0:
        raise ApiError(400, "invalid_image", "Image has no pixels")
    return rgb


def check_age(value) -> int | None:
    """Optional whole number 0-120. Empty or missing means unknown (None)."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        age = int(text)
    except ValueError:
        raise ApiError(422, "invalid_age", "Age must be a whole number from 0 to 120") from None
    if not 0 <= age <= 120:
        raise ApiError(422, "invalid_age", "Age must be a whole number from 0 to 120")
    return age


def check_site(value) -> str:
    """Optional HAM10000 body site. Empty or missing means 'unknown'."""
    if value is None or str(value).strip() == "":
        return "unknown"
    site = str(value).strip().lower()
    if site not in SITES:
        raise ApiError(422, "invalid_site", f"Body site must be one of: {', '.join(SITES)}")
    return site


def check_clinician(value) -> str:
    """Required reviewer ID picked at session start (there is no login; a stated limitation)."""
    clinician = "" if value is None else str(value).strip()
    if not clinician:
        raise ApiError(422, "invalid_clinician", "A reviewer ID is required")
    if len(clinician) > 64:
        raise ApiError(422, "invalid_clinician", "Reviewer ID must be 64 characters or fewer")
    return clinician
