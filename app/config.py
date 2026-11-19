"""Settings shared by the API (DERM-13). Values come from docs/contracts.md, contract C3."""
from __future__ import annotations

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB (C3)
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}  # checked on the decoded image, not the file name.
# MPO is the multi-picture JPEG some phone cameras save; Pillow decodes its first frame like a JPEG.

# The 15 HAM10000 body sites (C3). A missing value means "unknown".
SITES = (
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
    "lower extremity", "neck", "scalp", "trunk", "upper extremity", "unknown",
)

DECISIONS = ("urgent", "routine", "monitor")

NOTICE = "Decision support only, not a diagnosis."

# Until the real CNN is wired in (DERM-28), perception is mocked (app/mock_perception.py).
MOCK_MODEL_VERSION = "mock-cnn-v0"
