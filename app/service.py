"""The assessment flow behind /api/assess (DERM-13), kept free of FastAPI so it is easy to test.

check inputs -> hash the image -> perception (mocked, C1) -> fuzzy layer (C2) -> response.
The image bytes are only used in memory and never stored (proposal Section 6, Privacy).
"""
from __future__ import annotations

import hashlib
import uuid

from app import validation
from app.config import MOCK_MODEL_VERSION, NOTICE
from app.mock_perception import mock_c1
from reasoning.fuzzy import TriageEngine


class AssessmentService:
    def __init__(self, engine: TriageEngine | None = None):
        self.engine = engine or TriageEngine()

    @property
    def model_version(self) -> str:
        return MOCK_MODEL_VERSION

    @property
    def rules_version(self) -> str:
        return self.engine.version

    def assess(self, image: bytes, age=None, site=None, clinician=None) -> dict:
        """Validate the request and return {case_id, c1, c2, notice}. Raises ApiError on bad input."""
        clinician = validation.check_clinician(clinician)
        age = validation.check_age(age)
        site = validation.check_site(site)
        validation.check_image(image)
        image_sha256 = hashlib.sha256(image).hexdigest()

        c1 = mock_c1(image_sha256)
        c2 = self.engine.assess(c1, age=age, site=site)
        return {
            "case_id": str(uuid.uuid4()),
            "c1": c1,
            "c2": c2,
            "notice": NOTICE,
            # kept for the case log (DERM-16); not part of the C3 response shown to the screen
            "_log": {"image_sha256": image_sha256, "age": age, "site": site, "clinician": clinician},
        }
