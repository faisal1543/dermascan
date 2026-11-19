"""The assessment flow behind the API (DERM-13, DERM-16), kept free of FastAPI so it is easy to test.

check inputs -> hash the image -> perception (mocked, C1) -> fuzzy layer (C2) -> response.
The image bytes are only used in memory and never stored (proposal Section 6, Privacy).
"""
from __future__ import annotations

import hashlib
import uuid

from app import validation
from app.caselog import CaseLog
from app.config import MOCK_MODEL_VERSION, NOTICE
from app.mock_perception import mock_c1
from reasoning.fuzzy import TriageEngine


class AssessmentService:
    def __init__(self, engine: TriageEngine | None = None, caselog: CaseLog | None = None):
        self.engine = engine or TriageEngine()
        self.caselog = caselog  # None: nothing is logged (unit tests of the flow only)

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
        result = {
            "case_id": str(uuid.uuid4()),
            "c1": c1,
            "c2": c2,
            "notice": NOTICE,
            # kept for the case log (DERM-16); not part of the C3 response shown to the screen
            "_log": {"image_sha256": image_sha256, "age": age, "site": site, "clinician": clinician},
        }
        if self.caselog is not None:
            self.caselog.insert(result)  # every path, cannot_assess included, ends in the log
        return result

    def decide(self, case_id: str, final_decision, override_reason, clinician) -> dict:
        return self._log().decide(case_id, final_decision, override_reason, clinician)

    def get_case(self, case_id: str) -> dict:
        return self._log().get(case_id)

    def _log(self) -> CaseLog:
        if self.caselog is None:
            raise RuntimeError("this service was created without a case log")
        return self.caselog
