"""DermaScan backend (DERM-13, DERM-16): FastAPI app implementing contract C3, logging to C4.

Run locally:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000/ for the review screen, or /docs for the interactive API page.
The case log is a SQLite file, `caselog.db` in the working folder (set DERMASCAN_DB to change it).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.caselog import CaseLog
from app.config import MAX_IMAGE_BYTES
from app.errors import ApiError
from app.service import AssessmentService

STATIC = Path(__file__).resolve().parent / "static"
logger = logging.getLogger("dermascan.api")


class Decision(BaseModel):
    final_decision: str
    override_reason: str | None = None
    clinician: str


def create_app(service: AssessmentService | None = None, db_path: Path | str | None = None) -> FastAPI:
    if service is None:
        db_path = db_path or os.environ.get("DERMASCAN_DB", "caselog.db")
        service = AssessmentService(caselog=CaseLog(db_path))
    api = FastAPI(
        title="DermaScan API",
        version="0.2.0",
        description="Clinician decision support for dermoscopic images. Not a diagnosis.",
    )

    @api.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=exc.body())

    @api.exception_handler(Exception)
    async def _internal_error(_: Request, exc: Exception):
        # Anything unexpected (for example bad C1 from the real model in DERM-28) still uses
        # the one C3 error shape, and never leaks a stack trace to the screen.
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={
            "error": "internal_error", "message": "Something went wrong on the server. The case was not assessed."})

    @api.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError):
        fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors())
        return JSONResponse(status_code=422, content={
            "error": "invalid_request", "message": f"Missing or malformed field(s): {fields}"})

    @api.get("/health")
    def health():
        return {"status": "ok", "model_version": service.model_version,
                "rules_version": service.rules_version}

    @api.post("/api/assess")
    async def assess(
        image: UploadFile = File(..., description="Dermoscopic image, JPEG or PNG, up to 10 MB"),
        age: str | None = Form(None, description="Whole number 0-120, optional"),
        site: str | None = Form(None, description="HAM10000 body site, optional"),
        clinician: str | None = Form(None, description="Reviewer ID picked at session start"),
    ):
        data = await image.read(MAX_IMAGE_BYTES + 1)  # never read more than the limit + 1 byte
        result = service.assess(data, age=age, site=site, clinician=clinician)
        result.pop("_log", None)
        return result

    @api.post("/api/cases/{case_id}/decision")
    def decide(case_id: str, decision: Decision):
        return service.decide(case_id, decision.final_decision, decision.override_reason,
                              decision.clinician)

    @api.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        return service.get_case(case_id)

    # The review screen. Mounted last so the API routes above take priority.
    api.mount("/", StaticFiles(directory=STATIC, html=True), name="screen")
    return api


app = create_app()
