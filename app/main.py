"""DermaScan backend (DERM-13): FastAPI app implementing contract C3.

Run locally:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000/docs for the interactive API page.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import MAX_IMAGE_BYTES
from app.errors import ApiError
from app.service import AssessmentService


def create_app(service: AssessmentService | None = None) -> FastAPI:
    service = service or AssessmentService()
    api = FastAPI(
        title="DermaScan API",
        version="0.1.0",
        description="Clinician decision support for dermoscopic images. Not a diagnosis.",
    )

    @api.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=exc.body())

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

    return api


app = create_app()
