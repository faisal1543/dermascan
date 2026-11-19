"""One error shape for every API error (contract C3): {"error": "<code>", "message": "..."}."""
from __future__ import annotations


class ApiError(Exception):
    """An input or state problem the client can fix. Carries the C3 code and HTTP status."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict:
        return {"error": self.code, "message": self.message}
