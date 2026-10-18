"""Shared authentication dependency for the inference and field endpoints."""

import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException


def require_api_secret(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("INFERENCE_API_SECRET")
    if expected and not secrets.compare_digest(authorization or "", f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
