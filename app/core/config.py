"""Application settings. Model training configuration lives in ml/ instead."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "torch")
if INFERENCE_BACKEND not in {"torch", "onnx"}:
    raise ValueError("INFERENCE_BACKEND must be torch or onnx")
MODEL_PATH = MODEL_DIRECTORY / ("backbone.onnx" if INFERENCE_BACKEND == "onnx" else "model.pt")
POLICY_PATH = MODEL_DIRECTORY / "policy.json"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


def cors_origins() -> list[str]:
    return os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
