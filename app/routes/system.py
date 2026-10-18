import json

from fastapi import APIRouter, Depends, Request

from app.core.config import MODEL_DIRECTORY
from app.dependencies import get_screening_service
from app.services.screening import ScreeningService
from ml.common import CLASSES, LABELS_TH

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health")
def health(request: Request):
    service = request.app.state.predictor
    return {
        "ready": service is not None,
        "experimental": True,
        "model_version": service.meta["version"] if service else None,
        "weather_model_trained": False,
        "support_policy_ready": service.meta["support_policy_ready"] if service else False,
    }


@router.get("/model")
def model_info(service: ScreeningService = Depends(get_screening_service)):
    def optional_json(name):
        path = MODEL_DIRECTORY / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {
        "metadata": service.meta,
        "metrics": optional_json("metrics.json"),
        "policy_metrics": optional_json("policy_metrics.json"),
        "classes": [
            {"id": name, "label": label, "requires_confirmation": name == "RedRot"}
            for name, label in zip(CLASSES, LABELS_TH)
        ],
    }
