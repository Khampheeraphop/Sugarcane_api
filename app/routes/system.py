import json

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import MODEL_DIRECTORY
from app.dependencies import get_screening_service
from app.domain.knowledge import advice
from app.services.screening import ScreeningService
from ml.labels import CLASSES, LABELS_TH

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


@router.get("/diseases")
def disease_catalog():
    return {
        "classes": [
            {
                "id": name,
                "label": label.replace("อาการคล้าย", ""),
                "requires_confirmation": name == "RedRot",
                "leaf_output_enabled": name != "RedRot",
                "advice": advice(name),
            }
            for name, label in zip(CLASSES, LABELS_TH)
        ],
        "unsupported": ["โรคสนิม (Rust)", "อาการใบจาง/ใบลวก", "โรคใบด่าง (Mosaic)", "โรคอื่นนอกชุดฝึก"],
        "notice": "รองรับเพียงกลุ่มอาการในชุดฝึก ไม่ยืนยันเชื้อหรือสาเหตุจริงของต้นที่ถ่ายภาพ",
    }


@router.get("/ready")
def readiness(request: Request):
    result = health(request)
    if not result["ready"] or not result["support_policy_ready"]:
        raise HTTPException(503, "Model and matching support policy are not ready")
    return result
