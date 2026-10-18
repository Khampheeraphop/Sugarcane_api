"""Application service: quality -> model measurements -> policy -> user-facing result."""

import time
from pathlib import Path

from PIL import Image

from app.domain.diagnosis_policy import decide_screening
from app.domain.knowledge import advice
from app.services.images import assess_quality
from ml.common import CLASSES, LABELS_TH
from ml.inference import ModelRuntime


class ScreeningService:
    def __init__(self, model_path: Path, policy_path: Path | None = None):
        self.runtime = ModelRuntime(model_path, policy_path)
        self.meta = {
            **self.runtime.metadata,
            "version": self.runtime.metadata["version"] + "+support-v2",
            "requires_confirmation": ["RedRot"],
            "support_policy_ready": self.runtime.policy is not None,
            "confidence_display_enabled": False,
        }

    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        quality = assess_quality(image)
        result = {
            "model_version": self.meta["version"],
            "experimental": True,
            "quality": quality,
            "confidence": None,
            "candidates": [],
            "weather_risk": None,
            "notice": "ผลคัดกรองเบื้องต้นจากภาพ ไม่ยืนยันเชื้อหรือการปลอดโรค",
        }
        if not quality["passed"]:
            return {
                **result,
                "status": "poor_quality",
                "title": "กรุณาถ่ายภาพใหม่",
                "message": quality["issues"][0],
                "next_steps": ["ให้ใบอ้อยอยู่ใกล้กล้อง ภาพคมชัด และมีแสงธรรมชาติ"],
            }
        scores = self.runtime.measure(image)
        decision = decide_screening(scores, self.runtime.metadata, self.runtime.policy)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        if decision.status == "unrecognized":
            return {
                **result,
                "status": "unrecognized",
                "title": "ยังยืนยันว่าเป็นใบอ้อยไม่ได้",
                "message": "ภาพอาจเป็นพืชอื่น หรือแตกต่างจากภาพที่โมเดลเคยเรียนรู้ จึงยังไม่ระบุโรค",
                "next_steps": [
                    "ถ่ายใบอ้อย 1 ใบให้ชัดทั้งด้านหน้าและด้านหลัง",
                    "หลีกเลี่ยงภาพสิ่งของ ภาพหน้าจอ หรือใบพืชอื่น",
                ],
            }
        if decision.status == "uncertain":
            additional_parts = decision.reason == "additional_plant_parts_required"
            return {
                **result,
                "status": "uncertain",
                "title": "พบอาการที่ยังระบุโรคไม่ได้",
                "message": "ต้องตรวจลำต้นและอาการทั้งต้นประกอบ ภาพใบนี้ยังไม่พอให้ระบุชื่อโรค"
                if additional_parts
                else "รูปแบบอาการยังมีหลักฐานไม่พอกับกลุ่มที่โมเดลรองรับ อาจเป็นโรคหรือความผิดปกติชนิดอื่น",
                "next_steps": [
                    "ถ่ายทั้งต้น ใบด้านหลัง และบริเวณที่เริ่มมีอาการ",
                    "ให้เจ้าหน้าที่เกษตรหรือนักโรคพืชตรวจตัวอย่างก่อนจัดการแปลง",
                ],
            }
        index = CLASSES.index(decision.class_id)
        healthy = decision.status == "healthy"
        return {
            **result,
            "status": decision.status,
            "class_id": decision.class_id,
            "title": LABELS_TH[index],
            "advice": advice(decision.class_id),
            "message": "ไม่พบรูปแบบอาการในกลุ่มที่โมเดลรู้จัก ยังไม่สามารถยืนยันว่าปลอดโรคทั้งหมด"
            if healthy
            else "พบรูปแบบภาพใกล้เคียงกลุ่มอาการนี้ ควรตรวจยืนยันก่อนตัดสินใจจัดการแปลง",
            "next_steps": ["ติดตามอาการและให้ผู้เชี่ยวชาญตรวจยืนยันหากมีความผิดปกติเพิ่ม"],
        }
