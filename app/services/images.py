"""One upload/decoding path shared by leaf and field analysis."""

import io

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import ALLOWED_IMAGE_FORMATS, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES

Image.MAX_IMAGE_PIXELS = 20_000_000


def decode_image(content: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(415, "รองรับเฉพาะภาพ JPEG, PNG และ WebP")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "ภาพมีความละเอียดสูงเกินไป กรุณาลดขนาดไม่เกิน 16 ล้านพิกเซล")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, "อ่านภาพไม่ได้ กรุณาเลือกไฟล์ภาพที่สมบูรณ์")


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "ภาพต้องมีขนาดไม่เกิน 8 MB")
    if not content:
        raise HTTPException(400, "กรุณาเลือกภาพ")
    return decode_image(content)


def assess_quality(image: Image.Image) -> dict:
    gray = np.asarray(image.resize((224, 224)).convert("L"), dtype=np.float32)
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    issues = []
    if min(image.size) < 160:
        issues.append("ภาพเล็กเกินไป กรุณาถ่ายใหม่ให้เห็นรายละเอียดใบ")
    if gray.mean() < 18:
        issues.append("ภาพมืดเกินไป กรุณาถ่ายในที่สว่าง")
    if gray.mean() > 242:
        issues.append("ภาพสว่างเกินไป กรุณาหลีกเลี่ยงแสงสะท้อน")
    if gray.std() < 5 or lap.var() < 2:
        issues.append("รายละเอียดภาพไม่เพียงพอ กรุณาจับโฟกัสที่ใบอ้อย")
    return {"passed": not issues, "issues": issues}
