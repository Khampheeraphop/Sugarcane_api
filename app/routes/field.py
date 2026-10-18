import json
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.security import require_api_secret
from app.services.field_analysis import compare_field
from app.services.images import read_uploaded_image
from app.services.weather import forecast

router = APIRouter(
    prefix="/api/field", tags=["Field survey"], dependencies=[Depends(require_api_secret)]
)


@router.post("/analyze")
async def analyze(file: Annotated[UploadFile, File()], reference_tiles: Annotated[str, Form()]):
    image = await read_uploaded_image(file)
    try:
        references = json.loads(reference_tiles)
        if not isinstance(references, list):
            raise ValueError("เลือกช่องอ้างอิง 3–12 ช่อง")
        return await run_in_threadpool(compare_field, image, references)
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/weather")
async def weather(
    latitude: Annotated[float, Query(ge=-90, le=90)],
    longitude: Annotated[float, Query(ge=-180, le=180)],
):
    try:
        return await forecast(latitude, longitude)
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(502, "ดึงพยากรณ์อากาศไม่ได้ กรุณาลองอีกครั้ง ไม่มีการแสดงค่าทดแทน") from error
