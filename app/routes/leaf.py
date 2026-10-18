from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.security import require_api_secret
from app.dependencies import get_screening_service
from app.services.images import read_uploaded_image

router = APIRouter(prefix="/api", tags=["Leaf screening"])


@router.post("/analyze", dependencies=[Depends(require_api_secret)])
async def analyze(request: Request, file: Annotated[UploadFile, File()]):
    image = await read_uploaded_image(file)
    service = get_screening_service(request)
    return await run_in_threadpool(service.predict, image)
