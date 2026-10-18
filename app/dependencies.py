from fastapi import HTTPException, Request

from app.services.screening import ScreeningService


def get_screening_service(request: Request) -> ScreeningService:
    service = request.app.state.predictor
    if service is None:
        raise HTTPException(503, "โมเดลยังไม่พร้อม กรุณารอให้เทรนเสร็จก่อน")
    return service
