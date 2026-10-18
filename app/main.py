"""Composition root. HTTP routing, domain policy and ML runtime are separate modules."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import MODEL_PATH, POLICY_PATH, cors_origins
from app.routes import field, leaf, system
from app.services.screening import ScreeningService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ScreeningService(MODEL_PATH, POLICY_PATH) if MODEL_PATH.exists() else None
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="Sugarcane Care API", version="0.2.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    for router in [system.router, leaf.router, field.router]:
        application.include_router(router)
    return application


app = create_app()
