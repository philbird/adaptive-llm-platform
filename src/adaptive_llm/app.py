"""Health-only scaffold. No user-content endpoints or persistence are enabled."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    stage: Literal["scaffold"] = "scaffold"
    inference_enabled: Literal[False] = False


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.ready = True
    yield
    application.state.ready = False


def create_app() -> FastAPI:
    application = FastAPI(
        title="Adaptive LLM Specialisation Platform",
        version="0.1.0",
        description="Scaffolding only. Inference, persistence and training are not enabled.",
        lifespan=lifespan,
    )

    @application.get("/healthz", response_model=Health, tags=["operations"])
    async def health() -> Health:
        return Health()

    return application


app = create_app()
