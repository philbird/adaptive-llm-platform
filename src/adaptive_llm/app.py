"""Local in-memory vertical slice with replaceable serving boundaries."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from adaptive_llm.contracts import InferenceRequest, InferenceResponse
from adaptive_llm.events import EventSink, InMemoryEventSink
from adaptive_llm.gateway.identity import (
    Authenticator,
    GatewayError,
    Identity,
    IdentityConfig,
    Keyring,
    LocalAuthenticator,
)
from adaptive_llm.gateway.service import InferenceService
from adaptive_llm.policy import LocalPolicyEngine, PolicyEngine, ProcessingRedactor
from adaptive_llm.providers import FakeProvider, Provider
from adaptive_llm.rag import LocalRetriever, Retriever
from adaptive_llm.routing import FoundationRouter, Router
from adaptive_llm.validation import LocalValidator, Validator

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    inference_enabled: bool = True
    identity_path: Path = ROOT / "configs/identity/local.json"
    policy_path: Path = ROOT / "configs/policy/local.json"
    routing_path: Path = ROOT / "configs/routing/local.json"
    documents_path: Path = ROOT / "tests/fixtures/documents.json"
    event_capacity: int = 4096
    replay_capacity: int = 10_000
    secret: bytes | None = field(default=None, repr=False)
    authenticator: Authenticator | None = None
    policy: PolicyEngine | None = None
    redactor: ProcessingRedactor | None = None
    retriever: Retriever | None = None
    router: Router | None = None
    provider: Provider | None = None
    validator: Validator | None = None
    events: EventSink | None = None
    tracer: Tracer | None = None

    def __post_init__(self) -> None:
        if self.replay_capacity < 1:
            raise ValueError("invalid_replay_capacity")
        if self.secret is None:
            config = IdentityConfig.model_validate_json(self.identity_path.read_text())
            object.__setattr__(self, "secret", config.local_secret.encode("utf-8"))


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    stage: Literal["slice-1a"] = "slice-1a"
    inference_enabled: bool


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.ready = True
    yield
    application.state.ready = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    application = FastAPI(
        title="Adaptive LLM Specialisation Platform",
        version="0.1.0",
        description="In-memory foundation inference using local synthetic identity and documents.",
        lifespan=lifespan,
    )

    @application.exception_handler(GatewayError)
    async def gateway_error(request: Request, error: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code}},
            headers={"WWW-Authenticate": "Bearer"} if error.status_code == 401 else None,
        )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # The default validation response echoes rejected input, which can contain secrets.
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request"}})

    @application.get("/healthz", response_model=Health, tags=["operations"])
    async def health() -> Health:
        return Health(inference_enabled=settings.inference_enabled)

    if settings.inference_enabled:
        assert settings.secret is not None
        keyring = Keyring(settings.secret)
        authenticator = (
            settings.authenticator
            if settings.authenticator is not None
            else LocalAuthenticator(settings.identity_path, keyring)
        )
        events = (
            settings.events
            if settings.events is not None
            else InMemoryEventSink(settings.event_capacity)
        )
        service = InferenceService(
            keyring=keyring,
            replay_capacity=settings.replay_capacity,
            policy=settings.policy
            if settings.policy is not None
            else LocalPolicyEngine(settings.policy_path),
            redactor=settings.redactor if settings.redactor is not None else ProcessingRedactor(),
            retriever=settings.retriever
            if settings.retriever is not None
            else LocalRetriever(settings.documents_path, keyring),
            router=settings.router
            if settings.router is not None
            else FoundationRouter(settings.routing_path),
            provider=settings.provider if settings.provider is not None else FakeProvider(),
            validator=settings.validator if settings.validator is not None else LocalValidator(),
            events=events,
            tracer=settings.tracer
            if settings.tracer is not None
            else trace.get_tracer("adaptive_llm.gateway", "1.0"),
        )
        application.state.events = events
        application.state.inference = service

        def authenticate(
            authorization: Annotated[str | None, Header()] = None,
            x_subject: Annotated[str | None, Header()] = None,
        ) -> Identity:
            return authenticator.authenticate(authorization, x_subject)

        @application.post("/v1/inference", response_model=InferenceResponse, tags=["inference"])
        async def inference(
            body: InferenceRequest,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> InferenceResponse:
            return await service.infer(body, identity)

    return application


app = create_app()
