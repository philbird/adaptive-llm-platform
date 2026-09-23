"""Local serving with transactional metadata and encrypted payload persistence."""

import asyncio
import hashlib
import hmac
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from adaptive_llm.contracts import (
    Environment,
    InferenceRequest,
    InferenceResponse,
    SubjectDeletionInput,
)
from adaptive_llm.events import EventSink, InMemoryEventSink
from adaptive_llm.events.outbox import Dispatcher, OutboxBackoff, OutboxStore
from adaptive_llm.gateway.identity import (
    Authenticator,
    GatewayError,
    Identity,
    IdentityConfig,
    Keyring,
    LocalAuthenticator,
)
from adaptive_llm.gateway.service import InferenceService
from adaptive_llm.metrics import InProcessMetrics, Metrics, RequestMetrics
from adaptive_llm.policy import LocalPolicyEngine, PolicyEngine, ProcessingRedactor
from adaptive_llm.policy.persistence import LocalPersistenceRedactor, PersistenceRedactor
from adaptive_llm.providers import FakeProvider, Provider
from adaptive_llm.rag import LocalRetriever, Retriever
from adaptive_llm.routing import FoundationRouter, Router
from adaptive_llm.storage import MetadataStore, PayloadStore
from adaptive_llm.storage.crypto import PayloadCipher, load_keyring
from adaptive_llm.storage.outbox import SQLiteOutboxStore
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore
from adaptive_llm.validation import LocalValidator, Validator

ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    return ROOT / ".local"


@dataclass(frozen=True)
class Settings:
    inference_enabled: bool = True
    identity_path: Path = ROOT / "configs/identity/local.json"
    policy_path: Path = ROOT / "configs/policy/local.json"
    routing_path: Path = ROOT / "configs/routing/local.json"
    documents_path: Path = ROOT / "tests/fixtures/documents.json"
    event_capacity: int = 4096
    outbox_backoff: OutboxBackoff = field(default_factory=OutboxBackoff)
    outbox_pending_limit: int = 100_000
    outbox_dispatch_enabled: bool = True
    outbox_store: OutboxStore | None = None
    metrics: Metrics | None = None
    payload_keys: Mapping[str, bytes] | None = field(default=None, repr=False)
    payload_key_version: str = "local-1"
    payload_keyring_path: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["PAYLOAD_KEYRING"]) if os.environ.get("PAYLOAD_KEYRING") else None
        ),
        repr=False,
    )
    replay_capacity: int = 10_000
    data_dir: Path = field(default_factory=lambda: _default_data_dir())
    environment: Environment = "local"
    migrate_on_startup: bool = True
    replay_ttl_seconds: int = 86_400
    retention_seconds: int | None = None
    payload_key: bytes | None = field(default=None, repr=False)
    metadata_store: MetadataStore | None = None
    payload_store: PayloadStore | None = None
    persistence_redactor: PersistenceRedactor | None = None
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
        if self.payload_keyring_path is not None:
            keys, current = load_keyring(self.payload_keyring_path)
            object.__setattr__(self, "payload_keys", keys)
            object.__setattr__(self, "payload_key_version", current)
        if self.outbox_pending_limit < 1:
            raise ValueError("invalid_outbox_pending_limit")
        if self.replay_capacity < 1:
            raise ValueError("invalid_replay_capacity")
        if self.replay_ttl_seconds < 1 or (
            self.retention_seconds is not None and self.retention_seconds < 1
        ):
            raise ValueError("invalid_retention")
        if (self.metadata_store is None) != (self.payload_store is None):
            raise ValueError("storage_pair_required")
        if self.secret is None:
            config = IdentityConfig.model_validate_json(self.identity_path.read_text())
            object.__setattr__(self, "secret", config.local_secret.encode("utf-8"))
        if self.payload_keys is not None:
            PayloadCipher(self.payload_keys, self.payload_key_version)
            return
        if self.payload_key is None:
            if self.environment != "local":
                raise ValueError("payload_key_required")
            assert self.secret is not None
            # Local development only. Production obtains its independent data key from a KMS.
            object.__setattr__(
                self,
                "payload_key",
                hmac.new(self.secret, b"payload-encryption-local-v1", hashlib.sha256).digest(),
            )
        if self.payload_key is None or len(self.payload_key) != 32:
            raise ValueError("invalid_payload_key")


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    stage: Literal["milestone-1-local"] = "milestone-1-local"
    inference_enabled: bool
    outbox_pending: int = 0
    dead_letters: int = 0


def _start_inference(application: FastAPI, settings: Settings) -> None:
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
    metadata, payloads = settings.metadata_store, settings.payload_store
    if metadata is None or payloads is None:
        database = SQLiteDatabase(
            settings.data_dir,
            settings.environment,
            migrate_on_startup=settings.migrate_on_startup,
        )
        application.state.database = database
        metadata, payloads = SQLiteMetadataStore(database), SQLitePayloadStore(database)
    outbox = settings.outbox_store
    if outbox is None:
        if not isinstance(metadata, SQLiteMetadataStore):
            raise ValueError("outbox_store_required")
        outbox = SQLiteOutboxStore(metadata.database)
    metrics: Metrics = application.state.metrics
    key_material = (
        settings.payload_keys if settings.payload_keys is not None else settings.payload_key
    )
    assert key_material is not None
    persistence = Persistence(
        metadata,
        payloads,
        PayloadCipher(key_material, settings.payload_key_version),
        settings.persistence_redactor or LocalPersistenceRedactor(),
        keyring,
        outbox=outbox,
        metrics=metrics,
        fallback_sink=events,
        outbox_pending_limit=settings.outbox_pending_limit,
        replay_ttl_seconds=settings.replay_ttl_seconds,
        replay_capacity=settings.replay_capacity,
        retention_seconds=settings.retention_seconds,
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
        tracer=settings.tracer
        if settings.tracer is not None
        else trace.get_tracer("adaptive_llm.gateway", "1.0"),
        persistence=persistence,
    )
    application.state.outbox = outbox
    application.state.dispatcher = Dispatcher(
        outbox, events, metrics, backoff=settings.outbox_backoff, tracer=settings.tracer
    )
    application.state.dispatcher.refresh_metrics()
    application.state.events = events
    application.state.inference = service
    application.state.metadata = metadata
    application.state.payloads = payloads
    application.state.persistence = persistence

    application.state.authenticator = authenticator
    application.state.keyring = keyring


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings: Settings = application.state.settings
    application.state.ready = False
    dispatcher_task: asyncio.Task[None] | None = None
    try:
        if settings.inference_enabled:
            _start_inference(application, settings)
            if settings.outbox_dispatch_enabled:
                dispatcher_task = asyncio.create_task(application.state.dispatcher.run())
        application.state.ready = True
        yield
    finally:
        application.state.ready = False
        if dispatcher_task is not None:
            application.state.dispatcher.stop()
            await dispatcher_task
        if hasattr(application.state, "database"):
            application.state.database.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    application = FastAPI(
        title="Adaptive LLM Specialisation Platform",
        version="0.1.0",
        description="Local foundation inference with encrypted, tenant-scoped persistence.",
        lifespan=lifespan,
    )

    application.state.settings = settings
    application.state.metrics = (
        settings.metrics if settings.metrics is not None else InProcessMetrics()
    )

    application.add_middleware(RequestMetrics, metrics=application.state.metrics)

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
        # Cached gauges keep health responsive while a storage transaction is busy.
        metrics: Metrics = application.state.metrics
        return Health(
            inference_enabled=settings.inference_enabled,
            outbox_pending=int(metrics.get("outbox_pending")),
            dead_letters=int(metrics.get("outbox_dead")),
        )

    if settings.inference_enabled:

        def authenticate(
            authorization: Annotated[str | None, Header()] = None,
            x_subject: Annotated[str | None, Header()] = None,
        ) -> Identity:
            authenticator: Authenticator = application.state.authenticator
            identity = authenticator.authenticate(authorization, x_subject)
            if identity.environment != settings.environment:
                raise GatewayError(403, "environment_forbidden")
            return identity

        @application.post("/v1/inference", response_model=InferenceResponse, tags=["inference"])
        async def inference(
            body: InferenceRequest,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> InferenceResponse:
            service: InferenceService = application.state.inference
            return await service.infer(body, identity)

        @application.delete("/v1/privacy/interactions/{interaction_id}", status_code=204)
        async def delete_one(
            interaction_id: str, identity: Annotated[Identity, Depends(authenticate)]
        ) -> Response:
            persistence: Persistence = application.state.persistence
            deleted = await asyncio.to_thread(
                persistence.delete,
                identity.tenant_id,
                interaction_id,
                identity.subject_id_pseudonymous,
            )
            if deleted is None:
                raise GatewayError(404, "interaction_not_found")
            return Response(status_code=204)

        @application.post("/v1/privacy/subjects/deletion-requests")
        async def delete_subject(
            body: SubjectDeletionInput, identity: Annotated[Identity, Depends(authenticate)]
        ) -> dict[str, int]:
            persistence: Persistence = application.state.persistence
            keyring: Keyring = application.state.keyring
            pseudonym = keyring.pseudonym(identity.tenant_id, body.subject)
            _, deleted = await asyncio.to_thread(
                persistence.delete_subject,
                identity.tenant_id,
                pseudonym,
                identity.subject_id_pseudonymous,
            )
            return {"deleted": len(deleted)}

    return application


app = create_app()
