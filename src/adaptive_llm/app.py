"""Local serving with transactional metadata and encrypted payload persistence."""

import asyncio
import hashlib
import hmac
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from adaptive_llm.contracts import (
    CanaryReport,
    CorrectionInput,
    DatasetManifest,
    DatasetSpecification,
    Environment,
    EvaluationInput,
    EvaluationReport,
    Feedback,
    FeedbackInput,
    InferenceRequest,
    InferenceResponse,
    ModelManifest,
    OperatorNote,
    PromotionRequest,
    RouteControlNote,
    RoutePolicy,
    ShadowReport,
    SubjectDeletionInput,
    TrainingJob,
    TrainingJobSpecification,
)
from adaptive_llm.datasets.artifacts import DatasetApprovals
from adaptive_llm.datasets.builder import DatasetBuilder, LocalDatasetBuilder, code_revision
from adaptive_llm.datasets.sources import LocalSourceResolver, SourceResolver
from adaptive_llm.evaluation.data import LocalDatasetReader
from adaptive_llm.evaluation.service import EvaluationDeployment, Evaluator, LocalEvaluator
from adaptive_llm.evaluation.storage import EvaluationStore, SQLiteEvaluationStore
from adaptive_llm.events import EventSink, InMemoryEventSink
from adaptive_llm.events.outbox import Dispatcher, OutboxBackoff, OutboxStore
from adaptive_llm.gateway.feedback import FeedbackService
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
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.registry.sqlite import SQLiteModelRegistry
from adaptive_llm.routing import FoundationRouter, Router
from adaptive_llm.routing.canary import CanaryMonitor
from adaptive_llm.routing.chain import ChainPlanner
from adaptive_llm.routing.control import SQLiteRoutePolicies
from adaptive_llm.routing.live import LivePlanner
from adaptive_llm.routing.shadow import (
    RegistrySpecialists,
    ShadowWork,
    ShadowWorker,
    SpecialistLoader,
)
from adaptive_llm.routing.train import CounterfactualRows
from adaptive_llm.storage import MetadataStore, PayloadStore, StorageError
from adaptive_llm.storage.crypto import PayloadCipher, load_keyring
from adaptive_llm.storage.outbox import SQLiteOutboxStore
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.storage.sqlite import (
    SQLiteDatabase,
    SQLiteMetadataStore,
    SQLitePayloadStore,
    control_database,
)
from adaptive_llm.training import Trainer
from adaptive_llm.training.fake import FakeTrainer
from adaptive_llm.training.service import TrainingOrchestrator
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
    golden_dir: Path = ROOT / "tests/fixtures/golden"
    dataset_builder: DatasetBuilder | None = None
    dataset_sources: SourceResolver | None = None
    trainer: Trainer | None = None
    training_backend: Literal["fake", "lora"] = "fake"
    training_memory_limit_bytes: int = 2_000_000_000
    training_time_limit_seconds: float = 300
    registry: ModelRegistry | None = None
    evaluator: Evaluator | None = None
    evaluation_store: EvaluationStore | None = None
    evaluation_deployments: Mapping[str, EvaluationDeployment] | None = None
    evaluation_events: EventSink | None = None
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
    chain_planner: ChainPlanner | None = None
    specialist_loader: SpecialistLoader | None = None
    shadow_queue_capacity: int = 32
    shadow_timeout_seconds: float = 30
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
    circuit_breakers: dict[str, str] = {}
    kill_switch: bool = False
    live_planner_failures: int = 0
    shadow_queue_depth: int = 0


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
    policy = (
        settings.policy if settings.policy is not None else LocalPolicyEngine(settings.policy_path)
    )
    service = InferenceService(
        keyring=keyring,
        replay_capacity=settings.replay_capacity,
        policy=policy,
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
    application.state.feedback = FeedbackService(persistence, policy)
    revision = code_revision()
    application.state.datasets = settings.dataset_builder or LocalDatasetBuilder(
        persistence,
        policy,
        settings.dataset_sources or LocalSourceResolver(settings.documents_path),
        settings.data_dir,
        settings.golden_dir,
        revision=revision,
    )
    application.state.dataset_approvals = DatasetApprovals(
        application.state.datasets, persistence, settings.data_dir
    )
    evaluation_database = control_database(
        settings.data_dir,
        settings.environment,
        migrate_on_startup=settings.migrate_on_startup,
    )
    application.state.evaluation_database = evaluation_database
    evaluation_outbox = SQLiteOutboxStore(evaluation_database)
    evaluation_events = settings.evaluation_events or InMemoryEventSink(settings.event_capacity)
    application.state.evaluation_outbox = evaluation_outbox
    application.state.evaluation_events = evaluation_events
    application.state.evaluation_dispatcher = Dispatcher(
        evaluation_outbox,
        evaluation_events,
        InProcessMetrics(),
        backoff=settings.outbox_backoff,
        tracer=settings.tracer,
    )
    evaluation_store = settings.evaluation_store or SQLiteEvaluationStore(
        evaluation_database,
        evaluation_outbox,
        keyring,
        settings.data_dir,
        settings.outbox_pending_limit,
    )
    registry = settings.registry or SQLiteModelRegistry(
        evaluation_database,
        evaluation_outbox,
        keyring,
        evaluation_store,
        settings.data_dir,
        settings.outbox_pending_limit,
    )
    application.state.registry = registry
    foundation = FoundationRouter(settings.routing_path).deployment
    tenants = (
        frozenset(
            entry.tenant_id
            for entry in IdentityConfig.model_validate_json(
                settings.identity_path.read_text()
            ).keys.values()
        )
        if settings.authenticator is None
        else frozenset()
    )
    routes = SQLiteRoutePolicies(
        evaluation_database,
        evaluation_outbox,
        registry,
        settings.environment,
        foundation.model_deployment_id,
        tenants,
        keyring,
    )
    application.state.route_policies = routes
    if isinstance(registry, SQLiteModelRegistry):
        registry.progression_gate = routes.progression_gate
    if isinstance(application.state.datasets, LocalDatasetBuilder):
        application.state.datasets.routing_examples = CounterfactualRows(
            application.state.datasets,
            evaluation_database,
            evaluation_store,
            settings.data_dir,
            persistence.cipher,
            keyring,
            foundation.model_deployment_id,
        )
    service.live_outcomes = routes
    application.state.canary_monitor = CanaryMonitor(
        routes,
        Identity(
            tenant_id="control",
            application_ids=frozenset(),
            environment=settings.environment,
            subject_id_pseudonymous="automatic-rollback",
            key_class="operator",
            dataset_tenants=tenants,
        ),
    )
    service.chain_planner = settings.chain_planner or LivePlanner(
        routes,
        settings.provider if settings.provider is not None else FakeProvider(),
        metrics,
        registry,
        settings.specialist_loader
        or RegistrySpecialists(
            registry,
            foundation,
            keyring,
            settings.data_dir,
            tenants,
            allowed_states=frozenset({"canary", "production"}),
        ),
        keyring,
        settings.data_dir,
        tenants,
        service.breakers,
    )
    application.state.shadow = ShadowWorker(
        routes,
        settings.specialist_loader
        or RegistrySpecialists(
            registry,
            foundation,
            keyring,
            settings.data_dir,
            tenants,
        ),
        policy,
        settings.validator if settings.validator is not None else LocalValidator(),
        persistence,
        service.breakers,
        metrics,
        capacity=settings.shadow_queue_capacity,
        timeout_seconds=settings.shadow_timeout_seconds,
    )
    if isinstance(evaluation_store, SQLiteEvaluationStore) and isinstance(
        registry, SQLiteModelRegistry
    ):
        evaluation_store.on_publish = registry.record_evaluation
    trainer = settings.trainer
    if trainer is None:
        if settings.training_backend == "lora":
            from adaptive_llm.training.lora import LoraTrainer

            trainer = LoraTrainer(
                settings.data_dir,
                persistence.cipher,
                keyring,
                memory_limit_bytes=settings.training_memory_limit_bytes,
                time_limit_seconds=settings.training_time_limit_seconds,
            )
        else:
            trainer = FakeTrainer(settings.data_dir, persistence.cipher, keyring)
    application.state.training = TrainingOrchestrator(
        registry,
        application.state.datasets,
        policy,
        trainer,
        settings.data_dir,
        persistence.cipher,
        keyring,
        revision,
    )
    if settings.evaluator is not None:
        application.state.evaluations = settings.evaluator
    else:
        foundation = FoundationRouter(settings.routing_path).deployment
        deployments = settings.evaluation_deployments or {
            foundation.model_deployment_id: EvaluationDeployment(foundation, FakeProvider())
        }
        application.state.evaluations = LocalEvaluator(
            persistence,
            LocalDatasetReader(
                application.state.datasets, settings.data_dir, persistence.cipher, keyring
            ),
            evaluation_store,
            deployments,
            foundation.model_deployment_id,
            settings.golden_dir.parent,
            settings.routing_path,
            settings.validator if settings.validator is not None else LocalValidator(),
            revision,
            registry=registry,
            data_dir=settings.data_dir,
        )

    application.state.authenticator = authenticator
    application.state.keyring = keyring


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings: Settings = application.state.settings
    application.state.ready = False
    dispatcher_task: asyncio.Task[None] | None = None
    evaluation_dispatcher_task: asyncio.Task[None] | None = None
    training_task: asyncio.Task[None] | None = None
    shadow_task: asyncio.Task[None] | None = None
    canary_task: asyncio.Task[None] | None = None
    try:
        if settings.inference_enabled:
            _start_inference(application, settings)
            training_task = asyncio.create_task(application.state.training.worker())
            shadow_task = asyncio.create_task(application.state.shadow.run())
            canary_task = asyncio.create_task(application.state.canary_monitor.run())
            if settings.outbox_dispatch_enabled:
                dispatcher_task = asyncio.create_task(application.state.dispatcher.run())
                if hasattr(application.state, "evaluation_dispatcher"):
                    evaluation_dispatcher_task = asyncio.create_task(
                        application.state.evaluation_dispatcher.run()
                    )
        application.state.ready = True
        yield
    finally:
        application.state.ready = False
        if canary_task is not None:
            application.state.canary_monitor.stop()
            await canary_task
        if shadow_task is not None:
            application.state.shadow.stop()
            await shadow_task
        if training_task is not None:
            application.state.training.stop()
            await training_task
        if dispatcher_task is not None:
            application.state.dispatcher.stop()
            await dispatcher_task
        if evaluation_dispatcher_task is not None:
            application.state.evaluation_dispatcher.stop()
            await evaluation_dispatcher_task
        if hasattr(application.state, "evaluation_database"):
            application.state.evaluation_database.close()
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
        # Operational gauges are cached; route controls share the one-second pointer cache.
        metrics: Metrics = application.state.metrics
        killed = False
        if hasattr(application.state, "route_policies"):
            routes: SQLiteRoutePolicies = application.state.route_policies
            try:
                snapshot = await asyncio.to_thread(routes.snapshot)
                killed = snapshot.killed or bool(snapshot.policy and snapshot.policy.kill_switch)
            except Exception:
                killed = True
            metrics.gauge("kill_switch", int(killed))
        return Health(
            inference_enabled=settings.inference_enabled,
            outbox_pending=int(metrics.get("outbox_pending")),
            dead_letters=int(metrics.get("outbox_dead")),
            circuit_breakers=application.state.inference.breakers.states()
            if settings.inference_enabled and hasattr(application.state, "inference")
            else {},
            kill_switch=killed,
            live_planner_failures=int(metrics.get("live_planner_failures")),
            shadow_queue_depth=int(metrics.get("shadow_queue_depth")),
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
            background_tasks: BackgroundTasks,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> InferenceResponse:
            service: InferenceService = application.state.inference

            def schedule(work: ShadowWork) -> None:
                background_tasks.add_task(application.state.shadow.submit, work)

            return await service.infer(body, identity, schedule)

        @application.post("/v1/route-policies", response_model=RoutePolicy)
        async def create_route_policy(
            body: RoutePolicy,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> RoutePolicy:
            routes: SQLiteRoutePolicies = application.state.route_policies
            return await asyncio.to_thread(routes.create, body, identity)

        @application.post("/v1/route-policies/{policy_id}/activate", response_model=RoutePolicy)
        async def activate_route_policy(
            policy_id: str,
            body: RouteControlNote,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> RoutePolicy:
            routes: SQLiteRoutePolicies = application.state.route_policies
            return await asyncio.to_thread(routes.activate, policy_id, identity, body)

        @application.post("/v1/route-policies/kill-switch")
        async def engage_kill_switch(
            body: RouteControlNote,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> dict[str, bool]:
            routes: SQLiteRoutePolicies = application.state.route_policies
            await asyncio.to_thread(routes.switch, True, identity, body)
            return {"disabled": True}

        @application.delete("/v1/route-policies/kill-switch")
        async def release_kill_switch(
            body: RouteControlNote,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> dict[str, bool]:
            routes: SQLiteRoutePolicies = application.state.route_policies
            await asyncio.to_thread(routes.switch, False, identity, body)
            return {"disabled": False}

        @application.get("/v1/shadow/reports", response_model=ShadowReport)
        async def shadow_reports(
            policy: str,
            since: datetime,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> ShadowReport:
            routes: SQLiteRoutePolicies = application.state.route_policies
            return await asyncio.to_thread(routes.report, policy, since, identity)

        @application.get("/v1/canary/reports", response_model=CanaryReport)
        async def canary_reports(
            policy: str,
            since: datetime,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> CanaryReport:
            routes: SQLiteRoutePolicies = application.state.route_policies
            return await asyncio.to_thread(routes.canary_report, policy, since, identity)

        @application.post("/v1/interactions/{interaction_id}/feedback", response_model=Feedback)
        async def feedback(
            interaction_id: str,
            body: FeedbackInput,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> Feedback:
            service: FeedbackService = application.state.feedback
            try:
                return await asyncio.to_thread(service.record, identity, interaction_id, body)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "feedback_persistence_failed") from None

        @application.post("/v1/interactions/{interaction_id}/correction", response_model=Feedback)
        async def correction(
            interaction_id: str,
            body: CorrectionInput,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> Feedback:
            service: FeedbackService = application.state.feedback
            try:
                return await asyncio.to_thread(service.record, identity, interaction_id, body)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "feedback_persistence_failed") from None

        @application.post("/v1/datasets/builds", response_model=DatasetManifest)
        async def build_dataset(
            body: DatasetSpecification,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> DatasetManifest:
            builder: DatasetBuilder = application.state.datasets
            # The HTTP boundary enforces the key class even for injected builders.
            LocalDatasetBuilder._authorize(identity, body.tenant_ids)
            try:
                return await asyncio.to_thread(builder.build, body, identity)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "dataset_build_failed") from None

        @application.get(
            "/v1/datasets/{dataset_id}/versions/{version}", response_model=DatasetManifest
        )
        async def get_dataset(
            dataset_id: str,
            version: str,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> DatasetManifest:
            LocalDatasetBuilder._authorize(identity, [])
            builder: DatasetBuilder = application.state.datasets
            try:
                return await asyncio.to_thread(builder.get, dataset_id, version, identity)
            except StorageError:
                raise GatewayError(503, "dataset_read_failed") from None

        @application.post(
            "/v1/datasets/{dataset_id}/versions/{version}/approval", response_model=DatasetManifest
        )
        async def approve_dataset(
            dataset_id: str,
            version: str,
            body: OperatorNote,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> DatasetManifest:
            service: DatasetApprovals = application.state.dataset_approvals
            LocalDatasetBuilder._authorize(identity, [])
            try:
                return await asyncio.to_thread(
                    service.approve, dataset_id, version, identity, body.reason
                )
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "dataset_approval_failed") from None

        @application.post("/v1/training/jobs", response_model=TrainingJob)
        async def train(
            body: TrainingJobSpecification, identity: Annotated[Identity, Depends(authenticate)]
        ) -> TrainingJob:
            LocalDatasetBuilder._authorize(identity, [])
            service: TrainingOrchestrator = application.state.training
            try:
                return await asyncio.to_thread(service.submit, body, identity)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "training_failed") from None

        @application.post("/v1/training/jobs/{job_id}/cancel", response_model=TrainingJob)
        async def cancel_training_job(
            job_id: str, identity: Annotated[Identity, Depends(authenticate)]
        ) -> TrainingJob:
            registry: ModelRegistry = application.state.registry
            return await asyncio.to_thread(registry.cancel_job, job_id, identity)

        @application.get("/v1/training/jobs/{job_id}", response_model=TrainingJob)
        async def training_job(
            job_id: str, identity: Annotated[Identity, Depends(authenticate)]
        ) -> TrainingJob:
            LocalDatasetBuilder._authorize(identity, [])
            registry: ModelRegistry = application.state.registry
            result = await asyncio.to_thread(registry.job, job_id, identity)
            if result is None:
                raise GatewayError(404, "training_job_not_found")
            return result

        @application.post(
            "/v1/models/{model_version}/promotion-requests", response_model=ModelManifest
        )
        async def promote(
            model_version: str,
            body: PromotionRequest,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> ModelManifest:
            LocalDatasetBuilder._authorize(identity, [])
            if model_version != body.model_version:
                raise GatewayError(422, "model_version_mismatch")
            registry: ModelRegistry = application.state.registry
            try:
                return await asyncio.to_thread(registry.promote, body, identity)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "promotion_failed") from None

        @application.post(
            "/v1/deployments/{deployment_id}/rollback", response_model=list[ModelManifest]
        )
        async def rollback(
            deployment_id: str,
            body: OperatorNote,
            identity: Annotated[Identity, Depends(authenticate)],
        ) -> list[ModelManifest]:
            LocalDatasetBuilder._authorize(identity, [])
            registry: ModelRegistry = application.state.registry
            try:
                return await asyncio.to_thread(
                    registry.rollback, deployment_id, identity, body.reason
                )
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "rollback_failed") from None

        @application.post("/v1/evaluations", response_model=EvaluationReport)
        async def evaluate(
            body: EvaluationInput, identity: Annotated[Identity, Depends(authenticate)]
        ) -> EvaluationReport:
            LocalDatasetBuilder._authorize(identity, [])
            evaluator: Evaluator = application.state.evaluations
            try:
                return await asyncio.to_thread(evaluator.evaluate, body, identity)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "evaluation_failed") from None

        @application.get("/v1/evaluations/{evaluation_id}", response_model=EvaluationReport)
        async def get_evaluation(
            evaluation_id: str, identity: Annotated[Identity, Depends(authenticate)]
        ) -> EvaluationReport:
            LocalDatasetBuilder._authorize(identity, [])
            evaluator: Evaluator = application.state.evaluations
            try:
                return await asyncio.to_thread(evaluator.get, evaluation_id, identity)
            except GatewayError:
                raise
            except Exception:
                raise GatewayError(503, "evaluation_read_failed") from None

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
