"""Evaluation-only serving identities and injected, replaceable pipeline boundaries."""

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from opentelemetry import trace
from pydantic import JsonValue

from adaptive_llm.contracts import (
    Event,
    InferenceRequest,
    InferenceResponse,
    PolicyDecision,
    RagOptions,
    Region,
    uid,
)
from adaptive_llm.events.outbox import OutboxRow, OutboxStats
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.gateway.service import InferenceService
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.policy import ProcessingRedactor
from adaptive_llm.providers import Provider, ProviderRequest, ProviderResult
from adaptive_llm.rag import IndexedChunk, LocalRetriever, RetrievalResult, Retriever
from adaptive_llm.routing import PriceList, Router
from adaptive_llm.routing.tasks import RulesClassifier, TaskClassifier
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore
from adaptive_llm.validation import Validator


@dataclass(frozen=True)
class Case:
    item_id: str
    tenant_id: str
    request: InferenceRequest = field(repr=False)
    corpus: tuple[IndexedChunk, ...] = field(default=(), repr=False)
    target: str = field(default="", repr=False)
    expected_facts: tuple[str, ...] = field(default=(), repr=False)
    expected_citations: frozenset[tuple[str, str]] = frozenset()
    prohibited: tuple[str, ...] = field(default=(), repr=False)
    expect_json: bool = False
    expect_json_fields: dict[str, JsonValue] = field(default_factory=dict, repr=False)
    expect_json_text_match: dict[str, str | None] = field(default_factory=dict, repr=False)
    critical: bool = False
    category: str = "general"
    segments: tuple[str, ...] = ()
    k: int = 3


@dataclass(frozen=True)
class Outcome:
    response: InferenceResponse | None = field(repr=False)
    retrieval: RetrievalResult | None = field(repr=False)
    latency_ms: float
    error: bool = False
    cost_micros: int = 0


class Runner(Protocol):
    async def run(self, case: Case) -> Outcome: ...


class DiscardEvents:
    """No evaluation case event may reach an outbox, sink, dispatcher or tenant metric."""

    def emit(self, event: Event) -> None:
        pass

    def enqueue(self, events: Sequence[Event], pending_limit: int) -> int:
        return 0

    def claim(self, at: datetime) -> AbstractContextManager[OutboxRow | None]:
        return nullcontext(None)

    def next_due(self, at: datetime) -> OutboxRow | None:
        return None

    def finish(
        self,
        event_id: str,
        state: Literal["pending", "delivered", "dead"],
        attempts: int,
        next_attempt_at: datetime,
        error: Literal["sink_unavailable", "invalid_event"] | None,
    ) -> None:
        pass

    def stats(self, at: datetime) -> OutboxStats:
        return OutboxStats(pending=0, dead=0, lag_seconds=0)

    def dead_letters(self, tenant_id: str) -> list[tuple[str, str, str]]:
        return []

    def redeliver(self, event_id: str, at: datetime) -> bool:
        return False


@contextmanager
def isolated_persistence(source: Persistence) -> Iterator[Persistence]:
    """One private SQLite connection per evaluation, released on success or failure."""
    database = SQLiteDatabase(Path("."), in_memory=True)
    try:
        discard = DiscardEvents()
        yield Persistence(
            SQLiteMetadataStore(database),
            SQLitePayloadStore(database),
            source.cipher,
            source.redactor,
            source.keyring,
            outbox=discard,
            fallback_sink=discard,
            metrics=InProcessMetrics(),
            replay_capacity=source.replay_capacity,
            replay_ttl_seconds=source.replay_ttl_seconds,
            retention_seconds=source.retention_seconds,
            clock=source.clock,
        )
    finally:
        database.close()


class EvaluationPolicy:
    def __init__(self, application_id: str = "evaluation", residency: Region = "local") -> None:
        self.application_id, self.residency = application_id, residency

    def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
        return PolicyDecision(
            policy_version="evaluation-policy-1",
            processing_allowed=application_id == self.application_id
            and identity.application_ids == frozenset({self.application_id}),
            residency=self.residency,
            retention_seconds=3600,
            evaluation_allowed=True,
            content_logging_allowed=False,
            training_allowed=False,
        )


class CapturingRetriever:
    def __init__(self, delegate: Retriever) -> None:
        self.delegate = delegate
        self.result: RetrievalResult | None = None

    async def retrieve(
        self,
        query: str,
        identity: Identity,
        application_id: str,
        options: RagOptions,
        interaction_id: str,
        *,
        residency: Region,
    ) -> RetrievalResult:
        self.result = await self.delegate.retrieve(
            query, identity, application_id, options, interaction_id, residency=residency
        )
        return self.result


class MeteredProvider:
    def __init__(self, provider: Provider, prices: PriceList) -> None:
        self.provider, self.prices = provider, prices
        self.cost_micros = 0

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        result = await self.provider.generate(request)
        self.cost_micros += self.prices.estimate(
            result.usage.input_tokens, result.usage.output_tokens
        )
        return result


class PipelineRunner:
    def __init__(
        self,
        persistence: Persistence,
        operator: Identity,
        provider: Provider,
        router: Router,
        validator: Validator,
        prices: PriceList,
        classifier: TaskClassifier | None = None,
        residency: Region = "local",
    ) -> None:
        self.persistence, self.operator = persistence, operator
        self.provider, self.router, self.validator = provider, router, validator
        self.prices = prices
        self.classifier = classifier or RulesClassifier()
        self.residency = residency

    async def run(self, case: Case) -> Outcome:
        if (
            self.operator.key_class != "operator"
            or case.tenant_id not in self.operator.dataset_tenants
        ):
            raise GatewayError(403, "evaluation_tenant_forbidden")
        if (
            case.request.application_id != "evaluation"
            and case.request.application_id not in self.operator.application_ids
        ):
            raise GatewayError(403, "application_forbidden")
        identity = Identity(
            case.tenant_id,
            frozenset({case.request.application_id}),
            self.operator.environment,
            self.persistence.keyring.pseudonym(case.tenant_id, "evaluation"),
        )
        retriever = CapturingRetriever(
            LocalRetriever.from_chunks(case.corpus, self.persistence.keyring)
        )
        metered = MeteredProvider(self.provider, self.prices)
        service = InferenceService(
            keyring=self.persistence.keyring,
            replay_capacity=1,
            policy=EvaluationPolicy(case.request.application_id, self.residency),
            redactor=ProcessingRedactor(),
            retriever=retriever,
            router=self.router,
            provider=metered,
            validator=self.validator,
            tracer=trace.get_tracer("adaptive_llm.evaluation", "1.0"),
            persistence=self.persistence,
            classifier=self.classifier,
        )
        request = case.request.model_copy(update={"request_id": uid()})
        start = perf_counter()
        response: InferenceResponse | None = None
        try:
            response = await service.infer(request, identity)
        except GatewayError:
            # Failures count as zero scores; provider/validation bodies never enter reports.
            pass
        return Outcome(
            response,
            retriever.result,
            (perf_counter() - start) * 1000,
            response is None,
            metered.cost_micros,
        )
