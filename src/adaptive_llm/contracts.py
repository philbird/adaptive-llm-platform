"""Provider-neutral v1 contracts. Money is integer USD micros; unknown counts are null."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from secrets import randbits
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


def uid() -> str:
    """RFC 9562 UUIDv7, time sortable (random order within a millisecond)."""
    value = (int(time.time() * 1000) << 80) | (7 << 76) | (randbits(12) << 64)
    value |= (2 << 62) | randbits(62)
    return str(UUID(int=value))


def now() -> datetime:
    return datetime.now(UTC)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"


Identifier = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")]
Content = Annotated[
    str, Field(max_length=32_000, json_schema_extra={"classification": "confidential"})
]


class Message(Contract):
    role: Literal["user", "assistant"]
    content: Content


class RagOptions(Contract):
    enabled: bool = True
    index_id: Identifier = "support-kb"


class ResponseFormat(Contract):
    type: Literal["text", "json_object"] = "text"


class RoutingOptions(Contract):
    mode: Literal["auto", "foundation"] = "auto"
    max_cost_micros: int = Field(default=20_000, ge=0, le=1_000_000)
    deadline_ms: int = Field(default=5_000, ge=50, le=30_000)


class InferenceRequest(Contract):
    request_id: Identifier
    messages: list[Message] = Field(min_length=1, max_length=32)
    application_id: Identifier = "support-assistant"
    rag: RagOptions = Field(default_factory=RagOptions)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    routing: RoutingOptions = Field(default_factory=RoutingOptions)
    locale: Literal["en-GB", "en-US"] = "en-GB"
    max_output_tokens: int = Field(default=512, ge=1, le=2_048)
    stream: bool = False

    @model_validator(mode="after")
    def bounded_messages(self) -> InferenceRequest:
        if sum(len(m.content) for m in self.messages) > 32_000:
            raise ValueError("total message length exceeds limit")
        if self.messages[-1].role != "user":
            raise ValueError("last message must be a user message")
        return self


class PolicyDecision(Contract):
    policy_version: str
    processing_allowed: bool = True
    content_logging_allowed: bool = False
    evaluation_allowed: bool = False
    human_review_allowed: bool = False
    training_allowed: bool = False
    retention_seconds: int = Field(ge=1)
    residency: str = "local"
    redaction_version: str = "regex-local-1"
    redaction_counts: dict[str, int] = Field(default_factory=dict)


class Task(Contract):
    label: str
    language: str = "en"
    risk_tier: Literal["low", "medium", "high"] = "medium"
    classifier_version: str = "rules-1"
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str]


class Chunk(Contract):
    tenant_id: Identifier
    environment: Literal["local"] = "local"
    region: Literal["local"] = "local"
    allowed_applications: list[Identifier]
    index_id: Identifier = "support-kb"
    document_id: Identifier
    document_version: str
    document_family: Identifier
    chunk_id: Identifier
    content: Content
    licence_class: Literal["synthetic", "internal-approved", "unknown"] = "unknown"


class ChunkEvidence(Contract):
    document_id: str
    document_version: str
    chunk_id: str
    rank_retrieved: int
    retrieval_score: float
    supplied_to_model: bool
    context_position: int | None
    token_count: int
    content_hash: str
    licence_class: str


class RetrievalRun(Contract):
    retrieval_run_id: str = Field(default_factory=uid)
    interaction_id: str
    index_id: str
    index_version: str
    embedding_model: str | None = None
    reranker_model: str | None = None
    query_hash: str
    latency_ms: float
    candidates: list[ChunkEvidence]


class Candidate(Contract):
    model_deployment_id: str
    eligible: bool
    estimated_cost_micros: int
    predicted_quality: float | None = None
    ood_score: float | None = None
    reason_codes: list[str]


class RouteDecision(Contract):
    route_decision_id: str = Field(default_factory=uid)
    interaction_id: str
    router_version: str = "foundation-only-1"
    candidates: list[Candidate]
    selected_model_deployment_id: str
    fallback_deployment_ids: list[str] = Field(default_factory=list)
    decision_latency_ms: float
    policy_constraints: list[str]


class Usage(Contract):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    source: Literal["provider_reported", "locally_estimated"]
    tokenizer: str

    @model_validator(mode="after")
    def cached_is_subset(self) -> Usage:
        if self.cached_input_tokens is not None and self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input cannot exceed total input")
        return self


class Citation(Contract):
    document_id: str
    chunk_id: str


class ValidationCheck(Contract):
    name: str
    passed: bool


class Validation(Contract):
    validator_version: str = "baseline-validator-1"
    passed: bool
    checks: list[ValidationCheck]


class GenerationAttempt(Contract):
    attempt_id: str = Field(default_factory=uid)
    interaction_id: str
    attempt_number: int = 1
    model_provider: str
    model_id: str
    model_version: str
    deployment_id: str
    adapter_id: str | None = None
    usage: Usage | None = None
    output_ref: str | None = None
    output_hash: str | None = None
    first_token_latency_ms: float | None = None
    total_latency_ms: float
    estimated_cost_micros: int | None = None
    finish_reason: Literal["stop", "length", "error"]
    validation: Validation | None = None
    error_code: str | None = None


class Interaction(Contract):
    interaction_id: str
    trace_id: str
    request_id: str
    tenant_id: str
    subject_id_pseudonymous: str
    application_id: str
    environment: Literal["local"] = "local"
    started_at: datetime
    completed_at: datetime
    task: Task
    policy: PolicyDecision
    input_ref: str | None = None
    input_hash: str
    retrieval_run_id: str | None
    route_decision_id: str | None
    generation_attempt_ids: list[str]
    final_attempt_id: str | None
    status: Literal["completed", "failed"]
    error_code: str | None = None


class RouteSummary(Contract):
    fallback_used: bool = False


class InferenceResponse(Contract):
    interaction_id: str
    trace_id: str
    model_deployment_id: str
    content: Content
    citations: list[Citation]
    usage: Usage
    estimated_cost_micros: int
    route: RouteSummary = Field(default_factory=RouteSummary)
    finish_reason: Literal["stop", "length"]


class Started(Contract):
    interaction_id: str
    application_id: str
    policy_version: str


class FeedbackInput(Contract):
    request_id: Identifier
    score: int = Field(ge=1, le=5)


class Feedback(Contract):
    interaction_id: str
    feedback_id: str = Field(default_factory=uid)
    actor_id: str
    source: Literal["user"] = "user"
    label_type: Literal["rubric"] = "rubric"
    rubric_version: str = "user-satisfaction-1"
    score: int = Field(ge=1, le=5)
    created_at: datetime = Field(default_factory=now)


EventData = Started | RetrievalRun | RouteDecision | GenerationAttempt | Interaction | Feedback


class Event(Contract):
    event_id: str = Field(default_factory=uid)
    event_type: Literal[
        "interaction.started.v1",
        "retrieval.completed.v1",
        "route.decided.v1",
        "generation.completed.v1",
        "generation.failed.v1",
        "interaction.completed.v1",
        "feedback.recorded.v1",
    ]
    occurred_at: datetime = Field(default_factory=now)
    producer: Literal["gateway"] = "gateway"
    tenant_id: str
    trace_id: str
    data: EventData

    @model_validator(mode="after")
    def matching_event_data(self) -> Event:
        kinds = {
            "interaction.started.v1": Started,
            "retrieval.completed.v1": RetrievalRun,
            "route.decided.v1": RouteDecision,
            "generation.completed.v1": GenerationAttempt,
            "generation.failed.v1": GenerationAttempt,
            "interaction.completed.v1": Interaction,
            "feedback.recorded.v1": Feedback,
        }
        if not isinstance(self.data, kinds[self.event_type]):
            raise ValueError("event type and payload do not match")
        return self
