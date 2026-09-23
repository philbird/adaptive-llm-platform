"""Provider-neutral v1 contracts.

Money is integer USD micros; unknown counts are null. Ingress contracts (client-supplied
bodies) reject unknown fields. Records (stored and emitted payloads) ignore unknown fields so
producers may add fields before consumers upgrade, as required by the specification's
backward-compatibility rule. Both reject an unknown schema_version.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from secrets import randbits
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


def uid() -> str:
    """RFC 9562 UUIDv7, time sortable (random order within a millisecond)."""
    value = (int(time.time() * 1000) << 80) | (7 << 76) | (randbits(12) << 64)
    value |= (2 << 62) | randbits(62)
    return str(UUID(int=value))


def now() -> datetime:
    return datetime.now(UTC)


class Contract(BaseModel):
    """Ingress contract: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"


class Record(Contract):
    """Stored or emitted payload: unknown fields are ignored (additive compatibility)."""

    model_config = ConfigDict(extra="ignore")


Identifier = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")]
Region = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[a-z0-9-]+$")]
Content = Annotated[
    str, Field(max_length=32_000, json_schema_extra={"classification": "confidential"})
]
Reference = Annotated[str, Field(min_length=1, max_length=512)]
MetadataValue = Annotated[str, Field(max_length=200)]

Environment = Literal["local", "development", "staging", "production"]
Producer = Literal[
    "gateway",
    "policy_redaction",
    "rag_orchestrator",
    "model_router",
    "event_collector",
    "dataset_builder",
    "training_orchestrator",
    "evaluation_service",
    "deployment_controller",
]
FinishReason = Literal[
    "stop", "length", "error", "cancelled", "deadline_exceeded", "content_filter"
]
HashScheme = Literal["sha256", "hmac-sha256"]
Modality = Literal["text"]


def _text_only() -> list[Modality]:
    return ["text"]


LifecycleState = Literal[
    "candidate", "evaluating", "approved", "shadow", "canary", "production", "deprecated", "revoked"
]


# --------------------------------------------------------------------------- inference ingress


class Message(Contract):
    role: Literal["user", "assistant"]
    content: Content


class RagOptions(Contract):
    enabled: bool = False
    index_id: Identifier | None = None

    @model_validator(mode="after")
    def index_required_when_enabled(self) -> RagOptions:
        if self.enabled and self.index_id is None:
            raise ValueError("rag.index_id is required when rag is enabled")
        return self


class ResponseFormat(Contract):
    type: Literal["text", "json_object"] = "text"


class RoutingOptions(Contract):
    mode: Literal["auto", "foundation"] = "auto"
    max_cost_micros: int = Field(default=20_000, ge=0, le=1_000_000)
    deadline_ms: int = Field(default=5_000, ge=50, le=30_000)


class InferenceRequest(Contract):
    """Client request.

    `request_id` is the idempotency key, scoped to the authenticated tenant and
    `application_id`. A replay within the idempotency window returns the original response
    and marks it as replayed; a replay with a different body is rejected.
    """

    request_id: Identifier
    messages: list[Message] = Field(min_length=1, max_length=32)
    application_id: Identifier
    rag: RagOptions = Field(default_factory=RagOptions)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    routing: RoutingOptions = Field(default_factory=RoutingOptions)
    metadata: dict[Identifier, MetadataValue] = Field(default_factory=dict, max_length=16)
    max_output_tokens: int = Field(default=512, ge=1, le=2_048)
    stream: bool = False

    @model_validator(mode="after")
    def bounded_messages(self) -> InferenceRequest:
        if sum(len(m.content) for m in self.messages) > 32_000:
            raise ValueError("total message length exceeds limit")
        if self.messages[-1].role != "user":
            raise ValueError("last message must be a user message")
        return self


class Citation(Contract):
    document_id: str
    chunk_id: str


class RouteSummary(Contract):
    fallback_used: bool = False


class Usage(Record):
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


class InferenceResponse(Contract):
    interaction_id: str
    trace_id: str
    model_deployment_id: str
    content: Content
    citations: list[Citation]
    usage: Usage
    estimated_cost_micros: int
    route: RouteSummary = Field(default_factory=RouteSummary)
    finish_reason: Literal["stop", "length", "content_filter"]
    replayed: bool = False


# --------------------------------------------------------------------------- feedback ingress


class FeedbackValue(Record):
    score: int = Field(ge=0)
    max_score: int = Field(ge=1)

    @model_validator(mode="after")
    def score_within_scale(self) -> FeedbackValue:
        if self.score > self.max_score:
            raise ValueError("score cannot exceed max_score")
        return self


class FeedbackInput(Contract):
    """Body of POST /v1/interactions/{interaction_id}/feedback from an end user or reviewer."""

    label_type: Literal["thumb", "rubric", "correction", "resolution", "safety"]
    value: FeedbackValue
    comment: Content | None = None
    rubric_version: Identifier | None = None
    training_authorised: bool = False


class CorrectionInput(Contract):
    correction: Content = Field(min_length=1)
    training_authorised: bool = Field(strict=True)


class SubjectDeletionInput(Contract):
    model_config = ConfigDict(strict=True)
    subject: str = Field(
        min_length=1, max_length=512, json_schema_extra={"classification": "personal"}
    )


# --------------------------------------------------------------------------- records


class PolicyDecision(Record):
    policy_version: str
    processing_allowed: bool = True
    content_logging_allowed: bool = False
    evaluation_allowed: bool = False
    human_review_allowed: bool = False
    training_allowed: bool = False
    retention_seconds: int = Field(ge=1)
    residency: Region = "local"
    processing_redaction_version: str = "processing-regex-local-1"
    processing_redaction_counts: dict[str, int] = Field(default_factory=dict)
    persistence_redaction_version: str = "persistence-regex-local-1"
    persistence_redaction_counts: dict[str, int] = Field(default_factory=dict)


class Task(Record):
    label: str
    language: str = "en"
    risk_tier: Literal["low", "medium", "high"] = "medium"
    classifier_version: str = "rules-1"
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str]


class Chunk(Record):
    tenant_id: Identifier
    environment: Environment = "local"
    region: Region = "local"
    allowed_applications: list[Identifier]
    index_id: Identifier
    document_id: Identifier
    document_version: str
    document_family: Identifier
    chunk_id: Identifier
    content: Content
    licence_class: Literal["synthetic", "internal-approved", "unknown"] = "unknown"


class ChunkEvidence(Record):
    document_id: str
    document_version: str
    chunk_id: str
    rank_retrieved: int
    retrieval_score: float
    rerank_score: float | None = None
    supplied_to_model: bool
    context_position: int | None
    token_count: int
    content_hash: str
    content_ref: Reference | None = None
    licence_class: str


class RetrievalRun(Record):
    retrieval_run_id: str = Field(default_factory=uid)
    interaction_id: str
    index_id: str
    index_version: str
    embedding_model: str | None = None
    reranker_model: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    query_hash: str | None
    query_hash_scheme: HashScheme = "hmac-sha256"
    query_ref: Reference | None = None
    latency_ms: float
    candidates: list[ChunkEvidence]


class Candidate(Record):
    model_deployment_id: str
    eligible: bool
    processing_region: Region
    estimated_cost_micros: int
    price_list_version: str
    estimated_latency_ms: float | None = None
    predicted_quality: float | None = None
    ood_score: float | None = None
    reason_codes: list[str]


class RouteDecision(Record):
    route_decision_id: str = Field(default_factory=uid)
    interaction_id: str
    router_version: str = "foundation-only-1"
    experiment_id: str | None = None
    candidates: list[Candidate]
    selected_model_deployment_id: str | None
    fallback_deployment_ids: list[str] = Field(default_factory=list)
    decision_latency_ms: float
    policy_constraints: list[str]


class ValidationCheck(Record):
    name: str
    passed: bool


class Validation(Record):
    validator_version: str = "baseline-validator-1"
    passed: bool
    checks: list[ValidationCheck]


class RequestParameters(Record):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int = Field(ge=1)
    seed: int | None = None


class ToolCall(Record):
    """Reserved for tool use. Arguments are never stored inline; only a hash and a reference."""

    tool_name: Identifier
    arguments_hash: str
    arguments_ref: Reference | None = None
    allowed: bool
    result_ref: Reference | None = None


class GenerationAttempt(Record):
    attempt_id: str = Field(default_factory=uid)
    interaction_id: str
    attempt_number: int = 1
    model_provider: str
    model_id: str
    model_version: str
    deployment_id: str
    adapter_id: str | None = None
    request_parameters: RequestParameters | None = None
    usage: Usage | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output_ref: Reference | None = None
    output_hash: str | None = None
    first_token_latency_ms: float | None = None
    total_latency_ms: float
    estimated_cost_micros: int | None = None
    price_list_version: str | None = None
    finish_reason: FinishReason
    validation: Validation | None = None
    fallback_reason: str | None = None
    error_code: str | None = None


class InputSummary(Record):
    messages_ref: Reference | None = None
    content_hash: str | None
    hash_scheme: HashScheme = "hmac-sha256"
    token_count: int = Field(ge=0)
    tokenizer: str
    modality: list[Modality] = Field(default_factory=_text_only)


class Interaction(Record):
    interaction_id: str
    trace_id: str
    request_id: str
    parent_interaction_id: str | None = None
    tenant_id: str
    subject_id_pseudonymous: str | None
    application_id: str
    environment: Environment
    started_at: datetime
    completed_at: datetime
    task: Task
    policy: PolicyDecision
    input: InputSummary
    retrieval_run_id: str | None
    route_decision_id: str | None
    generation_attempt_ids: list[str]
    final_attempt_id: str | None
    feedback_ids: list[str] = Field(default_factory=list)
    status: Literal["completed", "failed"]
    error_code: str | None = None


class Started(Record):
    interaction_id: str
    application_id: str
    policy_version: str


class Feedback(Record):
    interaction_id: str
    feedback_id: str = Field(default_factory=uid)
    source: Literal["user", "reviewer", "automated", "business_outcome"]
    label_type: Literal["thumb", "rubric", "correction", "resolution", "safety"]
    value: FeedbackValue
    comment_ref: Reference | None = None
    correction_ref: Reference | None = None
    content_hash: str | None = None
    error_code: Literal["persistence_redaction_failed"] | None = None
    rubric_version: str | None = None
    judge_version: str | None = None
    actor_id_pseudonymous: str | None = None
    training_authorised: bool = False
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="after")
    def automated_labels_declare_judge(self) -> Feedback:
        if self.source == "automated" and self.judge_version is None:
            raise ValueError("automated feedback must record judge_version")
        if self.source != "automated" and self.judge_version is not None:
            raise ValueError("judge_version is only valid for automated feedback")
        return self


class DeletionRequest(Record):
    deletion_request_id: str = Field(default_factory=uid)
    scope: Literal["subject", "interaction", "tenant", "document", "dataset"]
    target_id: str
    requested_at: datetime = Field(default_factory=now)
    actor_id_pseudonymous: str | None = None


class DatasetBuilt(Record):
    dataset_id: Identifier
    version: str
    purpose: Literal["adapter_training", "distillation", "router_training", "evaluation"]
    manifest_digest: str
    deletions_applied_through: datetime
    approval_status: Literal["pending", "approved", "rejected"] = "pending"


DatasetPurpose = Literal["adapter_training", "distillation", "router_training", "evaluation"]
TargetSource = Literal["correction", "positive_resolution", "production_output"]
Split = Literal["train", "validation", "test"]
GroupingKey = Literal["document_family", "subject_id_pseudonymous"]


def _target_preferences() -> list[TargetSource]:
    return ["correction", "positive_resolution", "production_output"]


def _grouping_keys() -> list[GroupingKey]:
    return ["document_family", "subject_id_pseudonymous"]


class SourceWindow(Record):
    start: AwareDatetime
    end: AwareDatetime

    @field_validator("start", "end")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered(self) -> SourceWindow:
        if self.start >= self.end:
            raise ValueError("invalid_source_window")
        return self


class TimeSplit(Record):
    train_end: AwareDatetime
    validation_end: AwareDatetime

    @field_validator("train_end", "validation_end")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered(self) -> TimeSplit:
        if self.train_end >= self.validation_end:
            raise ValueError("invalid_time_split")
        return self


class DatasetSpecification(Record):
    dataset_id: Identifier
    purpose: DatasetPurpose = "adapter_training"
    tenant_ids: list[Identifier] = Field(min_length=1, max_length=100)
    source_window: SourceWindow
    eligibility_policy_version: Identifier
    target_preference_order: list[TargetSource] = Field(
        default_factory=_target_preferences,
        min_length=1,
        max_length=3,
    )
    split_strategy: Literal["subject_and_document_family"] = "subject_and_document_family"
    grouping_keys: list[GroupingKey] = Field(default_factory=_grouping_keys)
    time_split: TimeSplit | None = None
    minimum_examples: int = Field(default=1, ge=1)
    seed: int = Field(default=0, ge=0)
    near_duplicate_threshold: float = Field(default=0.8, ge=0, le=1)

    @model_validator(mode="after")
    def valid_specification(self) -> DatasetSpecification:
        if self.dataset_id in {".", ".."} or any(t in {".", ".."} for t in self.tenant_ids):
            raise ValueError("invalid_dataset_identifier")
        if len(set(self.tenant_ids)) != len(self.tenant_ids):
            raise ValueError("duplicate_tenants")
        if len(set(self.target_preference_order)) != len(self.target_preference_order):
            raise ValueError("duplicate_target_preferences")
        if sorted(self.grouping_keys) != ["document_family", "subject_id_pseudonymous"]:
            raise ValueError("joint_grouping_required")
        return self


class DatasetQualitySummary(Record):
    considered: int
    accepted_rate: float
    duplicate_rate: float
    exclusions: dict[str, int]
    label_mix: dict[str, int]
    languages: dict[str, int]


class DatasetApproval(Record):
    status: Literal["pending"] = "pending"
    actor: str | None = None


class DatasetManifest(Record):
    dataset_id: Identifier
    version: Identifier
    purpose: DatasetPurpose
    tenant_ids: list[Identifier]
    created_at: AwareDatetime = Field(default_factory=now)
    source_window: SourceWindow
    eligibility_policy_version: str
    transformation_code_revision: str
    redaction_version: str
    licence_policy_version: str = "synthetic-internal-approved-1"
    examples: dict[Split, int]
    split_strategy: Literal["subject_and_document_family"]
    content_digest: str
    deletions_applied_through: AwareDatetime
    quality_summary: DatasetQualitySummary
    approval: DatasetApproval = Field(default_factory=DatasetApproval)
    specification: DatasetSpecification


class TrainingCompleted(Record):
    job_id: str
    job_type: Literal["adapter", "sft", "distillation", "router"]
    dataset_refs: list[str]
    model_version: str | None
    status: Literal["succeeded", "failed", "cancelled"]


SuiteName = Literal["golden", "held_out", "safety", "retrieval", "performance"]
EvaluationMetrics = Annotated[dict[Identifier, float], Field(max_length=40)]


class EvaluationSpecification(Record):
    evaluation_id: str = Field(default_factory=uid)
    candidate_deployment_id: Identifier
    baseline_deployment_id: Identifier | None
    dataset_id: Identifier
    dataset_version: Identifier
    suites: list[SuiteName] = Field(min_length=1, max_length=5)
    rubric_version: Identifier | None = "synthetic-rubric-1"
    judge_version: Identifier | None = "deterministic-judge-1"
    seed: int = Field(default=23, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    non_inferiority_margin: float = Field(default=0.02, gt=0, lt=1)
    minimum_sample_size: int | None = Field(default=None, ge=1)
    critical_segments: list[Identifier] = Field(default_factory=list, max_length=16)
    performance_requests: int = Field(default=40, ge=1, le=10_000)
    concurrency: int = Field(default=4, ge=1, le=100)
    latency_p95_ms_max: float = Field(default=5000, gt=0)
    error_rate_max: float = Field(default=0, ge=0, le=1)
    cost_per_success_micros_max: int = Field(default=20_000, ge=0)

    @model_validator(mode="after")
    def valid_evaluation(self) -> EvaluationSpecification:
        try:
            identifier = UUID(self.evaluation_id)
        except ValueError:
            raise ValueError("invalid_evaluation_id") from None
        if identifier.version != 7 or str(identifier) != self.evaluation_id:
            raise ValueError("invalid_evaluation_id")
        if self.dataset_id in {".", ".."} or self.dataset_version in {".", ".."}:
            raise ValueError("invalid_dataset_identifier")
        if len(set(self.suites)) != len(self.suites):
            raise ValueError("duplicate_suites")
        if len(set(self.critical_segments)) != len(self.critical_segments):
            raise ValueError("duplicate_segments")
        return self


class EvaluationInput(EvaluationSpecification):
    model_config = ConfigDict(extra="forbid")
    replace: bool = Field(default=False, strict=True)
    operator_note: Content | None = None

    @model_validator(mode="after")
    def replacement_note(self) -> EvaluationInput:
        if self.replace and not (self.operator_note and self.operator_note.strip()):
            raise ValueError("replacement_note_required")
        return self


class ItemScore(Record):
    item_id: Identifier
    score: float = Field(ge=0, le=1)
    segments: list[Identifier] = Field(default_factory=list, max_length=32)


class SuiteResult(Record):
    suite: SuiteName
    items: int = Field(ge=0)
    metrics: EvaluationMetrics
    per_segment_metrics: dict[Identifier, EvaluationMetrics] = Field(
        default_factory=dict, max_length=1000
    )
    failures: dict[Identifier, int] = Field(default_factory=dict, max_length=40)
    scores: list[ItemScore] = Field(default_factory=list, max_length=100_000)
    completed: bool = True
    first_token_latency_ms: float | None = Field(default=None, ge=0)
    total_cost_micros: int | None = Field(default=None, ge=0)
    cost_per_success_micros: int | None = Field(default=None, ge=0)


class PairedComparison(Record):
    mean_delta: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    sample_size: int = Field(ge=0)


class GateDecision(Record):
    gate: Identifier
    passed: bool
    reason: Identifier


class EvaluationReport(Record):
    specification: EvaluationSpecification
    candidate_manifest_version: Identifier
    baseline_manifest_version: Identifier | None
    baseline_report_id: str | None = None
    dataset_content_digest: str
    suite_content_digest: str
    code_revision: str
    started_at: AwareDatetime
    completed_at: AwareDatetime
    suite_results: list[SuiteResult]
    baseline_suite_results: list[SuiteResult]
    paired_comparison: PairedComparison
    segment_comparisons: dict[Identifier, PairedComparison]
    coverage: dict[Identifier, int]
    pilot_standard_deviation: float = Field(ge=0)
    derived_minimum_sample_size: int = Field(ge=1)
    gate_decisions: list[GateDecision]
    passed: bool
    known_limitations: list[str]


class EvaluationCompleted(Record):
    evaluation_id: str
    model_version: str
    baseline_version: str | None
    suites: list[str]
    passed: bool
    report_ref: Reference


class DeploymentChanged(Record):
    deployment_id: str
    model_version: str
    previous_state: LifecycleState | None
    new_state: LifecycleState
    actor_id: str
    reason: str


EventData = (
    Started
    | RetrievalRun
    | RouteDecision
    | GenerationAttempt
    | Interaction
    | Feedback
    | DeletionRequest
    | DatasetBuilt
    | TrainingCompleted
    | EvaluationCompleted
    | DeploymentChanged
)

EventType = Literal[
    "interaction.started.v1",
    "retrieval.completed.v1",
    "route.decided.v1",
    "generation.completed.v1",
    "generation.failed.v1",
    "interaction.completed.v1",
    "feedback.recorded.v1",
    "privacy.deletion.requested.v1",
    "dataset.built.v1",
    "training.completed.v1",
    "evaluation.completed.v1",
    "deployment.changed.v1",
]

EVENT_PAYLOADS: dict[str, type[Record]] = {
    "interaction.started.v1": Started,
    "retrieval.completed.v1": RetrievalRun,
    "route.decided.v1": RouteDecision,
    "generation.completed.v1": GenerationAttempt,
    "generation.failed.v1": GenerationAttempt,
    "interaction.completed.v1": Interaction,
    "feedback.recorded.v1": Feedback,
    "privacy.deletion.requested.v1": DeletionRequest,
    "dataset.built.v1": DatasetBuilt,
    "training.completed.v1": TrainingCompleted,
    "evaluation.completed.v1": EvaluationCompleted,
    "deployment.changed.v1": DeploymentChanged,
}


class Event(Record):
    """Envelope. `event_type` carries the payload major version; `schema_version` must agree."""

    event_id: str = Field(default_factory=uid)
    event_type: EventType
    occurred_at: datetime = Field(default_factory=now)
    producer: Producer = "gateway"
    tenant_id: str
    trace_id: str
    data: EventData

    @model_validator(mode="after")
    def matching_event_data(self) -> Event:
        if not isinstance(self.data, EVENT_PAYLOADS[self.event_type]):
            raise ValueError("event type and payload do not match")
        if self.event_type.rsplit(".", 1)[1] != f"v{self.schema_version.split('.')[0]}":
            raise ValueError("event type version and schema major version disagree")
        return self
