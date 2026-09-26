"""Provider-neutral v1 contracts.

Money is integer USD micros; unknown counts are null. Ingress contracts (client-supplied
bodies) reject unknown fields. Records (stored and emitted payloads) ignore unknown fields so
producers may add fields before consumers upgrade, as required by the specification's
backward-compatibility rule. Both reject an unknown schema_version.
"""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from secrets import randbits
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from adaptive_llm.structured import canonical, check_schema


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
    role: Literal["system", "user", "assistant"]
    content: Content

    @model_validator(mode="after")
    def bounded_system(self) -> Message:
        if self.role == "system" and len(self.content) > 8000:
            raise ValueError("system_message_too_long")
        return self


class RagOptions(Contract):
    enabled: bool = False
    index_id: Identifier | None = None

    @model_validator(mode="after")
    def index_required_when_enabled(self) -> RagOptions:
        if self.enabled and self.index_id is None:
            raise ValueError("rag.index_id is required when rag is enabled")
        return self


class JsonSchema(Contract):
    model_config = ConfigDict(serialize_by_alias=True)
    name: Identifier
    schema_: dict[str, JsonValue] = Field(alias="schema", repr=False)

    @model_validator(mode="after")
    def valid_schema(self) -> JsonSchema:
        check_schema(self.schema_)
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical(self.schema_).encode("utf-8")).hexdigest()


class ResponseFormat(Contract):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchema | None = None

    @model_validator(mode="after")
    def schema_required(self) -> ResponseFormat:
        if (self.type == "json_schema") != (self.json_schema is not None):
            raise ValueError("response_format_schema_mismatch")
        return self


class RoutingOptions(Contract):
    mode: Literal["auto", "foundation", "specialist"] = "auto"
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
        if any(m.role == "system" for m in self.messages[1:]):
            raise ValueError("system_message_must_be_first_and_unique")
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
    specialist_served: bool = False


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


class BreakerThresholds(Contract):
    window_seconds: float = Field(default=60, gt=0, le=3600)
    minimum_samples: int = Field(default=5, ge=1, le=1000)
    error_rate: float = Field(default=0.5, gt=0, le=1)
    validation_failure_rate: float = Field(default=0.5, gt=0, le=1)
    p95_latency_ms: float = Field(default=5000, gt=0)
    cooldown_seconds: float = Field(default=30, gt=0, le=3600)


FallbackReason = Literal[
    "validation_failure",
    "endpoint_error",
    "deadline_risk",
    "policy_uncertainty",
    "unsupported_tool",
    "low_confidence",
    "out_of_distribution",
    "low_quality",
    "circuit_open",
    "live_specialists_disabled",
    "not_cheapest",
]


def _hard_fallback_reasons() -> list[FallbackReason]:
    return [
        "validation_failure",
        "endpoint_error",
        "deadline_risk",
        "policy_uncertainty",
        "unsupported_tool",
        "circuit_open",
    ]


class CanaryConfig(Contract):
    traffic_fraction: float = Field(default=0, ge=0, le=1)
    tenant_allowlist: list[Identifier] = Field(default_factory=list, max_length=100)
    assignment: Literal["sha256-interaction-policy-v1"] = "sha256-interaction-policy-v1"


class RollbackThresholds(Contract):
    interval_seconds: float = Field(default=1, gt=0, le=60)
    window_seconds: float = Field(default=300, gt=0, le=86400)
    minimum_samples: int = Field(default=4, ge=1)
    non_inferiority_margin: float = Field(default=0.02, gt=0, lt=1)
    validation_failure_rate_max: float = Field(default=0, ge=0, le=1)
    error_rate_max: float = Field(default=0, ge=0, le=1)
    p95_latency_ms_max: float = Field(default=5000, gt=0)
    cost_ratio_max: float = Field(default=1, gt=0)


class RoutingFeatures(Contract):
    task: Identifier
    language: Identifier = "en"
    risk_tier: Literal["low", "medium", "high"] = "medium"
    chunk_count: int = Field(default=0, ge=0)
    top_score: float = Field(default=0, ge=0)
    index_id: Identifier | None = None
    input_tokens: int = Field(ge=0)
    context_supplied: bool = False


class RoutingObservation(Contract):
    quality: float | None = Field(default=None, ge=0, le=1)
    validation_pass: bool | None = None
    cost_micros: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)


class RoutingRow(Contract):
    interaction_id: Identifier
    tenant_id: Identifier
    features: RoutingFeatures
    foundation_id: Identifier
    candidates: dict[Identifier, RoutingObservation | None]
    source_dataset_id: Identifier
    source_dataset_version: Identifier
    source_example_hash: Identifier | None = None
    split: Literal["train", "validation", "test"] = "train"
    transformation_code_revision: str = "unknown"


class RoutePolicy(Contract):
    """Each policy id names one immutable version; activation is a separate operation."""

    policy_id: Identifier = Field(default_factory=uid)
    eligible_specialist_versions: list[Identifier] = Field(default_factory=list, max_length=16)
    quality_threshold: float = Field(default=0.9, ge=0, le=1)
    router_confidence_threshold: float = Field(default=0.85, ge=0, le=1)
    ood_threshold_max: float = Field(default=0.15, ge=0, le=1)
    max_attempts: int = Field(default=2, ge=1, le=8)
    hard_fallback_on: list[FallbackReason] = Field(default_factory=_hard_fallback_reasons)
    foundation_fallback: Identifier = "fake-foundation-local-1"
    shadow_enabled: bool = False
    tenant_enabled: dict[Identifier, bool] = Field(default_factory=dict, max_length=100)
    task_enabled: dict[Identifier, bool] = Field(default_factory=dict, max_length=32)
    kill_switch: bool = False
    live_specialists_allowed: bool = False
    router_version: Identifier | None = None
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    rollback: RollbackThresholds = Field(default_factory=RollbackThresholds)
    breaker: BreakerThresholds = Field(default_factory=BreakerThresholds)

    @model_validator(mode="after")
    def unique_specialists(self) -> RoutePolicy:
        if len(set(self.eligible_specialist_versions)) != len(self.eligible_specialist_versions):
            raise ValueError("duplicate_specialists")
        if self.live_specialists_allowed and self.router_version is None:
            raise ValueError("promoted_router_required")
        return self


class Candidate(Record):
    model_deployment_id: str
    eligible: bool
    processing_region: Region
    estimated_cost_micros: int
    price_list_version: str
    estimated_latency_ms: float | None = None
    predicted_quality: float | None = None
    confidence: float | None = None
    ood_score: float | None = None
    reason_codes: list[str]


class RouteDecision(Record):
    route_decision_id: str = Field(default_factory=uid)
    interaction_id: str
    router_version: str = "foundation-only-1"
    experiment_id: str | None = None
    route_policy_id: str | None = None
    fallback_reasons: list[str] = Field(default_factory=list)
    candidates: list[Candidate]
    selected_model_deployment_id: str | None
    fallback_deployment_ids: list[str] = Field(default_factory=list)
    decision_latency_ms: float
    policy_constraints: list[str]
    response_schema_name: Identifier | None = None
    response_schema_sha256: str | None = None


class ValidationCheck(Record):
    version: str = "deterministic-1"
    name: str
    passed: bool
    severity: Literal["hard", "advisory"] = "hard"
    critical_safety: bool = False


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
    attempt_number: int = Field(default=1, ge=0)
    shadow: bool = False
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
    response_format_ref: Reference | None = None
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
    total_cost_micros: int = Field(default=0, ge=0)
    shadow_cost_micros: int = Field(default=0, ge=0)
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
    fixture_set: Identifier = Field(default="synthetic", exclude_if=lambda v: v == "synthetic")
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
    source_dataset_id: Identifier | None = None
    source_dataset_version: Identifier | None = None
    teacher_deployment_id: Identifier | None = None
    teacher_minimum_score: float = Field(default=0.9, ge=0, le=1)
    teacher_max_output_tokens: int = Field(default=256, ge=1, le=2048)
    general_safety_fraction: float = Field(default=0.2, ge=0, le=0.8)
    soft_targets: bool = False

    @model_validator(mode="after")
    def valid_specification(self) -> DatasetSpecification:
        if self.fixture_set in {".", ".."}:
            raise ValueError("invalid_fixture_set")
        if self.purpose == "distillation" and (
            self.source_dataset_id is None
            or self.source_dataset_version is None
            or self.teacher_deployment_id is None
        ):
            raise ValueError("distillation_source_and_teacher_required")
        if self.purpose == "router_training" and (
            self.source_dataset_id is None or self.source_dataset_version is None
        ):
            raise ValueError("routing_source_dataset_required")
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


class SignedRecord(Record):
    # Omitted on historical records to preserve their exact v1/v2 MAC encoding.
    signature: str | None = Field(default=None, exclude_if=lambda v: v is None)
    signature_key_id: Identifier | None = Field(default=None, exclude_if=lambda v: v is None)
    signature_version: Literal["ed25519-v1"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


class DatasetApproval(SignedRecord):
    status: Literal["pending", "approved", "rejected"] = "pending"
    actor: str | None = None
    reason: str | None = None
    at: AwareDatetime | None = None
    mac: str | None = None
    approval_mac_version: Literal["1", "2"] = "1"


class RouteControlNote(Contract):
    reason: str = Field(min_length=1, max_length=2000)
    tenant_id: Identifier | None = None
    task: Identifier | None = None
    specialist_version: Identifier | None = None

    @model_validator(mode="after")
    def valid_scope(self) -> RouteControlNote:
        if (
            not self.reason.strip()
            or sum(v is not None for v in (self.tenant_id, self.task, self.specialist_version)) > 1
        ):
            raise ValueError("invalid_route_control_note")
        return self


class OperatorNote(Record):
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator("reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("operator_note_required")
        return value


class DatasetManifest(SignedRecord):
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
    distillation: DistillationLineage | None = None


class SoftTargetFile(Record):
    tenant_id: Identifier
    example_hash: Identifier
    plaintext_hash: str
    key_version: Identifier


class DistillationLineage(Record):
    source_dataset_id: Identifier
    source_dataset_version: Identifier
    source_content_digest: str
    teacher_deployment_id: Identifier
    teacher_version: Identifier
    teacher_artifact_digest: str
    teacher_parameter_count: int | None = Field(default=None, gt=0)
    tokenizer_id: Identifier | None = None
    chat_template_version: Identifier | None = None
    generation_parameters: dict[Identifier, int | bool]
    judge_version: Identifier
    rubric_version: Identifier
    minimum_score: float
    requested_mix_fraction: float
    mix_counts: dict[Identifier, int]
    mix_content_digest: str
    soft_target_files: list[SoftTargetFile] = Field(default_factory=list)


class AdapterConfig(Record):
    target_modules: list[Identifier] = Field(
        default_factory=lambda: ["q_proj", "v_proj"], min_length=1
    )
    rank: int = Field(default=8, ge=1)
    alpha: float = Field(default=16, gt=0)
    dropout: float = Field(default=0, ge=0, lt=1)
    learning_rate: float = Field(default=0.0002, gt=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    gradient_accumulation: int = Field(default=1, ge=1)


class TrainingJobSpecification(Record):
    job_id: Identifier = Field(default_factory=uid)
    job_type: Literal["adapter", "router", "distillation"] = "adapter"
    registry_id: Identifier = "synthetic-specialist"
    dataset_id: Identifier
    dataset_version: Identifier
    base_model_id: Identifier = "fake-foundation"
    base_model_revision: Identifier = "fake-1"
    base_model_licence: Identifier = "synthetic"
    tokenizer_id: Identifier = "fake-whitespace-v1"
    chat_template_version: Identifier = "fake-chat-1"
    adapter_config: AdapterConfig = Field(default_factory=AdapterConfig)
    seed: int = Field(default=23, ge=0)
    steps: int = Field(default=2, ge=1)
    batch_size: int = Field(default=1, ge=1)
    max_sequence_length: int = Field(default=256, ge=2)
    checkpoint_every: int = Field(default=1, ge=1)
    hardware_class: Identifier = "cpu"
    container_digest: str = "local"
    code_revision: str = "server"
    input_micros_per_1000_tokens: int = Field(default=1000, ge=0)
    output_micros_per_1000_tokens: int = Field(default=2000, ge=0)
    student_training: Literal["full", "lora"] = "full"
    soft_target_weight: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def valid_training(self) -> TrainingJobSpecification:
        try:
            identifier = UUID(self.job_id)
        except ValueError:
            raise ValueError("invalid_job_id") from None
        if identifier.version != 7 or str(identifier) != self.job_id:
            raise ValueError("invalid_job_id")
        if any(v in {".", ".."} for v in (self.registry_id, self.dataset_id, self.dataset_version)):
            raise ValueError("invalid_training_identifier")
        if self.container_digest != "local" and not (
            self.container_digest.startswith("sha256:")
            and len(self.container_digest) == 71
            and all(c in "0123456789abcdef" for c in self.container_digest[7:])
        ):
            raise ValueError("invalid_container_digest")
        return self


class ResourceUsage(Record):
    examples: int = Field(default=0, ge=0)
    steps: int = Field(default=0, ge=0)
    artifact_bytes: int = Field(default=0, ge=0)
    cpu_seconds: float | None = Field(default=None, ge=0)
    wall_seconds: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)


class TrainingJob(Record):
    specification: TrainingJobSpecification
    trainer_architecture: Identifier = "deterministic-fake-adapter-v1"
    created_at: AwareDatetime = Field(default_factory=now)
    cancel_requested: bool = False
    model_version: Identifier = Field(default_factory=uid)
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    checkpoint_refs: list[Reference] = Field(default_factory=list)
    artifact_ref: Reference | None = None
    artifact_digest: str | None = None
    failure_code: Identifier | None = None


class LifecycleTransition(Record):
    from_state: LifecycleState | None
    to_state: LifecycleState
    actor: Identifier
    at: AwareDatetime = Field(default_factory=now)
    reason: str
    evaluation_id: Identifier | None = None


class DatasetLineage(Record):
    dataset_id: Identifier
    version: Identifier
    content_digest: str
    weight: float = Field(default=1, gt=0, le=1)


def _pruning_structures() -> list[Literal["attention_heads", "mlp_channels", "layers"]]:
    return ["attention_heads"]


class PruningPlan(Contract):
    structures: list[Literal["attention_heads", "mlp_channels", "layers"]] = Field(
        default_factory=_pruning_structures, min_length=1, max_length=3
    )
    maximum_fraction: float = Field(default=0.25, gt=0, le=0.5)
    ranking_rule: Identifier = "ablation_sensitivity"


class PruningLineage(Record):
    study_id: Identifier
    plan: PruningPlan
    removed_indices: dict[str, list[int]]
    parameter_count_before: int = Field(gt=0)
    parameter_count_after: int = Field(gt=0)
    base_digest: str
    adapter_version: Identifier | None = None
    adapter_digest: str | None = None
    calibration_dataset: DatasetLineage
    evaluation_dataset: DatasetLineage
    baseline_evaluation_id: Identifier


class ModelManifest(SignedRecord):
    registry_id: Identifier
    version: Identifier
    created_at: AwareDatetime = Field(default_factory=now)
    state: LifecycleState = "candidate"
    tenant_ids: list[Identifier]
    base_model_id: Identifier
    base_model_revision: Identifier
    base_model_licence: Identifier
    adapter_architecture: Identifier = "deterministic-fake-adapter-v1"
    adapter_config: AdapterConfig
    tokenizer_id: Identifier
    chat_template_version: Identifier
    datasets: list[DatasetLineage]
    training_job_id: Identifier
    code_revision: str
    container_digest: str
    configuration_digest: str
    seed: int
    hardware_class: Identifier
    evaluation_reports: dict[Identifier, bool] = Field(default_factory=dict)
    intended_tasks: list[Identifier] = Field(default_factory=lambda: ["synthetic-smoke"])
    excluded_tasks: list[Identifier] = Field(default_factory=lambda: ["real-inference"])
    languages: list[Identifier] = Field(default_factory=lambda: ["en"])
    context_limit: int = Field(default=4096, ge=1)
    capability_signature_version: Literal["1"] | None = None
    processing_region: Region = "local"
    modalities: list[Modality] = Field(default_factory=_text_only)
    tools_supported: bool = False
    input_micros_per_1000_tokens: int = Field(default=1000, ge=0)
    output_micros_per_1000_tokens: int = Field(default=2000, ge=0)
    safety_notes: list[str] = Field(
        default_factory=lambda: ["Synthetic adapter; no learned weights."]
    )
    known_limitations: list[str] = Field(
        default_factory=lambda: ["Local MAC, not asymmetric signing."]
    )
    artifact_hashes: dict[str, str]
    artifact_digest: str
    artifact_mac: str
    storage_location: Reference
    lifecycle_history: list[LifecycleTransition] = Field(default_factory=list)
    student_architecture: Identifier | None = None
    student_parameter_count: int | None = Field(default=None, gt=0)
    distillation: DistillationLineage | None = None
    manifest_mac_version: Literal["1", "2"] = "1"
    pruning: PruningLineage | None = Field(default=None, exclude_if=lambda v: v is None)


class PromotionRequest(OperatorNote):
    model_version: Identifier
    target_state: LifecycleState
    actor: Identifier | None = None
    evaluation_id: Identifier | None = None


class TrainingCompleted(Record):
    job_id: str
    job_type: Literal["adapter", "sft", "distillation", "router"]
    dataset_refs: list[str]
    model_version: str | None
    status: Literal["succeeded", "failed", "cancelled"]
    artifact_digest: str | None = None
    failure_code: Identifier | None = None


SuiteName = Literal["golden", "held_out", "safety", "retrieval", "performance", "routing"]
EvaluationMetrics = Annotated[dict[Identifier, float], Field(max_length=40)]


class EvaluationSpecification(Record):
    evaluation_id: str = Field(default_factory=uid)
    candidate_deployment_id: Identifier
    baseline_deployment_id: Identifier | None
    dataset_id: Identifier
    dataset_version: Identifier
    application_id: Identifier = "evaluation"
    fixture_set: Identifier | None = None
    fixture_tenant_id: Identifier | None = None
    suites: list[SuiteName] = Field(min_length=1, max_length=6)
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
        if self.fixture_set in {".", ".."}:
            raise ValueError("invalid_fixture_evaluation")
        if self.fixture_tenant_id is not None and (
            self.fixture_set is None or set(self.suites) != {"golden", "safety"}
        ):
            raise ValueError("invalid_fixture_evaluation")
        if (
            self.fixture_set is not None
            and self.fixture_tenant_id is None
            and set(self.suites) != {"golden", "safety", "held_out", "retrieval", "performance"}
        ):
            raise ValueError("invalid_fixture_evaluation")
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
    observation: RoutingObservation | None = None


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


class EvaluationReport(SignedRecord):
    specification: EvaluationSpecification
    candidate_manifest_version: Identifier
    candidate_model_version: str | None = None
    candidate_artifact_digest: str | None = None
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
    distillation: DistillationLineage | None = None


class BenchmarkSpecification(Record):
    benchmark_id: Identifier = Field(default_factory=uid)
    candidate_version: Identifier
    evaluation_id: Identifier
    requests: int = Field(default=40, ge=4, le=1000)

    @field_validator("benchmark_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        identifier = UUID(value)
        if identifier.version != 7 or str(identifier) != value:
            raise ValueError("invalid_benchmark_id")
        return value


class BenchmarkMeasurement(Record):
    requests: int
    successes: int
    concurrency: Literal[4] = 4
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    requests_per_second: float = Field(ge=0)
    peak_rss_bytes: int = Field(gt=0)
    total_cost_micros: int = Field(ge=0)
    cost_per_success_micros: int | None
    input_micros_per_1000_tokens: int
    output_micros_per_1000_tokens: int
    parameter_count: int | None = Field(default=None, gt=0, exclude_if=lambda v: v is None)


class BenchmarkReport(SignedRecord):
    specification: BenchmarkSpecification
    tenant_ids: list[Identifier]
    candidate_artifact_digest: str
    teacher_version: Identifier
    teacher_artifact_digest: str
    dataset_content_digest: str
    request_mix_digest: str
    created_at: AwareDatetime = Field(default_factory=now)
    student: BenchmarkMeasurement
    teacher: BenchmarkMeasurement
    quality_comparison: PairedComparison
    latency_reduction_fraction: float | None
    cost_reduction_fraction: float | None
    passed: bool
    known_limitations: list[str]
    mac: str = ""
    pruning_study_id: Identifier | None = Field(default=None, exclude_if=lambda v: v is None)
    peak_rss_reduction_fraction: float | None = Field(default=None, exclude_if=lambda v: v is None)
    gate_reasons: list[Identifier] = Field(default_factory=list, exclude_if=lambda v: not v)


class ShadowComparison(Record):
    comparison_id: str = Field(default_factory=uid)
    interaction_id: str
    policy_id: str
    tenant_id: str
    specialist_version: str
    created_at: AwareDatetime = Field(default_factory=now)
    segments: list[str]
    judge_version: str = "shadow-chunk-overlap-1"
    rubric_version: str = "synthetic-rubric-1"
    foundation_score: float
    specialist_score: float
    foundation_citation_precision: float
    foundation_citation_recall: float
    specialist_citation_precision: float
    specialist_citation_recall: float
    input_token_delta: int
    output_token_delta: int
    cost_delta_micros: int
    specialist_cost_micros: int | None = Field(default=None, ge=0)
    latency_delta_ms: float
    specialist_validation: Validation


class ShadowAggregate(Record):
    opportunities: int
    comparisons: int
    coverage: float
    specialist_validation_pass_rate: float | None
    score_delta: PairedComparison
    mean_cost_delta_micros: float | None
    mean_latency_delta_ms: float | None
    mean_input_token_delta: float | None
    mean_output_token_delta: float | None
    citation_metrics: dict[str, float]


class ShadowReport(Record):
    """Aggregate shadow-chunk-overlap-1 scores: coarse grounding proxies, not evaluation rubrics."""

    policy_id: str
    since: AwareDatetime
    overall: ShadowAggregate
    critical_segments: dict[str, ShadowAggregate]
    passed: bool = False


class LiveObservation(Contract):
    interaction_id: Identifier
    policy_id: Identifier
    tenant_id: Identifier
    created_at: AwareDatetime = Field(default_factory=now)
    features: RoutingFeatures
    specialist_version: Identifier | None = None
    specialist_served: bool = False
    fallback_used: bool = False
    success: bool
    validation_failure: bool = False
    error: bool = False
    critical_safety_incidents: int = Field(default=0, ge=0)
    quality: float = Field(ge=0, le=1)
    total_cost_micros: int = Field(ge=0)
    shadow_cost_micros: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)


class OutcomeAggregate(Record):
    interactions: int
    successes: int
    total_cost_micros: int
    cost_per_success_micros: int | None
    validation_failure_rate: float | None
    error_rate: float | None
    p95_latency_ms: float | None
    critical_safety_incidents: int


class CanaryAggregate(Record):
    specialist: OutcomeAggregate
    foundation: OutcomeAggregate
    specialist_served: OutcomeAggregate
    foundation_served: OutcomeAggregate
    quality_delta: PairedComparison
    cost_delta_micros: PairedComparison
    served_cost_delta_micros: PairedComparison
    cost_reduction_fraction: float | None
    served_cost_reduction_fraction: float | None
    shadow_cost_micros: int
    passed: bool
    breach_reasons: list[Identifier]
    trigger_segments: list[str] = Field(default_factory=list)


class CanaryReport(Record):
    policy_id: Identifier
    since: AwareDatetime
    until: AwareDatetime
    overall: CanaryAggregate
    critical_segments: dict[str, CanaryAggregate]
    specialists: dict[str, CanaryAggregate]
    passed: bool


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
    previous_state: LifecycleState | Literal["active", "inactive", "enabled", "disabled"] | None
    new_state: LifecycleState | Literal["active", "inactive", "enabled", "disabled"]
    actor_id: str
    reason: str
    evaluation_id: str | None = None


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
