"""Bounded research requests and content-free study summaries."""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator

from adaptive_llm.contracts import (
    Contract,
    DatasetLineage,
    EvaluationInput,
    Identifier,
    PruningPlan,
    TrainingJobSpecification,
    now,
    uid,
)


class BaseSpecification(Contract):
    base_model_id: Identifier = "tiny"
    base_model_revision: Identifier = "seed-17"
    base_model_licence: Identifier = "CC0-1.0"
    tokenizer_id: Identifier = "tiny-byte-v1"
    chat_template_version: Identifier = "tiny-chat-v1"
    adapter_version: Identifier | None = None


class BaselineSpecification(Contract):
    base: BaseSpecification
    evaluation: EvaluationInput


class ActivationSpecification(Contract):
    job_type: Literal["activation_study"] = "activation_study"
    study_id: Identifier = Field(default_factory=uid)
    base: BaseSpecification = Field(default_factory=BaseSpecification)
    calibration_dataset_id: Identifier
    calibration_dataset_version: Identifier
    evaluation_dataset_id: Identifier
    evaluation_dataset_version: Identifier
    baseline_evaluation_id: Identifier
    calibration_split: Literal["train", "validation"] = "validation"
    sample_size: int = Field(default=64, ge=1, le=512)
    max_sequence_length: int = Field(default=128, ge=2, le=512)
    seed: int = Field(default=23, ge=0)

    @field_validator("study_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        parsed = UUID(value)
        if parsed.version != 7 or str(parsed) != value:
            raise ValueError("invalid_study_id")
        return value


class PruneSpecification(Contract):
    job_type: Literal["structured_prune"] = "structured_prune"
    study_id: Identifier
    plan: PruningPlan = Field(default_factory=PruningPlan)
    training: TrainingJobSpecification


class StudySummary(Contract):
    specification: ActivationSpecification
    created_at: AwareDatetime = Field(default_factory=now)
    tenant_ids: list[Identifier]
    base_digest: str
    adapter_digest: str | None
    calibration_dataset: DatasetLineage
    evaluation_dataset: DatasetLineage
    sample_size: int
    hook_version: Literal["aggregate-taylor-v1", "aggregate-residual-taylor-v2"] = (
        "aggregate-residual-taylor-v2"
    )
    wall_seconds: float = Field(ge=0)
    shapes: dict[str, list[int]]
    rankings: dict[str, list[int]]
    parameter_count_before: int
    candidates: dict[str, int] = Field(default_factory=dict)
    benchmark_ids: list[Identifier] = Field(default_factory=list)
    evaluation_ids: list[Identifier] = Field(default_factory=list)
