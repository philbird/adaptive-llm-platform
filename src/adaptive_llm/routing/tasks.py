"""Versioned application/schema rules over authenticated application selection."""

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_llm.contracts import Identifier, InferenceRequest, Task


class TaskClassifier(Protocol):
    def classify(self, request: InferenceRequest) -> Task: ...


class TaskRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: Identifier
    schema_name: Identifier | None = None
    label: Identifier
    language: Identifier
    risk_tier: Literal["low", "medium", "high"]


class TaskRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["rules-1"] = "rules-1"
    rules: list[TaskRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique(self) -> "TaskRules":
        if len({(r.application_id, r.schema_name) for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate_task_rule")
        return self


class RulesClassifier:
    def __init__(self, path: Path | None = None) -> None:
        self.config = TaskRules.model_validate_json(path.read_text()) if path else TaskRules()

    def classify(self, request: InferenceRequest) -> Task:
        schema = request.response_format.json_schema
        matches = [
            r
            for r in self.config.rules
            if r.application_id == request.application_id
            and (r.schema_name is None or schema is not None and r.schema_name == schema.name)
        ]
        if matches:
            rule = max(matches, key=lambda r: r.schema_name is not None)
            return Task(
                label=rule.label,
                language=rule.language,
                risk_tier=rule.risk_tier,
                classifier_version=self.config.version,
                confidence=1,
                reason_codes=["application_task_map"],
            )
        return Task(
            label="question_answering" if request.rag.enabled else "general",
            classifier_version="placeholder-rag-flag-1",
            confidence=0.5,
            reason_codes=["rag_flag_only"],
        )
