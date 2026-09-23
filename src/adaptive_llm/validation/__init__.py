"""Deterministic output checks; results contain no model content."""

import json
from typing import Protocol

from adaptive_llm.contracts import Validation, ValidationCheck
from adaptive_llm.providers import CITATION_PATTERN, ProviderRequest, ProviderResult


class Validator(Protocol):
    def validate(self, request: ProviderRequest, result: ProviderResult) -> Validation: ...


def _reject_constant(value: str) -> None:
    raise ValueError("invalid_json_constant")


class LocalValidator:
    def validate(self, request: ProviderRequest, result: ProviderResult) -> Validation:
        supplied = {(c.document_id, c.chunk_id) for c in request.context}
        cited = {(c.document_id, c.chunk_id) for c in result.citations}
        cited.update(CITATION_PATTERN.findall(result.content))
        checks = [
            ValidationCheck(name="non_empty", passed=bool(result.content.strip())),
            ValidationCheck(name="citation_ids", passed=cited <= supplied),
        ]
        if request.response_format.type == "json_object":
            try:
                valid_json = isinstance(
                    json.loads(result.content, parse_constant=_reject_constant), dict
                )
            except (ValueError, RecursionError):
                valid_json = False
            checks.append(ValidationCheck(name="json_object", passed=valid_json))
        return Validation(passed=all(check.passed for check in checks), checks=checks)
