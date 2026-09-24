"""Versioned deterministic checks. Grounding/language are heuristics, not safety proofs."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol

from opentelemetry.trace import Tracer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_llm.contracts import Identifier, Validation, ValidationCheck
from adaptive_llm.providers import CITATION_PATTERN, ProviderRequest, ProviderResult


class Validator(Protocol):
    def validate(self, request: ProviderRequest, result: ProviderResult) -> Validation: ...


class TracedValidator:
    def __init__(self, delegate: Validator, tracer: Tracer, attributes: dict[str, str]) -> None:
        self.delegate, self.tracer, self.attributes = delegate, tracer, attributes

    def validate(self, request: ProviderRequest, result: ProviderResult) -> Validation:
        with self.tracer.start_as_current_span(
            "validation",
            attributes=self.attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            validation = self.delegate.validate(request, result)
            span.set_attribute("validator_version", validation.validator_version)
            return validation


class DomainTest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Identifier
    version: Identifier
    kind: Literal["required_text", "forbidden_text", "required_json_key"]
    value: str = Field(min_length=1, max_length=200)


Severity = Literal["hard", "advisory"]
DEFAULT_SEVERITIES: dict[str, Severity] = {
    "non_empty": "hard",
    "citation_ids": "hard",
    "json_object": "hard",
    "tool_allowlist": "hard",
    "citation_required": "advisory",
    "groundedness": "advisory",
    "language": "advisory",
    "repetition": "advisory",
    "truncation": "advisory",
}


class DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    language: Literal["en"] = "en"
    tests: list[DomainTest] = Field(default_factory=list, max_length=32)
    severity: dict[str, Severity] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_check_names(self) -> "DomainConfig":
        names = {f"domain.{test.name}" for test in self.tests}
        if len(names) != len(self.tests):
            raise ValueError("duplicate_domain_check")
        if not self.severity.keys() <= DEFAULT_SEVERITIES.keys() | names:
            raise ValueError("unknown_validation_check")
        return self


def _reject_constant(value: str) -> None:
    raise ValueError("invalid_json_constant")


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def grams(text: str) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


CONNECTIVES = frozenset(
    {
        "however",
        "therefore",
        "in addition",
        "for example",
        "in conclusion",
        "nevertheless",
        "furthermore",
        "to summarize",
    }
)


def grounded(text: str, context: tuple[str, ...]) -> bool:
    if not context:
        return True
    evidence = set().union(*(grams(chunk) for chunk in context))
    # Protect citation identifiers (which can contain periods) and decimal numbers.
    marked = CITATION_PATTERN.sub("\x00", text)
    for sentence in re.split(r"[!?]|\.(?=\s|$)|\n+", marked):
        cited = "\x00" in sentence
        claim = sentence.replace("\x00", "").strip()
        claim = re.sub(r"^SYNTHETIC ANSWER:\s*", "", claim).strip()
        normalized = " ".join(words(claim))
        if not normalized:
            continue
        if not cited and normalized in CONNECTIVES:
            continue
        if not grams(claim) & evidence:
            return False
    return True


def text_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in text_values(item)]
    if isinstance(value, list):
        return [text for item in value for text in text_values(item)]
    return []


class LocalValidator:
    def __init__(self, directory: Path | None = None) -> None:
        directory = directory or Path(__file__).resolve().parents[3] / "configs/validation"
        self.configs = {
            path.stem: DomainConfig.model_validate_json(path.read_text())
            for path in sorted(directory.glob("*.json"))
        }

    def validate(self, request: ProviderRequest, result: ProviderResult) -> Validation:
        supplied = {(c.document_id, c.chunk_id) for c in request.context}
        cited = {(c.document_id, c.chunk_id) for c in result.citations}
        cited.update(CITATION_PATTERN.findall(result.content))
        checks: list[ValidationCheck] = []
        config = self.configs.get(request.application_id, DomainConfig())

        def check(
            name: str,
            passed: bool,
            version: str = "deterministic-1",
            default_severity: Severity = "hard",
        ) -> None:
            severity = config.severity.get(name, DEFAULT_SEVERITIES.get(name, default_severity))
            checks.append(
                ValidationCheck(
                    name=name,
                    version=version,
                    passed=passed,
                    severity=severity,
                )
            )

        check("non_empty", bool(result.content.strip()))
        check("citation_ids", cited <= supplied)
        check("citation_required", not supplied or bool(cited))
        text = result.content
        obj: object = None
        if request.response_format.type == "json_object":
            try:
                obj = json.loads(result.content, parse_constant=_reject_constant)
            except (ValueError, RecursionError):
                pass
            check("json_object", isinstance(obj, dict))
            text = "\n".join(text_values(obj))
        check(
            "groundedness",
            grounded(text, tuple(c.content for c in request.context)),
            "five-gram-heuristic-1",
        )
        letters = [c for c in text if c.isalpha()]
        check(
            "language",
            not letters
            or sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters) >= 0.9,
            "english-latin-script-heuristic-1",
        )
        tokens = words(text)
        counts = Counter(tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4))
        check("repetition", max(counts.values(), default=0) < 3)
        check("truncation", result.finish_reason != "length")
        check("tool_allowlist", not result.tool_calls)
        for test in config.tests:
            passed = (
                test.value.casefold() in text.casefold()
                if test.kind == "required_text"
                else test.value.casefold() not in text.casefold()
                if test.kind == "forbidden_text"
                else isinstance(obj, dict) and test.value in obj
            )
            check(
                f"domain.{test.name}",
                passed,
                test.version,
                "advisory" if test.kind == "required_text" else "hard",
            )
        return Validation(
            validator_version="local-validator-3",
            passed=all(item.passed for item in checks if item.severity == "hard"),
            checks=checks,
        )
