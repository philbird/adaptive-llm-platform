import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import Citation, ResponseFormat, ToolCall, ValidationCheck
from adaptive_llm.providers import FakeProvider, ProviderRequest
from adaptive_llm.validation import LocalValidator, grounded


@pytest.mark.parametrize(
    "sentence,valid",
    [
        ("However.", True),
        ("Therefore!", True),
        ("In addition.", True),
        ("However the moon is made of synthetic cheese.", False),
        ("Therefore you are entitled to free synthetic money.", False),
        ("However [synthetic/doc].", False),
        ("SYNTHETIC customers may return unused items.", True),
    ],
)
def test_grounded_claims_and_narrow_connectives(sentence: str, valid: bool) -> None:
    assert grounded(sentence, ("SYNTHETIC customers may return unused items.",)) is valid


@pytest.mark.parametrize(
    "name,updates",
    [
        ("citation_required", {"content": "SYNTHETIC answer", "citations": ()}),
        ("groundedness", {"content": "SYNTHETIC completely invented answer without evidence."}),
        ("language", {"content": "这是完全合成的测试文本"}),
        ("repetition", {"content": "synthetic one two three four " * 4}),
        ("truncation", {"finish_reason": "length"}),
        (
            "tool_allowlist",
            {
                "tool_calls": (
                    ToolCall(tool_name="synthetic", arguments_hash="hash", allowed=False),
                )
            },
        ),
    ],
)
async def test_each_expanded_check(
    provider_request: ProviderRequest,
    name: str,
    updates: dict[str, object],
) -> None:
    result = replace(await FakeProvider().generate(provider_request), **updates)
    validation = LocalValidator().validate(provider_request, result)
    assert not next(c for c in validation.checks if c.name == name).passed
    assert all(c.version for c in validation.checks)
    assert validation.passed is (name != "tool_allowlist")
    assert next(c for c in validation.checks if c.name == name).severity == (
        "hard" if name == "tool_allowlist" else "advisory"
    )


async def test_domain_tests_are_loaded_per_application(
    tmp_path: Path,
    provider_request: ProviderRequest,
) -> None:
    (tmp_path / "synthetic-app.json").write_text(
        '{"language":"en","tests":['
        '{"name":"required","version":"v1","kind":"required_text","value":"SYNTHETIC_REQUIRED"},'
        '{"name":"forbidden","version":"v2","kind":"forbidden_text","value":"SYNTHETIC ANSWER"},'
        '{"name":"schema","version":"v3","kind":"required_json_key","value":"answer"}]}'
    )
    validator = LocalValidator(tmp_path)
    request = replace(provider_request, application_id="synthetic-app")
    result = await FakeProvider().generate(request)
    checks = {c.name: c for c in validator.validate(request, result).checks}
    assert all(not checks[f"domain.{name}"].passed for name in ["required", "forbidden", "schema"])
    assert checks["domain.forbidden"].version == "v2"
    assert checks["domain.required"].severity == "advisory"
    assert checks["domain.forbidden"].severity == checks["domain.schema"].severity == "hard"
    assert not any(
        c.name.startswith("domain.") for c in validator.validate(provider_request, result).checks
    )


async def test_supplied_citations_validate(provider_request: ProviderRequest) -> None:
    result = await FakeProvider().generate(provider_request)
    assert LocalValidator().validate(provider_request, result).passed
    invented = replace(result, citations=(Citation(document_id="invented", chunk_id="unknown"),))
    assert not LocalValidator().validate(provider_request, invented).passed
    inline = replace(result, content="SYNTHETIC [invented/unknown]")
    assert not LocalValidator().validate(provider_request, inline).passed
    assert not LocalValidator().validate(replace(provider_request, context=()), result).passed


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
async def test_empty_output_rejected(provider_request: ProviderRequest, content: str) -> None:
    result = replace(await FakeProvider().generate(provider_request), content=content)
    assert not LocalValidator().validate(provider_request, result).passed


@pytest.mark.parametrize(
    "content,valid",
    [
        ('{"synthetic": true}', True),
        ("{}", True),
        ("[]", False),
        ("null", False),
        ("{", False),
        ('{"value": NaN}', False),
    ],
)
async def test_json_object_validation(
    provider_request: ProviderRequest, content: str, valid: bool
) -> None:
    request = replace(provider_request, response_format=ResponseFormat(type="json_object"))
    result = replace(await FakeProvider().generate(request), content=content)
    assert LocalValidator().validate(request, result).passed is valid


def test_check_severity_defaults_to_hard_for_legacy_records() -> None:
    assert ValidationCheck.model_validate({"name": "synthetic", "passed": False}).severity == "hard"


@pytest.mark.parametrize("severity,passed", [("hard", False), ("advisory", True)])
async def test_application_override_hardens_or_relaxes_check(
    tmp_path: Path,
    provider_request: ProviderRequest,
    severity: str,
    passed: bool,
) -> None:
    (tmp_path / "synthetic-app.json").write_text(
        json.dumps(
            {
                "severity": {"groundedness": severity},
            }
        )
    )
    validator = LocalValidator(tmp_path)
    request = replace(provider_request, application_id="synthetic-app")
    result = replace(
        await FakeProvider().generate(request),
        content="SYNTHETIC paraphrased answer.",
    )
    validation = validator.validate(request, result)
    assert validation.passed is passed
    assert next(c for c in validation.checks if c.name == "groundedness").severity == severity
    # The configured override cannot harden another application.
    assert validator.validate(provider_request, result).passed


async def test_domain_severity_overrides_and_hard_failure_dominates(
    tmp_path: Path,
    provider_request: ProviderRequest,
) -> None:
    path = tmp_path / "synthetic-app.json"
    tests = [
        {"name": "required", "version": "v1", "kind": "required_text", "value": "SYNTHETIC_ABSENT"},
        {"name": "forbidden", "version": "v1", "kind": "forbidden_text", "value": "SYNTHETIC"},
        {"name": "schema", "version": "v1", "kind": "required_json_key", "value": "answer"},
    ]
    request = replace(provider_request, application_id="synthetic-app")
    result = await FakeProvider().generate(request)
    path.write_text(
        json.dumps(
            {
                "tests": tests,
                "severity": {"domain.forbidden": "advisory", "domain.schema": "advisory"},
            }
        )
    )
    validation = LocalValidator(tmp_path).validate(request, result)
    assert validation.passed
    assert len([c for c in validation.checks if not c.passed and c.severity == "advisory"]) == 3
    path.write_text(
        json.dumps(
            {
                "tests": tests,
                "severity": {
                    "domain.required": "hard",
                    "domain.forbidden": "advisory",
                    "domain.schema": "advisory",
                },
            }
        )
    )
    assert not LocalValidator(tmp_path).validate(request, result).passed


@pytest.mark.parametrize("name", ["groundednes", "domain.missing", "required_text", "unknown"])
def test_unknown_severity_names_rejected(tmp_path: Path, name: str) -> None:
    (tmp_path / "synthetic-app.json").write_text(json.dumps({"severity": {name: "hard"}}))
    with pytest.raises(ValidationError, match="unknown_validation_check"):
        LocalValidator(tmp_path)


async def test_malformed_json_remains_hard_when_truncated(
    provider_request: ProviderRequest,
) -> None:
    request = replace(provider_request, response_format=ResponseFormat(type="json_object"))
    result = replace(await FakeProvider().generate(request), content="{", finish_reason="length")
    validation = LocalValidator().validate(request, result)
    assert not validation.passed
    checks = {c.name: c for c in validation.checks}
    assert checks["json_object"].severity == "hard" and not checks["json_object"].passed
    assert checks["truncation"].severity == "advisory" and not checks["truncation"].passed
