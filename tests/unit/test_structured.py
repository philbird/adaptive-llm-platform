import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import (
    InferenceRequest,
    Message,
    PolicyDecision,
    ResponseFormat,
    RoutePolicy,
    RoutingOptions,
)
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.providers import FakeProvider
from adaptive_llm.routing import FoundationRouter
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.chain import ChainPlan, ExecutionCandidate, run_chain
from adaptive_llm.routing.tasks import RulesClassifier
from adaptive_llm.structured import canonical, matches_schema
from adaptive_llm.validation import LocalValidator


def format_for(schema):
    return ResponseFormat.model_validate(
        {"type": "json_schema", "json_schema": {"name": "triage", "schema": schema}}
    )


def test_system_role_bounds_and_order(inference_request):
    data = inference_request.model_dump()
    system = {"role": "system", "content": "s" * 8000}
    user = {"role": "user", "content": "u" * 24000}
    assert len(InferenceRequest.model_validate({**data, "messages": [system, user]}).messages) == 2
    for messages in (
        [user, system, user],
        [system, system, user],
        [system],
        [{**system, "content": "s" * 8001}, user],
        [system, {**user, "content": "u" * 24001}],
    ):
        with pytest.raises(ValidationError):
            InferenceRequest.model_validate({**data, "messages": messages})


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "wrong"},
        {"required": "x"},
        {"type": 12},
        {"description": "ü" * 4096},
        {"$ref": "https://example.invalid/schema"},
        {"$defs": {"unused": {"$dynamicRef": "file:///private/tmp/private.json"}}},
    ],
)
def test_bad_schema_is_rejected_at_ingress(schema):
    with pytest.raises(ValidationError, match="invalid_json_schema"):
        format_for(schema)


def test_schema_format_pairing_hash_and_local_references():
    for body in (
        {"type": "json_schema"},
        {"type": "text", "json_schema": {"name": "a", "schema": {}}},
        {"type": "json_schema", "json_schema": {"name": "a", "schema": []}},
    ):
        with pytest.raises(ValidationError):
            ResponseFormat.model_validate(body)
    a = format_for(
        {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    )
    b = format_for(
        {"properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "type": "object"}
    )
    assert a.json_schema.sha256 == b.json_schema.sha256
    assert ResponseFormat.model_validate_json(a.model_dump_json()) == a
    schema = {"$defs": {"value": {"type": "boolean"}}, "$ref": "#/$defs/value"}
    assert matches_schema("true", format_for(schema).json_schema.schema_)
    assert not matches_schema("1", schema)
    assert not matches_schema("NaN", {})
    assert not matches_schema("true", {"$ref": "https://example.invalid/no-network"})
    assert canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'


async def test_json_schema_hard_validation_and_fallback(provider_request, settings, tmp_path):
    schema = format_for(
        {
            "type": "object",
            "properties": {"ok": {"const": True}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    )
    request = replace(provider_request, context=(), response_format=schema)
    result = await FakeProvider().generate(request)
    path = tmp_path / "support-assistant.json"
    path.write_text(json.dumps({"severity": {"json_schema": "advisory"}}))
    validator = LocalValidator(tmp_path)
    for content in ('{"ok":false}', '{"ok":true,"extra":1}', "[]", "NaN", ""):
        check = validator.validate(request, replace(result, content=content))
        assert not check.passed
        assert next(c for c in check.checks if c.name == "json_schema").severity == "hard"
    assert validator.validate(request, replace(result, content='{"ok":true}')).passed
    foundation = FoundationRouter(settings.routing_path).deployment

    class Fixed:
        def __init__(self, content):
            self.content = content

        async def generate(self, request):
            return replace(result, content=self.content)

    first = ExecutionCandidate(
        foundation.model_copy(update={"model_deployment_id": "bad"}), Fixed('{"ok":false}')
    )
    last = ExecutionCandidate(foundation, Fixed('{"ok":true}'))
    attempts = []
    metrics = InProcessMetrics()
    options = RoutingOptions()
    policy = PolicyDecision(policy_version="test", retention_seconds=1)
    final, _ = await run_chain(
        ChainPlan((first, last), RoutePolicy()),
        request,
        options,
        policy,
        "test",
        perf_counter() + 1,
        attempts,
        validator,
        CircuitBreakers(metrics),
        metrics,
    )
    assert final.content == '{"ok":true}'
    assert attempts[0].error_code == "validation_failed"
    with pytest.raises(GatewayError, match="^validation_failed$"):
        await run_chain(
            ChainPlan((first,), RoutePolicy()),
            request,
            options,
            policy,
            "test",
            perf_counter() + 1,
            [],
            validator,
            CircuitBreakers(metrics),
            metrics,
        )


async def test_system_tokens_and_fake_behavior(provider_request):
    messages = (
        Message(role="system", content="SYNTHETIC system policy"),
        *provider_request.messages,
    )
    with_system = replace(provider_request, messages=messages)
    old = await FakeProvider().generate(provider_request)
    new = await FakeProvider().generate(with_system)
    assert old.content == new.content
    assert new.usage.input_tokens == old.usage.input_tokens + 3


def test_rules_specific_fallback_and_unmapped(inference_request, tmp_path):
    classifier = RulesClassifier(Path("configs/tasks/manyfails.json"))
    request = inference_request.model_copy(
        update={"application_id": "research-sweep", "response_format": format_for({})}
    )
    task = classifier.classify(request)
    assert (task.label, task.language, task.risk_tier, task.classifier_version) == (
        "candidate_triage",
        "en",
        "low",
        "rules-1",
    )
    assert task.reason_codes == ["application_task_map"]
    assert classifier.classify(inference_request).label == "question_answering"
    plain = InferenceRequest(
        request_id="test",
        application_id="research-sweep",
        messages=[Message(role="user", content="SYNTHETIC")],
    )
    assert classifier.classify(plain).reason_codes == ["rag_flag_only"]
    assert classifier.classify(plain).label == "general"
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "application_id": "research-sweep",
                        "label": "generic",
                        "language": "en",
                        "risk_tier": "medium",
                    }
                ]
            }
        )
    )
    assert RulesClassifier(path).classify(request).label == "generic"


def test_specialist_generation_template_receives_system_role(provider_request, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from adaptive_llm.training import lora

    generator = object.__new__(lora.LoraGenerator)
    generator.libs = SimpleNamespace(torch=MagicMock())
    generator.libs.torch.inference_mode.return_value = nullcontext()
    generator.manifest = SimpleNamespace(seed=23, tokenizer_id="synthetic")
    generator.tokenizer = MagicMock()
    generator.tokenizer.apply_chat_template.return_value = [1, 2]
    generator.tokenizer.eos_token_id = 9
    generator.tokenizer.decode.return_value = "SYNTHETIC output"
    generator.model = MagicMock()
    generator.model.generate.return_value.__getitem__.return_value.tolist.return_value = [3, 9]
    generator.limit = 100
    monkeypatch.setattr(lora, "cpu", lambda *args: nullcontext())
    request = replace(
        provider_request,
        messages=(
            Message(role="system", content="SYNTHETIC system policy"),
            *provider_request.messages,
        ),
    )
    result = generator.generate(request)
    messages = generator.tokenizer.apply_chat_template.call_args.args[0]
    assert messages[0] == {"role": "system", "content": "SYNTHETIC system policy"}
    assert messages[1] == {"role": "user", "content": provider_request.messages[0].content}
    assert result.finish_reason == "stop"
