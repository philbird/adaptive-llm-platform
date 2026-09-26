import asyncio
import json
import logging
import traceback
from dataclasses import replace

import httpx
import pytest

from adaptive_llm.contracts import Message, ResponseFormat
from adaptive_llm.providers import ProviderError, render_context
from adaptive_llm.providers.openrouter import OpenRouterProvider, strip_json_fence
from adaptive_llm.validation import LocalValidator

MODEL = "anthropic/claude-haiku-4.5"
KEY = "SYNTHETIC-PRIVATE-KEY"
PRIVATE = "SYNTHETIC-UPSTREAM-PRIVATE"


def completion(content='{"ok":true}', finish="stop", details=True):
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    if details:
        usage.update(
            prompt_tokens_details={"cached_tokens": 2},
            completion_tokens_details={"reasoning_tokens": 3},
        )
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": usage}


@pytest.mark.parametrize("kind", ["text", "json_object", "json_schema"])
async def test_translation_headers_deadline_context_and_fence(provider_request, kind, caplog):
    caplog.set_level(logging.DEBUG)
    fmt = ResponseFormat.model_validate(
        {
            "type": kind,
            **(
                {"json_schema": {"name": "triage", "schema": {"type": "object"}}}
                if kind == "json_schema"
                else {}
            ),
        }
    )
    request = replace(
        provider_request,
        response_format=fmt,
        deadline_ms=1234,
        max_output_tokens=400,
        messages=(Message(role="system", content="SYNTHETIC system"), *provider_request.messages),
    )

    def transport(sent):
        body = json.loads(sent.content)
        assert str(sent.url) == "https://example.invalid/api/v1/chat/completions"
        assert sent.headers["authorization"] == f"Bearer {KEY}"
        assert sent.headers["HTTP-Referer"] == "https://github.com/philbird/adaptive-llm-platform"
        assert "Adaptive" in sent.headers["X-Title"]
        assert sent.extensions["timeout"]["read"] == 1.234
        assert body["messages"][0] == {"role": "system", "content": "SYNTHETIC system"}
        assert body["messages"][-1] == {"role": "user", "content": render_context(request.context)}
        assert body["provider"] == {"data_collection": "deny"}
        assert body["model"] == MODEL and body["max_tokens"] == 400 and body["temperature"] == 0
        if kind == "json_schema":
            assert body["response_format"]["json_schema"] == {
                "name": "triage",
                "schema": {"type": "object"},
                "strict": True,
            }
        elif kind == "json_object":
            assert body["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in body
        logging.getLogger("httpcore.http11").debug(PRIVATE)
        return httpx.Response(
            200, json=completion('```json\n{"ok":true}\n```'), headers={"X-Private": PRIVATE}
        )

    result = await OpenRouterProvider(
        MODEL,
        KEY,
        base_url="https://example.invalid/api/v1",
        transport=httpx.MockTransport(transport),
    ).generate(request)
    assert result.usage.source == "provider_reported" and result.usage.tokenizer == MODEL
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.cached_input_tokens,
        result.usage.reasoning_tokens,
    ) == (10, 5, 2, 3)
    assert result.latency_ms >= 0
    assert result.content == ('```json\n{"ok":true}\n```' if kind == "text" else '{"ok":true}')
    if kind != "text":
        assert LocalValidator().validate(request, result).passed
    assert KEY not in caplog.text and PRIVATE not in caplog.text
    logging.getLogger("httpx").info("SYNTHETIC ordinary unrelated log")
    assert "ordinary unrelated" in caplog.text


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("stop", "stop"),
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("length", "length"),
        ("max_tokens", "length"),
        ("content_filter", "content_filter"),
        (None, "stop"),
        ("tool_calls", "stop"),
        ("upstream_variant", "stop"),
    ],
)
async def test_finish_usage_and_citations(provider_request, reason, expected):
    c = provider_request.context[0]
    content = f"SYNTHETIC [{c.document_id}/{c.chunk_id}]"
    provider = OpenRouterProvider(
        MODEL,
        KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion(content, reason, False))
        ),
    )
    result = await provider.generate(provider_request)
    assert result.finish_reason == expected
    assert result.usage.cached_input_tokens is None and result.usage.reasoning_tokens is None
    assert [(c.document_id, c.chunk_id) for c in result.citations] == [(c.document_id, c.chunk_id)]


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "provider_unavailable"),
        (429, "provider_rate_limited"),
        (503, "provider_unavailable"),
        (408, "provider_timeout"),
        (504, "provider_timeout"),
        (302, "provider_unavailable"),
        (200, "provider_invalid_response"),
    ],
)
async def test_fixed_errors_no_bodies_headers_or_keys(provider_request, status, code, caplog):
    caplog.set_level(logging.DEBUG)
    provider = OpenRouterProvider(
        MODEL,
        KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status,
                text=PRIVATE,
                headers={"secret": KEY},
                extensions={"reason_phrase": PRIVATE.encode()},
            )
        ),
    )
    with pytest.raises(ProviderError) as error:
        await provider.generate(provider_request)
    assert str(error.value) == code
    rendered = "".join(traceback.format_exception(error.value)) + caplog.text
    assert PRIVATE not in rendered and KEY not in rendered


@pytest.mark.parametrize(
    "failure,code",
    [
        (httpx.ReadTimeout, "provider_timeout"),
        (httpx.ConnectError, "provider_unavailable"),
        (RuntimeError, "provider_invalid_response"),
    ],
)
async def test_transport_errors_are_sanitized(provider_request, failure, code):
    def broken(_):
        raise failure(PRIVATE)

    with pytest.raises(ProviderError, match=f"^{code}$"):
        await OpenRouterProvider(MODEL, KEY, transport=httpx.MockTransport(broken)).generate(
            provider_request
        )


@pytest.mark.parametrize(
    "body",
    [
        {},
        completion(content="", finish="tool_calls"),
        completion(content=" ", finish=None),
        {
            **completion(),
            "choices": [
                {"message": {"content": "{}", "tool_calls": [{}]}, "finish_reason": "stop"}
            ],
        },
        {**completion(), "usage": {"prompt_tokens": -1, "completion_tokens": 0}},
        {**completion(), "usage": {"prompt_tokens": True, "completion_tokens": 2}},
        {**completion(), "choices": []},
    ],
)
async def test_malformed_provider_result_is_fixed(provider_request, body):
    with pytest.raises(ProviderError, match="^provider_invalid_response$"):
        await OpenRouterProvider(
            MODEL, KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ).generate(provider_request)


async def test_overall_timeout_and_missing_key(provider_request):
    async def slow(_):
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=completion())

    with pytest.raises(ProviderError, match="^provider_timeout$"):
        await OpenRouterProvider(MODEL, KEY, transport=httpx.MockTransport(slow)).generate(
            replace(provider_request, deadline_ms=1)
        )
    with pytest.raises(ValueError, match="^openrouter_api_key_required$"):
        OpenRouterProvider(MODEL, "")
    assert strip_json_fence("  ```\n{}\n```\n") == "{}"
    assert strip_json_fence("prefix ```json\n{}\n```").startswith("prefix")


async def test_pool_reused_per_loop_and_explicitly_closed(provider_request):
    provider = OpenRouterProvider(
        MODEL, KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion()))
    )
    await asyncio.gather(*(provider.generate(provider_request) for _ in range(4)))
    loop = asyncio.get_running_loop()
    first = provider._clients[loop]
    assert len(provider._clients) == 1 and not first.is_closed
    await provider.generate(provider_request)
    assert provider._clients[loop] is first

    async def other_loop():
        await provider.generate(provider_request)
        second = provider._clients[asyncio.get_running_loop()]
        assert second is not first and len(provider._clients) == 2
        await provider.aclose()
        assert second.is_closed and not first.is_closed

    await asyncio.to_thread(lambda: asyncio.run(other_loop()))
    await provider.aclose()
    await provider.aclose()
    assert first.is_closed and not provider._clients


async def test_evaluation_failure_closes_its_pool(provider_request):
    from adaptive_llm.contracts import EvaluationSpecification
    from adaptive_llm.evaluation.runner import PipelineRunner
    from adaptive_llm.evaluation.service import LocalEvaluator

    provider = OpenRouterProvider(
        MODEL, KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion()))
    )
    clients = []

    class FailingSuite:
        async def run(self, runner, cases, specification):
            await runner.provider.generate(provider_request)
            clients.extend(provider._clients.values())
            raise RuntimeError("synthetic_evaluation_failure")

    evaluator = object.__new__(LocalEvaluator)
    evaluator.suites = {"golden": FailingSuite()}
    runner = object.__new__(PipelineRunner)
    runner.provider = provider
    spec = EvaluationSpecification(
        candidate_deployment_id="fake",
        baseline_deployment_id=None,
        dataset_id="synthetic",
        dataset_version="synthetic",
        suites=["golden"],
    )
    with pytest.raises(RuntimeError, match="synthetic_evaluation_failure"):
        await evaluator._run(spec, {"golden": []}, runner)
    assert clients and all(client.is_closed for client in clients) and not provider._clients
