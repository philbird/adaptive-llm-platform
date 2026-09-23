"""Reusable provider conformance suite; register each adapter in the provider fixture."""

import json
from dataclasses import replace
from typing import Literal

import pytest

from adaptive_llm.contracts import ResponseFormat
from adaptive_llm.providers import TOKENIZER, FakeProvider, Provider, ProviderRequest, token_count


@pytest.fixture(params=[FakeProvider], ids=["deterministic_fake"])
def provider(request: pytest.FixtureRequest) -> Provider:
    return request.param()


async def test_provider_returns_canonical_result(
    provider: Provider, provider_request: ProviderRequest
) -> None:
    result = await provider.generate(provider_request)
    assert result.content.strip()
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0
    assert result.usage.input_tokens > 0
    assert 0 < result.usage.output_tokens <= provider_request.max_output_tokens
    assert {(c.document_id, c.chunk_id) for c in result.citations} == {
        (c.document_id, c.chunk_id) for c in provider_request.context
    }


async def test_provider_token_limit(provider: Provider, provider_request: ProviderRequest) -> None:
    result = await provider.generate(replace(provider_request, max_output_tokens=1))
    assert result.finish_reason == "length"
    assert result.usage.output_tokens <= 1


async def test_provider_json(provider: Provider, provider_request: ProviderRequest) -> None:
    result = await provider.generate(
        replace(provider_request, response_format=ResponseFormat(type="json_object"))
    )
    assert isinstance(json.loads(result.content), dict)


async def test_fake_is_deterministic_and_accounts_for_tokens(
    provider_request: ProviderRequest,
) -> None:
    provider = FakeProvider()
    result = await provider.generate(provider_request)
    assert await provider.generate(provider_request) == result
    assert result.usage.source == "locally_estimated"
    assert result.usage.tokenizer == TOKENIZER
    assert result.usage.output_tokens == token_count(result.content)
    assert result.usage.input_tokens == provider_request.input_tokens
    assert result.usage.cached_input_tokens is None


@pytest.mark.parametrize("failure", ["error", "deadline_exceeded"])
async def test_fake_failure_control(
    provider_request: ProviderRequest, failure: Literal["error", "deadline_exceeded"]
) -> None:
    result = await FakeProvider(test_only_failure=failure).generate(provider_request)
    assert result.finish_reason == failure
    assert result.content == ""
    assert result.citations == ()
    assert result.usage.output_tokens == 0
