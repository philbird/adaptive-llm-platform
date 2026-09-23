from uuid import UUID

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import InferenceRequest, Message, Usage, now, uid


def test_identifiers_are_uuid7_and_time_is_utc() -> None:
    identifiers = {uid() for _ in range(1_000)}
    assert len(identifiers) == 1_000
    assert all(UUID(value).version == 7 for value in identifiers)
    assert now().utcoffset().total_seconds() == 0


def test_unknown_provider_counts_remain_null() -> None:
    usage = Usage(input_tokens=10, output_tokens=3, source="provider_reported", tokenizer="fake-1")
    assert usage.cached_input_tokens is None
    assert usage.reasoning_tokens is None
    with pytest.raises(ValidationError):
        Usage(
            input_tokens=1,
            output_tokens=0,
            cached_input_tokens=2,
            source="provider_reported",
            tokenizer="fake-1",
        )


def test_requests_are_bounded_and_tool_roles_not_accepted() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(request_id="test", messages=[Message(role="user", content="x" * 32_001)])
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(
            {
                "request_id": "test",
                "messages": [
                    {"role": "tool", "content": "synthetic"},
                ],
            }
        )
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(
            {
                "request_id": "test",
                "messages": [
                    {"role": "user", "content": "synthetic"},
                ],
                "tenant_id": "untrusted-client-tenant",
            }
        )
