from uuid import UUID

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import (
    Feedback,
    FeedbackValue,
    GenerationAttempt,
    InferenceRequest,
    Message,
    Started,
    Usage,
    now,
    uid,
)


def request(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "request_id": "req-1",
        "application_id": "synthetic-app",
        "messages": [{"role": "user", "content": "synthetic"}],
    }
    body.update(overrides)
    return body


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
        InferenceRequest(
            request_id="test",
            application_id="synthetic-app",
            messages=[Message(role="user", content="x" * 32_001)],
        )
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(
            request(messages=[{"role": "tool", "content": "synthetic"}])
        )
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(request(tenant_id="untrusted-client-tenant"))


def test_requests_carry_no_demo_defaults() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(
            {"request_id": "r", "messages": [{"role": "user", "content": "x"}]}
        )
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(request(rag={"enabled": True}))
    parsed = InferenceRequest.model_validate(request())
    assert parsed.rag.enabled is False
    assert parsed.metadata == {}


def test_ingress_rejects_unknown_fields_but_records_ignore_them() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(request(surprise=True))
    started = Started.model_validate(
        {"interaction_id": "i", "application_id": "a", "policy_version": "p", "added_later": 1}
    )
    assert started.interaction_id == "i"
    assert "added_later" not in started.model_dump()


def test_finish_reasons_cover_cancellation_and_deadline() -> None:
    for reason in ("cancelled", "deadline_exceeded"):
        attempt = GenerationAttempt(
            interaction_id="i",
            model_provider="fake",
            model_id="fake-1",
            model_version="1",
            deployment_id="foundation-primary",
            total_latency_ms=1.0,
            finish_reason=reason,
        )
        assert attempt.finish_reason == reason


def test_automated_feedback_declares_judge_and_never_masquerades_as_human() -> None:
    value = FeedbackValue(score=4, max_score=5)
    with pytest.raises(ValidationError):
        Feedback(interaction_id="i", source="automated", label_type="rubric", value=value)
    with pytest.raises(ValidationError):
        Feedback(
            interaction_id="i",
            source="reviewer",
            label_type="rubric",
            value=value,
            judge_version="judge-1",
        )
    with pytest.raises(ValidationError):
        FeedbackValue(score=6, max_score=5)
    automated = Feedback(
        interaction_id="i",
        source="automated",
        label_type="rubric",
        value=value,
        judge_version="judge-1",
    )
    assert automated.actor_id_pseudonymous is None
