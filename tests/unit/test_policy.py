from dataclasses import replace

from adaptive_llm.app import Settings
from adaptive_llm.contracts import InferenceRequest, Message
from adaptive_llm.gateway.identity import Identity
from adaptive_llm.policy import LocalPolicyEngine, ProcessingRedactor


def test_policy_defaults_and_unknown_tenant(settings: Settings, identity: Identity) -> None:
    engine = LocalPolicyEngine(settings.policy_path)
    policy = engine.decide(identity, "support-assistant")
    assert policy.processing_allowed
    assert not policy.content_logging_allowed
    assert not policy.evaluation_allowed
    assert not policy.human_review_allowed
    assert not policy.training_allowed
    assert not engine.decide(
        replace(identity, tenant_id="unknown"), "support-assistant"
    ).processing_allowed
    assert not engine.decide(identity, "forbidden-application").processing_allowed


def test_processing_redaction_preserves_business_identifiers(
    inference_request: InferenceRequest,
) -> None:
    original = inference_request.model_copy(
        update={
            "messages": [
                Message(
                    role="user",
                    content=(
                        "SYNTHETIC: order ORD-12345 reference 1234567890123456 "
                        "for demo@example.test; "
                        "sk-synthetic123456789 and 4111 1111 1111 1111; password=synthetic-pass "
                        "Bearer synthetic-credential api_key='synthetic value'"
                    ),
                )
            ]
        }
    )
    result, counts = ProcessingRedactor().redact(original)
    content = result.messages[0].content
    assert "ORD-12345" in content
    assert "1234567890123456" in content
    assert "demo@example.test" in content
    for secret in [
        "sk-synthetic123456789",
        "4111",
        "synthetic-pass",
        "synthetic-credential",
        "synthetic value",
    ]:
        assert secret not in content
    assert counts == {"credentials": 1, "secrets": 2, "api_keys": 1, "card_numbers": 1}
    assert ProcessingRedactor().redact(result) == (result, {})
    assert "sk-synthetic123456789" in original.messages[0].content


def test_metadata_values_use_the_same_processing_redaction(
    inference_request: InferenceRequest,
) -> None:
    content = (
        "SYNTHETIC ORD-12345 sk-synthetic123456789 4111 1111 1111 1111; "
        "password=synthetic-pass Bearer synthetic-credential"
    )
    original = inference_request.model_copy(
        update={
            "messages": [Message(role="user", content=content)],
            "metadata": {"note": content, "order": "ORD-12345"},
        }
    )
    result, counts = ProcessingRedactor().redact(original)
    assert result.metadata["note"] == result.messages[0].content
    assert result.metadata["order"] == "ORD-12345"
    assert "ORD-12345" in result.metadata["note"]
    assert counts == {"credentials": 2, "secrets": 2, "api_keys": 2, "card_numbers": 2}
    for secret in ("sk-synthetic123456789", "4111", "synthetic-pass", "synthetic-credential"):
        assert secret not in result.metadata["note"]
    assert original.metadata["note"] == content
    assert ProcessingRedactor().redact(result) == (result, {})
