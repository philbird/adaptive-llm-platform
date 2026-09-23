from adaptive_llm.contracts import InferenceRequest, Message, PolicyDecision
from adaptive_llm.policy import ProcessingRedactor
from adaptive_llm.policy.persistence import LocalPersistenceRedactor


def test_two_pass_redaction_preserves_processing_identifiers() -> None:
    text = (
        "SYNTHETIC ORD-12345 user@example.test +44 7700 900123 "
        "sk-synthetic123456789 4111 1111 1111 1111 password=synthetic-secret"
    )
    request = InferenceRequest(
        request_id="synthetic",
        application_id="synthetic",
        messages=[Message(role="user", content=text)],
    )
    processing, counts = ProcessingRedactor().redact(request)
    assert counts == {"api_keys": 1, "card_numbers": 1, "secrets": 1}
    assert "user@example.test" in processing.messages[0].content
    assert "+44 7700 900123" in processing.messages[0].content
    redactor = LocalPersistenceRedactor()
    policy = PolicyDecision(policy_version="synthetic", retention_seconds=60)
    redacted, pii_counts = redactor.redact_text(processing.messages[0].content, policy)
    assert pii_counts == {"emails": 1, "phones": 1}
    assert "ORD-12345" in redacted
    assert "example.test" not in redacted
    assert "7700" not in redacted
    assert redactor.version != ProcessingRedactor.version
    assert redactor.redact_text(text, policy)[0] == redacted
