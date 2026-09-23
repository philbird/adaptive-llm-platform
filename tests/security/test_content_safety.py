from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from opentelemetry.trace import Tracer

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Event, InferenceRequest, RagOptions, Region
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.gateway.identity import Identity, Keyring
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.rag import LocalRetriever, RetrievalResult

HEADERS = {"Authorization": "Bearer synthetic-key-a", "X-Subject": "synthetic-subject"}


def test_redaction_precedes_retrieval_and_generation_and_telemetry_is_content_free(
    inference_request: InferenceRequest,
    settings: Settings,
    keyring: Keyring,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "SYNTHETIC: receipt ORD-12345 sk-synthetic123456789 4111 1111 1111 1111"
    query_seen: list[str] = []
    request_seen: list[ProviderRequest] = []

    class InspectRetriever(LocalRetriever):
        async def retrieve(
            self,
            query: str,
            identity: Identity,
            application_id: str,
            options: RagOptions,
            interaction_id: str,
            *,
            residency: Region,
        ) -> RetrievalResult:
            query_seen.append(query)
            return await super().retrieve(
                query, identity, application_id, options, interaction_id, residency=residency
            )

    class InspectProvider(FakeProvider):
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            request_seen.append(request)
            return await super().generate(request)

    sink, tracer = InMemoryEventSink(), MagicMock(spec=Tracer)
    app = create_app(
        Settings(
            events=sink,
            tracer=tracer,
            provider=InspectProvider(),
            retriever=InspectRetriever(settings.documents_path, keyring),
        )
    )
    body = inference_request.model_dump()
    body["messages"][0]["content"] = prompt
    body["metadata"] = {"tenant_id": "synthetic-b", "environment": "production"}
    with TestClient(app) as client:
        response = client.post("/v1/inference", json=body, headers=HEADERS)
    assert response.status_code == 200
    assert len(query_seen) == len(request_seen) == 1
    assert query_seen[0] == request_seen[0].messages[-1].content
    assert "ORD-12345" in query_seen[0]
    assert "sk-synthetic123456789" not in query_seen[0]
    assert "4111" not in query_seen[0]
    assert all(e.tenant_id == "synthetic-a" for e in sink.events)
    interaction = sink.events[-1].data
    assert interaction.environment == "local"
    assert interaction.policy.processing_redaction_counts == {"api_keys": 1, "card_numbers": 1}
    assert interaction.policy.persistence_redaction_counts == {}
    telemetry = (
        " ".join(e.model_dump_json() for e in sink.events) + str(tracer.mock_calls) + caplog.text
    )
    for sensitive in [
        prompt,
        "ORD-12345",
        "sk-synthetic123456789",
        "4111 1111",
        "synthetic-subject",
        response.json()["content"],
    ]:
        assert sensitive not in telemetry
    calls = tracer.start_as_current_span.call_args_list
    assert {call.args[0] for call in calls} >= {
        "policy",
        "retrieval",
        "routing",
        "generation",
        "validation",
        "event_emission",
    }
    for call in calls:
        assert call.kwargs["record_exception"] is False
        assert call.kwargs["set_status_on_exception"] is False
        assert all(name.endswith(("_id", "_version")) for name in call.kwargs["attributes"])


def test_provider_exceptions_never_reach_traces_events_or_http(
    inference_request: InferenceRequest, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenProvider(FakeProvider):
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            raise RuntimeError("SYNTHETIC-private-provider-error-body")

    sink, tracer = InMemoryEventSink(), MagicMock(spec=Tracer)
    with TestClient(
        create_app(Settings(provider=BrokenProvider(), events=sink, tracer=tracer))
    ) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
    assert result.status_code == 502
    assert result.json() == {"error": {"code": "provider_failed"}}
    # Inspect attributes, not the context manager's Python exception arguments: those
    # stay in memory and automatic OTel exception recording is explicitly disabled.
    calls = tracer.start_as_current_span.call_args_list
    telemetry = (
        result.text + caplog.text + str(calls) + " ".join(e.model_dump_json() for e in sink.events)
    )
    assert "SYNTHETIC-private-provider-error-body" not in telemetry
    assert all(call.kwargs["record_exception"] is False for call in calls)
    assert sink.events[3].event_type == "generation.failed.v1"


def test_sink_failure_and_full_queue_never_fail_serving(
    inference_request: InferenceRequest,
) -> None:
    class BrokenSink:
        def emit(self, event: Event) -> None:
            raise RuntimeError("synthetic-sink-unavailable")

    app = create_app(Settings(events=BrokenSink()))
    with TestClient(app) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
    assert result.status_code == 200
    assert app.state.inference.emission_failures == 5
    sink = InMemoryEventSink(capacity=1)
    with TestClient(create_app(Settings(events=sink))) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
    assert result.status_code == 200
    assert sink.dropped_events == 4


@pytest.mark.parametrize("field", ["tenant_id", "subject_id", "environment", "test_only_failure"])
def test_body_cannot_supply_identity_or_test_controls(
    inference_request: InferenceRequest, field: str
) -> None:
    with TestClient(create_app()) as client:
        result = client.post(
            "/v1/inference",
            headers=HEADERS,
            json={**inference_request.model_dump(), field: "SYNTHETIC-secret-invalid-input"},
        )
    assert result.status_code == 422
    assert result.json() == {"error": {"code": "invalid_request"}}


def test_unknown_citation_is_rejected_without_leaking_output(
    inference_request: InferenceRequest,
) -> None:
    class InventedCitation(FakeProvider):
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            return replace(await super().generate(request), content="SYNTHETIC [unknown/secret]")

    with TestClient(create_app(Settings(provider=InventedCitation()))) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
    assert result.status_code == 502
    assert result.json() == {"error": {"code": "validation_failed"}}
