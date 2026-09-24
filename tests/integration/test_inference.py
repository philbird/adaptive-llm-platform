import asyncio
import json
from dataclasses import replace
from typing import Literal
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    GenerationAttempt,
    InferenceRequest,
    InferenceResponse,
    Interaction,
    PolicyDecision,
    RagOptions,
    Region,
    RetrievalRun,
    RouteDecision,
)
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.gateway.identity import Identity, Keyring, LocalAuthenticator
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.rag import RetrievalResult

HEADERS = {"Authorization": "Bearer synthetic-key-a", "X-Subject": "synthetic-subject"}
EVENT_SEQUENCE = [
    "interaction.started.v1",
    "retrieval.completed.v1",
    "route.decided.v1",
    "generation.completed.v1",
    "interaction.completed.v1",
]


def test_end_to_end_correlated_events_and_cost(
    inference_request: InferenceRequest, keyring: Keyring
) -> None:
    sink = InMemoryEventSink()
    app = create_app(Settings(events=sink))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
        client.app.state.dispatcher.dispatch_once()
    assert response.status_code == 200
    assert app.state.metrics.get("dispatcher_failures") == 0
    assert sink.dropped_events == 0
    result = InferenceResponse.model_validate(response.json())
    assert result.citations[0].chunk_id == "refund-window"
    assert result.usage.source == "locally_estimated"
    assert (
        result.estimated_cost_micros == result.usage.input_tokens + 2 * result.usage.output_tokens
    )
    events = sink.events_for_trace(result.trace_id)
    assert [e.event_type for e in events] == EVENT_SEQUENCE
    assert {e.data.interaction_id for e in events} == {result.interaction_id}
    assert {e.tenant_id for e in events} == {"synthetic-a"}
    assert all(UUID(e.event_id).version == 7 for e in events)
    assert UUID(result.interaction_id).version == UUID(result.trace_id).version == 7
    assert all(e.occurred_at.utcoffset().total_seconds() == 0 for e in events)
    retrieval, route, generation, interaction = (e.data for e in events[1:])
    assert isinstance(retrieval, RetrievalRun)
    assert isinstance(route, RouteDecision)
    assert isinstance(generation, GenerationAttempt)
    assert isinstance(interaction, Interaction)
    assert events[0].occurred_at == interaction.started_at
    assert events[0].occurred_at < interaction.completed_at
    assert interaction.retrieval_run_id == retrieval.retrieval_run_id
    assert interaction.route_decision_id == route.route_decision_id
    assert interaction.generation_attempt_ids == [generation.attempt_id]
    assert interaction.final_attempt_id == generation.attempt_id
    assert interaction.status == "completed"
    assert interaction.task.classifier_version == "placeholder-rag-flag-1"
    assert interaction.task.confidence == 0.5
    assert interaction.task.reason_codes == ["rag_flag_only"]
    assert generation.validation.passed
    assert generation.estimated_cost_micros == result.estimated_cost_micros
    assert generation.price_list_version == route.candidates[0].price_list_version
    assert generation.output_ref is None
    assert interaction.input.messages_ref is None
    assert not interaction.policy.content_logging_allowed
    assert interaction.subject_id_pseudonymous != HEADERS["X-Subject"]
    assert interaction.subject_id_pseudonymous == keyring.pseudonym(
        "synthetic-a", HEADERS["X-Subject"]
    )
    assert retrieval.query_hash == keyring.content_hash(
        inference_request.messages[-1].content, purpose="query"
    )
    assert generation.output_hash == keyring.content_hash(result.content, purpose="output")
    assert interaction.input.content_hash == keyring.content_hash(
        json.dumps([message.model_dump(mode="json") for message in inference_request.messages]),
        purpose="input",
    )


@pytest.mark.parametrize("key,status", [(None, 401), ("unknown", 401), ("synthetic-key-a", 403)])
def test_identity_rejections(
    inference_request: InferenceRequest, key: str | None, status: int
) -> None:
    body = inference_request.model_dump()
    if status == 403:
        body["application_id"] = "forbidden-application"
    sink = InMemoryEventSink()
    with TestClient(create_app(Settings(events=sink))) as client:
        result = client.post(
            "/v1/inference", json=body, headers={"Authorization": f"Bearer {key}"} if key else {}
        )
        client.app.state.dispatcher.dispatch_once()
    assert result.status_code == status
    assert not sink.events


def test_streaming_rejected(inference_request: InferenceRequest) -> None:
    with TestClient(create_app()) as client:
        result = client.post(
            "/v1/inference",
            json={**inference_request.model_dump(), "stream": True},
            headers=HEADERS,
        )
    assert result.status_code == 501


def test_replay_conflict_and_tenant_scope(inference_request: InferenceRequest) -> None:
    sink = InMemoryEventSink()
    with TestClient(create_app(Settings(events=sink))) as client:
        original = client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
        client.app.state.dispatcher.dispatch_once()
        replay = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
        assert replay.json() == {**original.json(), "replayed": True}
        assert len(sink.events) == 5
        conflict = client.post(
            "/v1/inference",
            json={**inference_request.model_dump(), "max_output_tokens": 64},
            headers=HEADERS,
        )
        client.app.state.dispatcher.dispatch_once()
        assert conflict.status_code == 409
        assert conflict.json() == {"error": {"code": "request_id_conflict"}}
        other = client.post(
            "/v1/inference",
            json=inference_request.model_dump(),
            headers={"Authorization": "Bearer synthetic-key-b"},
        )
        client.app.state.dispatcher.dispatch_once()
        assert other.status_code == 200
        assert not other.json()["replayed"]
        assert other.json()["interaction_id"] != original.json()["interaction_id"]
        assert other.json()["citations"][0]["chunk_id"] == "other-refund-window"
        assert len(sink.events) == 10


def test_application_scoped_replay(
    inference_request: InferenceRequest, settings: Settings, keyring: Keyring
) -> None:
    class TwoApplications:
        def authenticate(self, authorization: str | None, subject: str | None) -> Identity:
            identity = LocalAuthenticator(settings.identity_path, keyring).authenticate(
                authorization, subject
            )
            return replace(identity, application_ids=frozenset({"support-assistant", "second-app"}))

    with TestClient(create_app(Settings(authenticator=TwoApplications()))) as client:
        original = client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
        other = client.post(
            "/v1/inference",
            json={**inference_request.model_dump(), "application_id": "second-app"},
            headers=HEADERS,
        )
        assert other.status_code == 200
        assert other.json()["interaction_id"] != original.json()["interaction_id"]
        assert not other.json()["replayed"]
        assert other.json()["citations"] == []


@pytest.mark.parametrize("failure,status", [("error", 502), ("deadline_exceeded", 504)])
def test_provider_failure_events(
    inference_request: InferenceRequest, failure: Literal["error", "deadline_exceeded"], status: int
) -> None:
    sink = InMemoryEventSink()
    with TestClient(
        create_app(Settings(provider=FakeProvider(test_only_failure=failure), events=sink))
    ) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
        retry = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
    assert result.status_code == retry.status_code == status
    assert len(sink.events) == 10
    for events in (sink.events[:5], sink.events[5:]):
        assert [e.event_type for e in events] == [
            *EVENT_SEQUENCE[:3],
            "generation.failed.v1",
            EVENT_SEQUENCE[-1],
        ]
        assert len({e.trace_id for e in events}) == 1
        assert len({e.data.interaction_id for e in events}) == 1
        assert events[3].data.finish_reason == failure
        assert events[-1].data.status == "failed"
    assert sink.events[0].trace_id != sink.events[5].trace_id
    assert sink.events[0].data.interaction_id != sink.events[5].data.interaction_id


def test_validation_failure_is_content_free(inference_request: InferenceRequest) -> None:
    class EmptyProvider(FakeProvider):
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            return replace(await super().generate(request), content="")

    sink = InMemoryEventSink()
    with TestClient(create_app(Settings(provider=EmptyProvider(), events=sink))) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
    assert result.status_code == 502
    assert result.json() == {"error": {"code": "validation_failed"}}
    assert sink.events[3].event_type == "generation.failed.v1"
    assert not sink.events[3].data.validation.passed


def test_json_no_rag_and_length(inference_request: InferenceRequest) -> None:
    class MustNotRetrieve:
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
            pytest.fail("disabled RAG reached retriever")

    sink = InMemoryEventSink()
    with TestClient(create_app(Settings(events=sink, retriever=MustNotRetrieve()))) as client:
        result = client.post(
            "/v1/inference",
            headers=HEADERS,
            json={
                **inference_request.model_dump(),
                "rag": {"enabled": False},
                "response_format": {"type": "json_object"},
            },
        )
        client.app.state.dispatcher.dispatch_once()
        assert result.status_code == 200
        assert result.json()["citations"] == []
        events = sink.events_for_trace(result.json()["trace_id"])
        assert [event.event_type for event in events] == [EVENT_SEQUENCE[0], *EVENT_SEQUENCE[2:]]
        assert events[-1].data.retrieval_run_id is None
        assert events[-1].data.task.label == "general"
        assert events[-1].data.task.classifier_version == "placeholder-rag-flag-1"
        assert events[-1].data.task.confidence == 0.5
        assert events[-1].data.task.reason_codes == ["rag_flag_only"]
    with TestClient(create_app()) as client:
        short = client.post(
            "/v1/inference",
            headers=HEADERS,
            json={
                **inference_request.model_dump(),
                "request_id": "short",
                "max_output_tokens": 1,
            },
        )
        client.app.state.dispatcher.dispatch_once()
        assert short.status_code == 200
        assert short.json()["finish_reason"] == "length"
        assert short.json()["usage"]["output_tokens"] == 1
        metrics = client.app.state.metrics
        assert metrics.get("validation_advisory_failures", check_name="truncation") == 1
        replay = client.post(
            "/v1/inference",
            headers=HEADERS,
            json={**inference_request.model_dump(), "request_id": "short", "max_output_tokens": 1},
        )
        assert replay.json()["replayed"]
        assert metrics.get("validation_advisory_failures", check_name="truncation") == 1


@pytest.mark.parametrize("constraint", ["processing", "residency", "budget"])
def test_policy_constraints_prevent_provider_call(
    inference_request: InferenceRequest, constraint: str
) -> None:
    class RestrictedPolicy:
        def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
            return PolicyDecision(
                policy_version="synthetic-restricted",
                retention_seconds=1,
                processing_allowed=constraint != "processing",
                residency="eu-west" if constraint == "residency" else "local",
            )

    class MustNotGenerate(FakeProvider):
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            pytest.fail("ineligible request reached provider")

    body = inference_request.model_dump()
    if constraint == "budget":
        body["routing"]["max_cost_micros"] = 0
    sink = InMemoryEventSink()
    with TestClient(
        create_app(Settings(policy=RestrictedPolicy(), provider=MustNotGenerate(), events=sink))
    ) as client:
        result = client.post("/v1/inference", json=body, headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
    status, code = {
        "processing": (403, "processing_forbidden"),
        "residency": (403, "residency_unavailable"),
        "budget": (422, "cost_limit_exceeded"),
    }[constraint]
    assert result.status_code == status
    assert result.json() == {"error": {"code": code}}
    if constraint != "processing":
        assert sink.events[2].data.selected_model_deployment_id is None
        assert sink.events[-1].data.error_code == code


async def test_concurrent_replay_executes_once(inference_request: InferenceRequest) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingProvider(FakeProvider):
        calls = 0

        async def generate(self, request: ProviderRequest) -> ProviderResult:
            self.calls += 1
            entered.set()
            await release.wait()
            return await super().generate(request)

    provider = WaitingProvider()
    app = create_app(Settings(provider=provider))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = asyncio.create_task(
            client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            duplicate = await client.post(
                "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
            )
            assert duplicate.status_code == 409
            assert duplicate.json() == {"error": {"code": "request_in_progress"}}
        finally:
            release.set()
        original = await first
        replay = await client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
    assert original.status_code == 200
    assert replay.json() == {**original.json(), "replayed": True}
    assert provider.calls == 1


def test_deadline_bounds_provider(inference_request: InferenceRequest) -> None:
    class SlowProvider(FakeProvider):
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            await asyncio.sleep(1)
            return await super().generate(request)

    sink = InMemoryEventSink()
    body = inference_request.model_dump()
    body["routing"]["deadline_ms"] = 50
    with TestClient(create_app(Settings(provider=SlowProvider(), events=sink))) as client:
        result = client.post("/v1/inference", json=body, headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
    assert result.status_code == 504
    assert sink.events[3].data.finish_reason == "deadline_exceeded"
    assert sink.events[-1].data.status == "failed"
