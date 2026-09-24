import asyncio
import math
from time import perf_counter
from typing import TYPE_CHECKING

import httpx
import pytest

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import GenerationAttempt, InferenceRequest, Interaction, RouteDecision
from adaptive_llm.events import InMemoryEventSink

if TYPE_CHECKING:
    from conftest import RouterSeed

from routing_helpers import infer, policy, prepare_canary, promote


@pytest.mark.load
async def test_200_requests_have_under_50ms_p95_overhead(
    inference_request: InferenceRequest,
) -> None:
    sink = InMemoryEventSink(capacity=1000)
    app = create_app(Settings(events=sink))
    overheads = []
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        for index in range(200):
            body = {**inference_request.model_dump(), "request_id": f"synthetic-load-{index}"}
            started = perf_counter()
            response = await client.post(
                "/v1/inference", json=body, headers={"Authorization": "Bearer synthetic-key-a"}
            )
            total_ms = (perf_counter() - started) * 1000
            assert response.status_code == 200
            await asyncio.to_thread(app.state.dispatcher.dispatch_once)
            events = sink.events_for_trace(response.json()["trace_id"])
            attempt = events[3].data
            assert isinstance(attempt, GenerationAttempt)
            assert len(events) == 5
            overheads.append(total_ms - attempt.total_latency_ms)
    p95 = sorted(overheads)[math.ceil(0.95 * len(overheads)) - 1]
    print(f"200 requests: p95 overhead = {p95:.3f} ms (limit 50 ms)")
    assert p95 < 50
    assert sink.dropped_events == 0


@pytest.mark.load
def test_200_live_requests_have_under_50ms_p95_overhead(
    router_seed: "RouterSeed", monkeypatch: pytest.MonkeyPatch
) -> None:
    r = router_seed
    with monkeypatch.context() as assignment:
        prepare_canary(r, assignment)
        for _ in range(4):
            assert infer(r)["route"]["specialist_served"]
        promote(r, "production")
    # Full assignment is allowed only after the ordinary canary/production gates pass.
    live = policy(r, canary={"traffic_fraction": 1}, shadow_enabled=False)
    overheads: list[float] = []
    routing_times: list[float] = []
    for _ in range(200):
        started = perf_counter()
        result = infer(r)
        elapsed_ms = (perf_counter() - started) * 1000
        assert result["route"]["specialist_served"] and not result["route"]["fallback_used"]
        interaction = r.seed.app.state.metadata.get(
            "synthetic-a", Interaction, result["interaction_id"]
        )
        attempt = r.seed.app.state.metadata.get(
            "synthetic-a", GenerationAttempt, interaction.final_attempt_id
        )
        route = r.seed.app.state.metadata.get(
            "synthetic-a", RouteDecision, interaction.route_decision_id
        )
        assert route.route_policy_id == live.policy_id and route.router_version == r.router
        assert attempt.deployment_id == r.specialist
        # Includes classification, routing, logging, persistence and HTTP overhead;
        # provider generation is the only time excluded, as in the foundation load gate.
        overheads.append(elapsed_ms - attempt.total_latency_ms)
        routing_times.append(route.decision_latency_ms)
    p95 = sorted(overheads)[math.ceil(0.95 * len(overheads)) - 1]
    routing_p95 = sorted(routing_times)[math.ceil(0.95 * len(routing_times)) - 1]
    print(f"200 live requests: p95 overhead = {p95:.3f} ms; routing = {routing_p95:.3f} ms")
    assert p95 < 50
    assert r.seed.client.get("/healthz").json()["live_planner_failures"] == 0
