import asyncio
from collections import defaultdict
from time import perf_counter

import httpx
import pytest

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Event, InferenceRequest, now
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.events.outbox import OutboxBackoff


@pytest.mark.load
async def test_1000_interactions_have_999_per_mille_event_correlation(
    inference_request: InferenceRequest,
) -> None:
    class FlakySink(InMemoryEventSink):
        calls = 0

        def emit(self, event: Event) -> None:
            self.calls += 1
            if self.calls % 17 == 0:
                raise RuntimeError("synthetic_transient_outage")
            super().emit(event)
            if self.calls % 31 == 0:
                raise RuntimeError("synthetic_lost_acknowledgement")

    sink = FlakySink(capacity=5000)
    app = create_app(
        Settings(events=sink, outbox_backoff=OutboxBackoff(base=0.001, cap=0.01, max_attempts=50))
    )
    results = []
    started = perf_counter()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        for index in range(1000):
            response = await client.post(
                "/v1/inference",
                headers={"Authorization": "Bearer synthetic-key-a"},
                json={
                    **inference_request.model_dump(),
                    "request_id": f"synthetic-correlation-{index}",
                },
            )
            assert response.status_code == 200
            results.append(response.json())
        async with asyncio.timeout(30):
            while app.state.outbox.stats(now()).pending:  # noqa: ASYNC110 - observe background drain
                await asyncio.sleep(0.01)
        by_trace = defaultdict(list)
        for event in sink.events:
            by_trace[event.trace_id].append(event)
        valid = 0
        expected = [
            "interaction.started.v1",
            "retrieval.completed.v1",
            "route.decided.v1",
            "generation.completed.v1",
            "interaction.completed.v1",
        ]
        for result in results:
            events = by_trace[result["trace_id"]]
            if [e.event_type for e in events] != expected:
                continue
            if {e.data.interaction_id for e in events} != {result["interaction_id"]}:
                continue
            if {e.tenant_id for e in events} != {"synthetic-a"}:
                continue
            _, retrieval, route, attempt, interaction = [e.data for e in events]
            if (
                interaction.trace_id == result["trace_id"]
                and interaction.retrieval_run_id == retrieval.retrieval_run_id
                and interaction.route_decision_id == route.route_decision_id
                and interaction.final_attempt_id == attempt.attempt_id
                and interaction.generation_attempt_ids == [attempt.attempt_id]
            ):
                valid += 1
        assert app.state.metrics.get("outbox_retried") > 0
        assert app.state.metrics.get("dropped_events") == 0
        assert app.state.outbox.stats(now()).dead == 0
        assert len(sink.events) == len({e.event_id for e in sink.events}) == 5000
    ratio = valid / 1000
    print(
        f"event correlation: valid={valid}/1000 ratio={ratio:.3%} events={len(sink.events)}"
        f" elapsed_s={perf_counter() - started:.3f}"
    )
    assert ratio >= 0.999
