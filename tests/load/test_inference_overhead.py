import math
from time import perf_counter

import httpx
import pytest

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import GenerationAttempt, InferenceRequest
from adaptive_llm.events import InMemoryEventSink


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
            events = sink.events_for_trace(response.json()["trace_id"])
            attempt = events[3].data
            assert isinstance(attempt, GenerationAttempt)
            assert len(events) == 5
            overheads.append(total_ms - attempt.total_latency_ms)
    p95 = sorted(overheads)[math.ceil(0.95 * len(overheads)) - 1]
    print(f"200 requests: p95 overhead = {p95:.3f} ms (limit 50 ms)")
    assert p95 < 50
    assert sink.dropped_events == 0
