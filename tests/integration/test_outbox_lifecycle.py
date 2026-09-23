import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Event, InferenceRequest, now
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.events.outbox import OutboxBackoff
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.outbox import SQLiteOutboxStore
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore

HEADERS = {"Authorization": "Bearer synthetic-key-a", "X-Subject": "synthetic-subject"}


def test_committed_backlog_survives_restart_and_health_reports_degradation(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    app = create_app(Settings(data_dir=tmp_path, outbox_dispatch_enabled=False))
    with TestClient(app) as client:
        result = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        assert result.status_code == 200
        assert app.state.events.events == []
        assert app.state.outbox.stats(now()).pending == 5
        row = app.state.database.connection.execute(
            "SELECT envelope, next_attempt_at FROM outbox "
            "WHERE event_type = 'interaction.started.v1'"
        ).fetchone()
        started = Event.model_validate_json(row["envelope"])
        assert row["next_attempt_at"] == started.occurred_at.isoformat()
    sink = InMemoryEventSink()
    restarted = create_app(Settings(data_dir=tmp_path, events=sink, outbox_dispatch_enabled=False))
    with TestClient(restarted) as client:
        assert client.get("/healthz").json()["outbox_pending"] == 5
        assert restarted.state.ready
        assert restarted.state.dispatcher.dispatch_once() == 5
        assert len(sink.events_for_trace(result.json()["trace_id"])) == 5
        replay = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        assert replay.json() == {**result.json(), "replayed": True}
        assert restarted.state.outbox.stats(now()).pending == 0
        assert restarted.state.metrics.get("replay_hits") == 1
        assert restarted.state.metrics.get("requests", status_class="2xx", method="POST") == 1
        assert client.post("/v1/inference", json={}).status_code == 401
        assert restarted.state.metrics.get("requests", status_class="4xx", method="POST") == 1


async def test_blocked_delivery_does_not_block_serving_or_health(
    inference_request: InferenceRequest,
) -> None:
    entered, release = threading.Event(), threading.Event()

    class BlockingSink(InMemoryEventSink):
        def emit(self, event: Event) -> None:
            entered.set()
            assert release.wait(5)
            super().emit(event)

    app = create_app(Settings(events=BlockingSink()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        try:
            first = await asyncio.wait_for(
                client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump()),
                1,
            )
            assert first.status_code == 200
            assert await asyncio.to_thread(entered.wait, 2)
            second = await asyncio.wait_for(
                client.post(
                    "/v1/inference",
                    headers=HEADERS,
                    json={**inference_request.model_dump(), "request_id": "synthetic-second"},
                ),
                1,
            )
            assert second.status_code == 200
            assert (await asyncio.wait_for(client.get("/healthz"), 0.5)).status_code == 200
            assert app.state.events.events == []
            assert (await client.get("/healthz")).json()["outbox_pending"] == 10
        finally:
            release.set()
        await asyncio.to_thread(app.state.dispatcher.dispatch_once)
        assert len(app.state.events.events) == 10


def test_backlog_limit_preserves_billing_completion_and_privacy(
    inference_request: InferenceRequest,
) -> None:
    app = create_app(Settings(outbox_pending_limit=1, outbox_dispatch_enabled=False))
    with TestClient(app) as client:
        first = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        second = client.post(
            "/v1/inference",
            headers=HEADERS,
            json={**inference_request.model_dump(), "request_id": "synthetic-next"},
        )
        assert first.status_code == second.status_code == 200
        assert app.state.metrics.get("dropped_events") == 5
        assert app.state.metrics.get("fallback_free_attempts") == 2
        assert (
            client.delete(
                f"/v1/privacy/interactions/{first.json()['interaction_id']}", headers=HEADERS
            ).status_code
            == 204
        )
        assert client.post(
            "/v1/privacy/subjects/deletion-requests",
            headers=HEADERS,
            json={"subject": "synthetic-subject"},
        ).json() == {"deleted": 1}
        assert app.state.dispatcher.dispatch_once() == 8
        kinds = [e.event_type for e in app.state.events.events]
        assert kinds.count("generation.completed.v1") == 2
        assert kinds.count("interaction.completed.v1") == 2
        assert kinds.count("privacy.deletion.requested.v1") == 3
        assert "retrieval.completed.v1" not in kinds
        assert app.state.metrics.get("outbox_pending") == 0


def test_outbox_failure_rolls_back_graph_and_privacy_atomically(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    database = SQLiteDatabase(tmp_path)

    class BrokenOutbox(SQLiteOutboxStore):
        broken = True
        attempted: list[Event] = []

        def enqueue(self, events: Sequence[Event], pending_limit: int) -> int:
            self.attempted = list(events)
            result = super().enqueue(events, pending_limit)
            if self.broken:
                raise StorageError("synthetic_outbox_failed")
            return result

    store = BrokenOutbox(database)
    app = create_app(
        Settings(
            metadata_store=SQLiteMetadataStore(database),
            payload_store=SQLitePayloadStore(database),
            outbox_store=store,
            outbox_dispatch_enabled=False,
        )
    )
    try:
        with TestClient(app) as client:
            result = client.post(
                "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
            )
            assert result.status_code == 200
            for table in ("outbox", "interactions", "payloads", "replay_entries"):
                assert (
                    database.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
                )
            assert app.state.metrics.get("persistence_failures") == 1
            assert app.state.metrics.get("degraded_emissions") == 5
            assert app.state.events.events == store.attempted
            store.broken = False
            result = client.post(
                "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
            )
            iid = result.json()["interaction_id"]
            store.broken = True
            with pytest.raises(StorageError, match="synthetic_outbox_failed"):
                app.state.persistence.delete("synthetic-a", iid, None)
            assert app.state.events.events[-1] == store.attempted[0]
            assert app.state.events.events[-1].trace_id == result.json()["trace_id"]
            assert app.state.metrics.get("persistence_failures") == 2
            assert app.state.metrics.get("degraded_emissions") == 6
            assert app.state.metadata.state("synthetic-a", iid) == "active"
            assert app.state.metadata.get_tombstone("synthetic-a", iid) is None
            assert store.stats(now()).pending == 5
            assert database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0] == 1
    finally:
        database.close()


def test_dead_health_and_failed_request_metrics(inference_request: InferenceRequest) -> None:
    from adaptive_llm.providers import FakeProvider

    class Down:
        def emit(self, event: Event) -> None:
            raise RuntimeError("synthetic_down")

    app = create_app(
        Settings(
            events=Down(),
            outbox_dispatch_enabled=False,
            outbox_backoff=OutboxBackoff(max_attempts=1),
            provider=FakeProvider(test_only_failure="error"),
        )
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
            ).status_code
            == 502
        )
        assert app.state.dispatcher.dispatch_once() == 5
        health = client.get("/healthz").json()
        assert health["status"] == "ok" and health["dead_letters"] == 5
        assert app.state.ready
        assert app.state.metrics.get("requests", status_class="5xx", method="POST") == 1


@pytest.mark.parametrize("sink_fails", [False, True])
def test_broken_metadata_still_emits_content_free_correlated_events(
    tmp_path: Path,
    inference_request: InferenceRequest,
    sink_fails: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from datetime import datetime

    from adaptive_llm.storage import StoredRecord

    database = SQLiteDatabase(tmp_path)

    class BrokenMetadata(SQLiteMetadataStore):
        def put(self, tenant_id: str, record: StoredRecord, expires_at: datetime) -> None:
            super().put(tenant_id, record, expires_at)
            raise StorageError("synthetic_metadata_failed")

    class Sink(InMemoryEventSink):
        def emit(self, event: Event) -> None:
            assert not database.connection.in_transaction
            super().emit(event)
            if sink_fails:
                raise RuntimeError("SYNTHETIC private sink error")

    sink = Sink()
    app = create_app(
        Settings(
            metadata_store=BrokenMetadata(database),
            payload_store=SQLitePayloadStore(database),
            events=sink,
            outbox_dispatch_enabled=False,
        )
    )
    try:
        with TestClient(app) as client:
            result = client.post(
                "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
            )
            assert result.status_code == 200
            events = sink.events
            assert [e.event_type for e in events] == [
                "interaction.started.v1",
                "retrieval.completed.v1",
                "route.decided.v1",
                "generation.completed.v1",
                "interaction.completed.v1",
            ]
            assert {e.trace_id for e in events} == {result.json()["trace_id"]}
            assert {e.data.interaction_id for e in events} == {result.json()["interaction_id"]}
            assert {e.tenant_id for e in events} == {"synthetic-a"}
            assert events[0].occurred_at == events[-1].data.started_at
            assert app.state.persistence.failures == 1
            assert app.state.metrics.get("persistence_failures") == 1
            assert app.state.metrics.get("degraded_emissions") == 5
            assert app.state.metrics.get("outbox_delivered") == 0
            for table in ("outbox", "interactions", "payloads", "replay_entries"):
                assert (
                    database.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
                )
            telemetry = " ".join(e.model_dump_json() for e in events) + caplog.text
            assert "SYNTHETIC private sink error" not in telemetry
            assert inference_request.messages[-1].content not in telemetry
            assert result.json()["content"] not in telemetry
            assert HEADERS["X-Subject"] not in telemetry
    finally:
        database.close()


@pytest.mark.parametrize("scope", ["interaction", "subject"])
@pytest.mark.parametrize("failure_stage", ["begin", "write"])
def test_privacy_failure_emits_requests_and_preserves_original_failure(
    tmp_path: Path,
    inference_request: InferenceRequest,
    scope: str,
    failure_stage: str,
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    from adaptive_llm.contracts import DeletionRequest

    database = SQLiteDatabase(tmp_path)

    class BrokenMetadata(SQLiteMetadataStore):
        broken = False

        @contextmanager
        def transaction(self) -> Iterator[None]:
            if self.broken and failure_stage == "begin":
                raise StorageError("synthetic_privacy_failed")
            with super().transaction():
                yield

        def tombstone(self, tenant_id: str, interaction_id: str, deletion: DeletionRequest) -> None:
            super().tombstone(tenant_id, interaction_id, deletion)
            if self.broken:
                raise StorageError("synthetic_privacy_failed")

    class FailingSink(InMemoryEventSink):
        def emit(self, event: Event) -> None:
            assert not database.connection.in_transaction
            super().emit(event)
            raise RuntimeError("synthetic_sink_failed")

    metadata, sink = BrokenMetadata(database), FailingSink()
    app = create_app(
        Settings(
            metadata_store=metadata,
            payload_store=SQLitePayloadStore(database),
            events=sink,
            outbox_dispatch_enabled=False,
        )
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
            )
            assert response.status_code == 200
            iid = response.json()["interaction_id"]
            pseudonym = app.state.keyring.pseudonym("synthetic-a", HEADERS["X-Subject"])
            metadata.broken = True
            with pytest.raises(StorageError, match="^synthetic_privacy_failed$"):
                if scope == "interaction":
                    app.state.persistence.delete("synthetic-a", iid, None)
                else:
                    app.state.persistence.delete_subject("synthetic-a", pseudonym, None)
            expected = 2 if scope == "subject" and failure_stage == "write" else 1
            assert len(sink.events) == expected
            assert {e.event_type for e in sink.events} == {"privacy.deletion.requested.v1"}
            assert sink.events[-1].data.scope == scope
            assert sink.events[-1].data.target_id == (iid if scope == "interaction" else pseudonym)
            assert {e.tenant_id for e in sink.events} == {"synthetic-a"}
            if failure_stage == "write":
                assert sink.events[0].trace_id == response.json()["trace_id"]
            assert app.state.metrics.get("persistence_failures") == 1
            assert app.state.metrics.get("degraded_emissions") == expected
            assert metadata.get_tombstone("synthetic-a", iid) is None
            assert metadata.get_subject_tombstone("synthetic-a", pseudonym) is None
            assert metadata.state("synthetic-a", iid) == "active"
            assert database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0] == 1
            assert app.state.outbox.stats(now()).pending == 5
            if scope == "interaction":
                failed = client.delete(f"/v1/privacy/interactions/{iid}", headers=HEADERS)
                method = "DELETE"
            else:
                failed = client.post(
                    "/v1/privacy/subjects/deletion-requests",
                    headers=HEADERS,
                    json={"subject": HEADERS["X-Subject"]},
                )
                method = "POST"
            assert failed.status_code == 500
            assert app.state.metrics.get("requests", method=method, status_class="5xx") == 1
    finally:
        database.close()


def test_privacy_request_metrics_use_only_bounded_method_and_status_labels(
    inference_request: InferenceRequest,
) -> None:
    from adaptive_llm.metrics import InProcessMetrics

    metrics = InProcessMetrics()
    app = create_app(Settings(metrics=metrics, outbox_dispatch_enabled=False))
    with TestClient(app) as client:
        result = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        path = f"/v1/privacy/interactions/{result.json()['interaction_id']}"
        assert client.delete(path, headers=HEADERS).status_code == 204
        assert client.delete(path).status_code == 401
        assert (
            client.delete("/v1/privacy/interactions/synthetic-missing", headers=HEADERS).status_code
            == 404
        )
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/not-a-route").status_code == 404
        assert client.request("SYNTHETIC_UNBOUNDED_METHOD", path).status_code == 405
        assert metrics.get("requests", method="POST", status_class="2xx") == 1
        assert metrics.get("requests", method="DELETE", status_class="2xx") == 1
        assert metrics.get("requests", method="DELETE", status_class="4xx") == 2
        assert metrics.get("requests", method="GET", status_class="4xx") == 1
        assert metrics.get("requests", method="OTHER", status_class="4xx") == 1
        assert metrics.get("requests", method="GET", status_class="2xx") == 0
        assert set(metrics._values) >= {
            ("requests", None, "2xx", "POST"),
            ("requests", None, "2xx", "DELETE"),
            ("requests", None, "4xx", "DELETE"),
            ("requests", None, "4xx", "GET"),
            ("requests", None, "4xx", "OTHER"),
        }
        assert path not in repr(metrics._values)
        assert result.json()["interaction_id"] not in repr(metrics._values)
