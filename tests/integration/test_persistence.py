import asyncio
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    Feedback,
    FeedbackValue,
    GenerationAttempt,
    InferenceRequest,
    Interaction,
    RetrievalRun,
    RouteDecision,
    Started,
    now,
)
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.storage import EncryptedPayload, StoredRecord
from adaptive_llm.storage.retention import sweep
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore

HEADERS = {"Authorization": "Bearer synthetic-key-a"}


@pytest.mark.parametrize("cancel_during_write", [False, True])
async def test_slow_sqlite_write_does_not_block_loop_or_release_reservation_early(
    tmp_path: Path,
    inference_request: InferenceRequest,
    cancel_during_write: bool,
) -> None:
    entered, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()
    database = SQLiteDatabase(tmp_path)

    class SlowMetadata(SQLiteMetadataStore):
        def put(self, tenant_id: str, record: StoredRecord, expires_at: datetime) -> None:
            if isinstance(record, Interaction):
                assert threading.get_ident() != event_loop_thread
                entered.set()
                assert release.wait(timeout=2)
            super().put(tenant_id, record, expires_at)

    app = create_app(
        Settings(metadata_store=SlowMetadata(database), payload_store=SQLitePayloadStore(database))
    )
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            pending = asyncio.create_task(
                client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                assert not pending.done()
                if cancel_during_write:
                    pending.cancel()
                health = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
                assert health.status_code == 200
                assert not pending.done()
                duplicate = await asyncio.wait_for(
                    client.post(
                        "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
                    ),
                    timeout=0.5,
                )
                assert duplicate.status_code == 409
                assert len(app.state.inference._replays) == 1
            finally:
                release.set()
                if cancel_during_write:
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    assert (await pending).status_code == 200
            assert not app.state.inference._replays
            assert app.state.persistence.failures == 0
    finally:
        database.close()


def test_graph_and_encrypted_replay_survive_restart(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    sink = InMemoryEventSink()
    settings = Settings(data_dir=tmp_path, events=sink)
    app = create_app(settings)
    with TestClient(app) as client:
        first = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
        assert first.status_code == 200
        iid = first.json()["interaction_id"]
        metadata = app.state.metadata
        interaction = metadata.get("synthetic-a", Interaction, iid)
        assert interaction == sink.events[-1].data
        assert metadata.get("synthetic-a", Started, iid) == sink.events[0].data
        retrieval = metadata.get("synthetic-a", RetrievalRun, interaction.retrieval_run_id)
        route = metadata.get("synthetic-a", RouteDecision, interaction.route_decision_id)
        attempt = metadata.get("synthetic-a", GenerationAttempt, interaction.final_attempt_id)
        assert retrieval == sink.events[1].data
        assert route == sink.events[2].data
        assert attempt == sink.events[3].data
        assert interaction.input.messages_ref is retrieval.query_ref is attempt.output_ref is None
        assert app.state.persistence.failures == 0
        assert (
            metadata.get_replay(
                "synthetic-a", inference_request.application_id, inference_request.request_id, now()
            )
            is not None
        )
        feedback = Feedback(
            interaction_id=iid,
            source="user",
            label_type="thumb",
            value=FeedbackValue(score=1, max_score=1),
        )
        with metadata.transaction():
            metadata.put("synthetic-a", feedback, now() + timedelta(hours=1))
        assert metadata.get("synthetic-a", Feedback, feedback.feedback_id) == feedback
    restarted = create_app(replace(settings, events=InMemoryEventSink(), migrate_on_startup=False))
    with TestClient(restarted) as client:
        replay = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        client.app.state.dispatcher.dispatch_once()
        assert replay.json() == {**first.json(), "replayed": True}
        changed = {**inference_request.model_dump(), "max_output_tokens": 30}
        assert client.post("/v1/inference", json=changed, headers=HEADERS).status_code == 409
        assert not restarted.state.events.events


def test_graph_and_blobs_rollback_together_on_write_failure(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    database = SQLiteDatabase(tmp_path)
    metadata = SQLiteMetadataStore(database)

    class BrokenPayloadStore(SQLitePayloadStore):
        def put(self, payload: EncryptedPayload) -> None:
            super().put(payload)
            raise RuntimeError("synthetic-write-failure")

    app = create_app(Settings(metadata_store=metadata, payload_store=BrokenPayloadStore(database)))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
        assert response.status_code == 200
        assert metadata.get("synthetic-a", Interaction, response.json()["interaction_id"]) is None
        for table in (
            "interactions",
            "started",
            "retrieval_runs",
            "route_decisions",
            "attempts",
            "payloads",
            "replay_entries",
        ):
            assert database.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        assert app.state.persistence.failures == 1
        assert not app.state.inference._replays
    database.close()


def test_retention_and_replay_ttl_with_frozen_clock(inference_request: InferenceRequest) -> None:
    at = datetime(2026, 9, 23, tzinfo=UTC)
    app = create_app(Settings(retention_seconds=20, replay_ttl_seconds=5))
    with TestClient(app) as client:
        app.state.persistence.clock = lambda: at
        response = client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
        iid = response.json()["interaction_id"]
        meta, payloads = app.state.metadata, app.state.payloads
        replay = meta.get_replay(
            "synthetic-a", "support-assistant", inference_request.request_id, at
        )
        assert replay.expires_at == at + timedelta(seconds=5)
        assert payloads.get("synthetic-a", replay.response_ref, at) is not None
        assert sweep(meta, payloads, "synthetic-a", clock=lambda: at + timedelta(seconds=4)) == 0
        assert (
            meta.get_replay(
                "synthetic-a",
                "support-assistant",
                inference_request.request_id,
                at + timedelta(seconds=5),
            )
            is None
        )
        assert sweep(meta, payloads, "synthetic-a", clock=lambda: at + timedelta(seconds=5)) == 0
        assert payloads.get("synthetic-a", replay.response_ref, at) is None
        assert meta.state("synthetic-a", iid) == "active"
        assert sweep(meta, payloads, "synthetic-b", clock=lambda: at + timedelta(days=1)) == 0
        assert sweep(meta, payloads, "synthetic-a", clock=lambda: at + timedelta(seconds=20)) == 1
        assert meta.state("synthetic-a", iid) == "expired"
        assert sweep(meta, payloads, "synthetic-a", clock=lambda: at + timedelta(seconds=20)) == 0
        assert (
            app.state.database.connection.execute(
                "SELECT state FROM attempts WHERE tenant_id = ?", ("synthetic-a",)
            ).fetchone()[0]
            == "expired"
        )
        app.state.persistence.clock = lambda: at + timedelta(seconds=21)
        fresh = client.post("/v1/inference", json=inference_request.model_dump(), headers=HEADERS)
        assert fresh.status_code == 200
        assert fresh.json()["interaction_id"] != iid
        assert fresh.json()["replayed"] is False


@pytest.mark.parametrize("failure", ["error", "deadline_exceeded"])
def test_failed_attempts_are_persisted_without_replay(
    inference_request: InferenceRequest, failure: str
) -> None:
    from adaptive_llm.providers import FakeProvider

    app = create_app(Settings(provider=FakeProvider(test_only_failure=failure)))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
        )
        client.app.state.dispatcher.dispatch_once()
        assert response.status_code in (502, 504)
        completed = app.state.events.events[-1].data
        persisted = app.state.metadata.get("synthetic-a", Interaction, completed.interaction_id)
        assert persisted.status == "failed"
        assert persisted.error_code == completed.error_code
        assert (
            app.state.metadata.get_replay(
                "synthetic-a", inference_request.application_id, inference_request.request_id, now()
            )
            is None
        )
        assert app.state.persistence.failures == 0


def test_policy_denial_has_no_durable_rows(inference_request: InferenceRequest) -> None:
    from adaptive_llm.contracts import PolicyDecision
    from adaptive_llm.gateway.identity import Identity

    class Denied:
        def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
            return PolicyDecision(
                policy_version="synthetic", processing_allowed=False, retention_seconds=60
            )

    app = create_app(Settings(policy=Denied()))
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/inference", json=inference_request.model_dump(), headers=HEADERS
            ).status_code
            == 403
        )
        for table in ("interactions", "payloads", "replay_entries"):
            assert (
                app.state.database.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                == 0
            )
