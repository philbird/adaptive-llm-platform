import asyncio
import math
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Event, GenerationAttempt, InferenceRequest, now
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.events.outbox import OutboxBackoff
from adaptive_llm.storage.__main__ import main
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.operations import rotate_key
from adaptive_llm.storage.retention import sweep

HEADERS = {"Authorization": "Bearer synthetic-key-a", "X-Subject": "synthetic-drill-subject"}


@pytest.mark.drill
async def test_telemetry_outage(inference_request: InferenceRequest) -> None:
    class FlakySink(InMemoryEventSink):
        failures = 25

        def emit(self, event: Event) -> None:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("synthetic_outage")
            super().emit(event)

    sink = FlakySink(capacity=1000)
    app = create_app(
        Settings(events=sink, outbox_backoff=OutboxBackoff(base=0.001, cap=0.01, max_attempts=50))
    )
    timings = []
    started = perf_counter()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        for index in range(200):
            before = perf_counter()
            response = await client.post(
                "/v1/inference",
                headers=HEADERS,
                json={**inference_request.model_dump(), "request_id": f"synthetic-outage-{index}"},
            )
            total_ms = (perf_counter() - before) * 1000
            assert response.status_code == 200
            timings.append((response.json()["trace_id"], total_ms))
        async with asyncio.timeout(10):
            while app.state.outbox.stats(now()).pending:  # noqa: ASYNC110 - observe background drain
                await asyncio.sleep(0.01)
        overheads = []
        for trace_id, total_ms in timings:
            events = sink.events_for_trace(trace_id)
            assert len(events) == 5
            attempt = events[3].data
            assert isinstance(attempt, GenerationAttempt)
            overheads.append(total_ms - attempt.total_latency_ms)
        assert len(sink.events) == len({e.event_id for e in sink.events}) == 1000
        assert app.state.outbox.stats(now()).dead == 0
        assert app.state.metrics.get("outbox_retried") == 25
    p95 = sorted(overheads)[math.ceil(0.95 * len(overheads)) - 1]
    print(
        f"telemetry outage: requests=200 retries=25 delivered=1000 p95_overhead_ms={p95:.3f}"
        f" elapsed_s={perf_counter() - started:.3f}"
    )
    assert p95 < 50


@pytest.mark.drill
def test_dead_letter_recovery(
    tmp_path: Path, inference_request: InferenceRequest, capsys: pytest.CaptureFixture[str]
) -> None:
    class Down:
        def emit(self, event: Event) -> None:
            raise RuntimeError("synthetic_down")

    app = create_app(
        Settings(
            data_dir=tmp_path,
            events=Down(),
            outbox_dispatch_enabled=False,
            outbox_backoff=OutboxBackoff(max_attempts=1),
        )
    )
    started = perf_counter()
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
            ).status_code
            == 200
        )
        assert app.state.dispatcher.dispatch_once() == 5
        dead = app.state.outbox.dead_letters("synthetic-a")
        assert len(dead) == 5
        main(["dead-letters", "--data-dir", str(tmp_path), "--tenant", "synthetic-a"])
        listing = capsys.readouterr().out
        assert all(len(line.split()) == 3 for line in listing.splitlines())
        assert all(event_id in listing for event_id, _, _ in dead)
        for event_id, _, _ in dead:
            main(["redeliver", "--data-dir", str(tmp_path), "--event-id", event_id])
        assert capsys.readouterr().out == "redelivered=1\n" * 5
        sink = InMemoryEventSink()
        main(["dispatch-once", "--data-dir", str(tmp_path)], sink=sink)
        assert capsys.readouterr().out == "processed=5 pending=0 dead=0\n"
        assert len(sink.events) == 5
    print(f"dead-letter recovery: dead=5 recovered=5 elapsed_s={perf_counter() - started:.3f}")


@pytest.mark.drill
async def test_retention_deletion_under_load(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    app = create_app(Settings(data_dir=tmp_path, retention_seconds=60))
    marker = "SYNTHETIC_PRIVATE_DRILL_PAYLOAD"
    body = {**inference_request.model_dump(), "messages": [{"role": "user", "content": marker}]}
    started = perf_counter()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        app.state.persistence.clock = lambda: now() - timedelta(seconds=120)
        for index in range(100):
            assert (
                await client.post(
                    "/v1/inference",
                    headers=HEADERS,
                    json={**body, "request_id": f"synthetic-expired-{index}"},
                )
            ).status_code == 200
        app.state.persistence.clock = now
        sweeping = asyncio.create_task(
            asyncio.to_thread(sweep, app.state.metadata, app.state.payloads, "synthetic-a")
        )
        for index in range(100):
            assert (
                await client.post(
                    "/v1/inference",
                    headers=HEADERS,
                    json={**body, "request_id": f"synthetic-live-{index}"},
                )
            ).status_code == 200
        assert await sweeping == 100
        deleted = await client.post(
            "/v1/privacy/subjects/deletion-requests",
            headers=HEADERS,
            json={"subject": "synthetic-drill-subject"},
        )
        assert deleted.json() == {"deleted": 200}
        with app.state.database.lock:
            assert (
                app.state.database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0]
                == 0
            )
            assert (
                app.state.database.connection.execute(
                    "SELECT count(*) FROM replay_entries"
                ).fetchone()[0]
                == 0
            )
        path = app.state.database.path
    assert marker.encode() not in path.read_bytes()
    assert b"synthetic-drill-subject" not in path.read_bytes()
    assert b"SYNTHETIC ANSWER:" not in path.read_bytes()
    print(
        f"retention/deletion: interactions=200 swept=100 deleted=200 blobs=0 plaintext_matches=0"
        f" elapsed_s={perf_counter() - started:.3f}"
    )


@pytest.mark.drill
def test_backup_restore_round_trip(
    tmp_path: Path, inference_request: InferenceRequest, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(data_dir=tmp_path / "data", outbox_dispatch_enabled=False)
    output = tmp_path / "backup.sqlite3"
    started = perf_counter()
    app = create_app(settings)
    with TestClient(app) as client:
        first = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        assert first.status_code == 200
        from adaptive_llm.contracts import TrainingJob, TrainingJobSpecification, uid

        control_job = TrainingJob(
            specification=TrainingJobSpecification(
                dataset_id="synthetic-drill", dataset_version=uid()
            )
        )
        app.state.registry.save_job(control_job, ["synthetic-a"])
        main(["backup", "--data-dir", str(settings.data_dir), "--out", str(output)])
        assert capsys.readouterr().out == "backup_complete\n"
        app.state.evaluation_database.connection.execute("DELETE FROM training_jobs")
        assert (
            client.delete(
                f"/v1/privacy/interactions/{first.json()['interaction_id']}", headers=HEADERS
            ).status_code
            == 204
        )
        assert (
            app.state.database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0]
            == 0
        )
    with pytest.raises(SystemExit):
        main(["restore", "--data-dir", str(settings.data_dir), "--out", str(output)])
    assert capsys.readouterr().err == "restore_destination_exists\n"
    main(["restore", "--data-dir", str(settings.data_dir), "--out", str(output), "--force"])
    assert capsys.readouterr().out == "database_restored\n"
    restarted = create_app(settings)
    with TestClient(restarted) as client:
        replay = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        assert replay.json() == {**first.json(), "replayed": True}
        assert [
            r[0]
            for r in restarted.state.database.connection.execute(
                "SELECT version FROM schema_migrations"
            )
        ] == [1, 2, 3, 4, 5, 7]
        control = restarted.state.evaluation_database.connection
        assert [r[0] for r in control.execute("SELECT version FROM schema_migrations")] == [
            3,
            6,
            7,
            8,
            9,
        ]
        assert (
            control.execute("SELECT data FROM training_jobs").fetchone()[0]
            == control_job.model_dump_json()
        )
        assert restarted.state.dispatcher.dispatch_once() == 5
    print(
        "backup/restore: replay=matched tenant_migrations=1-5,7 control_migrations=3,6,7,8,9"
        " restored_jobs=1 restored_events=5"
        f" elapsed_s={perf_counter() - started:.3f}"
    )


@pytest.mark.drill
async def test_key_rotation_with_live_traffic(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    old_settings = Settings(data_dir=tmp_path, payload_key=b"a" * 32)
    started = perf_counter()
    app = create_app(old_settings)
    async with app.router.lifespan_context(app):
        identity = app.state.authenticator.authenticate(
            HEADERS["Authorization"], HEADERS["X-Subject"]
        )
        for index in range(100):
            await app.state.inference.infer(
                inference_request.model_copy(update={"request_id": f"synthetic-old-{index}"}),
                identity,
            )
    keys = {"local-1": b"a" * 32, "local-2": b"b" * 32}
    app = create_app(replace(old_settings, payload_keys=keys, payload_key_version="local-2"))
    async with app.router.lifespan_context(app):
        original = inference_request.model_copy(update={"request_id": "synthetic-old-0"})
        before = await app.state.inference.infer(original, identity)
        assert before.replayed
        rotation = asyncio.create_task(
            asyncio.to_thread(
                rotate_key,
                app.state.database,
                "synthetic-a",
                PayloadCipher(keys, "local-2"),
                batch_size=7,
            )
        )
        for index in range(100):
            result = await app.state.inference.infer(
                inference_request.model_copy(update={"request_id": f"synthetic-new-{index}"}),
                identity,
            )
            assert not result.replayed
        assert await rotation == 100
        app.state.persistence.cipher = PayloadCipher({"local-2": b"b" * 32}, "local-2")
        after = await app.state.inference.infer(original, identity)
        assert after == before
        with app.state.database.lock:
            versions = app.state.database.connection.execute(
                "SELECT key_version, count(*) FROM payloads GROUP BY key_version"
            ).fetchall()
        assert [(r[0], r[1]) for r in versions] == [("local-2", 200)]
    print(
        f"key rotation: rotated=100 live_requests=100 current_key_blobs=200 replay=matched"
        f" elapsed_s={perf_counter() - started:.3f}"
    )
