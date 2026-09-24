from datetime import timedelta
from pathlib import Path

import pytest
from postgres_support import stored_bytes

from adaptive_llm.contracts import Event, Started, now, uid
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.events.outbox import Dispatcher, OutboxBackoff
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.outbox import SQLiteOutboxStore
from adaptive_llm.storage.sqlite import SQLiteDatabase


def event(interaction_id: str | None = None) -> Event:
    return Event(
        tenant_id="synthetic-a",
        trace_id=uid(),
        event_type="interaction.started.v1",
        data=Started(
            interaction_id=interaction_id or uid(),
            application_id="support-assistant",
            policy_version="synthetic-1",
        ),
    )


def test_outbox_transaction_rollback_idempotence_and_order(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path)
    try:
        store = SQLiteOutboxStore(database)
        first = event()
        second = event(first.data.interaction_id)
        other = event()
        with pytest.raises(StorageError, match="storage_transaction_required"):
            store.enqueue([first], 100)
        with pytest.raises(RuntimeError):
            with database.transaction():
                store.enqueue([first], 100)
                raise RuntimeError("synthetic_rollback")
        assert store.stats(now()).pending == 0
        with database.transaction():
            assert store.enqueue([first, second, other, first], 100) == 0
        assert store.stats(now()).pending == 3
        future = now() + timedelta(seconds=20)
        store.finish(first.event_id, "pending", 1, future, "sink_unavailable")
        assert store.next_due(now()).event_id == other.event_id
        sink = InMemoryEventSink()
        metrics = InProcessMetrics()
        dispatcher = Dispatcher(store, sink, metrics)
        assert dispatcher.dispatch_once() == 1
        assert sink.events == [other]
        assert dispatcher.dispatch_once() == 0
        dispatcher.clock = lambda: future
        assert dispatcher.dispatch_once() == 2
        assert sink.events == [other, first, second]
        assert metrics.get("outbox_delivered") == 3
        assert metrics.get("outbox_pending") == 0
    finally:
        database.close()


def test_bounded_retry_dead_letter_recovery_and_lag(tmp_path: Path) -> None:
    class FailingSink:
        def emit(self, event: Event) -> None:
            raise RuntimeError("SYNTHETIC-private-sink-body")

    database = SQLiteDatabase(tmp_path)
    try:
        store = SQLiteOutboxStore(database)
        first, second = event(), event()
        second = second.model_copy(update={"data": first.data})
        at = now()
        with database.transaction():
            store.enqueue([first, second], 100)
        metrics = InProcessMetrics()
        dispatcher = Dispatcher(
            store,
            FailingSink(),
            metrics,
            backoff=OutboxBackoff(base=2, cap=3, max_attempts=3),
            clock=lambda: at,
            jitter=lambda: 1.0,
        )
        assert dispatcher.dispatch_once() == 1
        row = database.connection.execute("SELECT * FROM outbox ORDER BY sequence").fetchone()
        assert row["attempts"] == 1 and row["last_error_code"] == "sink_unavailable"
        assert row["next_attempt_at"] == (at + timedelta(seconds=2)).isoformat()
        assert dispatcher.dispatch_once() == 0
        at += timedelta(seconds=2)
        assert dispatcher.dispatch_once() == 1
        at += timedelta(seconds=3)
        assert dispatcher.dispatch_once(limit=1) == 1
        assert store.dead_letters("synthetic-b") == []
        assert store.dead_letters("synthetic-a") == [
            (first.event_id, first.event_type, "sink_unavailable")
        ]
        assert metrics.get("outbox_retried") == 2
        assert metrics.get("dead_letter") == 1
        assert metrics.get("dispatcher_lag_seconds") >= 5
        assert store.redeliver(first.event_id, at)
        assert not store.redeliver(first.event_id, at)
        dispatcher.sink = InMemoryEventSink()
        assert dispatcher.dispatch_once() == 2
        assert dispatcher.sink.events == [first, second]
        assert store.stats(at).pending == store.stats(at).dead == 0
        assert "SYNTHETIC-private-sink-body" not in stored_bytes(database).decode(errors="ignore")
    finally:
        database.close()


@pytest.mark.parametrize("corruption", ["json", "schema", "binding"])
def test_invalid_event_is_quarantined_without_sink_call_or_retry(
    tmp_path: Path, corruption: str
) -> None:
    database = SQLiteDatabase(tmp_path)
    try:
        store, metrics, sink = SQLiteOutboxStore(database), InProcessMetrics(), InMemoryEventSink()
        first = event()
        with database.transaction():
            store.enqueue([first], 100)
            envelope = {
                "json": "{SYNTHETIC-invalid-json",
                "schema": '{"event_type":"unknown"}',
                "binding": first.model_copy(update={"tenant_id": "synthetic-b"}).model_dump_json(),
            }[corruption]
            database.connection.execute("UPDATE outbox SET envelope = ?", (envelope,))
        dispatcher = Dispatcher(store, sink, metrics)
        assert dispatcher.dispatch_once() == 1
        assert dispatcher.dispatch_once() == 0
        assert not sink.events
        assert store.dead_letters("synthetic-a") == [
            (first.event_id, first.event_type, "invalid_event")
        ]
        assert metrics.get("dead_letter") == 1
        assert metrics.get("outbox_retried") == 0
        assert store.redeliver(first.event_id, now())
        assert dispatcher.dispatch_once() == 1  # corrupt data is quarantined again
        assert metrics.get("dead_letter") == 2
    finally:
        database.close()


def test_accept_then_fail_is_idempotent_at_consumer(tmp_path: Path) -> None:
    class LostAcknowledgement(InMemoryEventSink):
        failed = False

        def emit(self, event: Event) -> None:
            super().emit(event)
            if not self.failed:
                self.failed = True
                raise RuntimeError("synthetic_lost_ack")

    database = SQLiteDatabase(tmp_path)
    try:
        store, sink = SQLiteOutboxStore(database), LostAcknowledgement()
        first = event()
        with database.transaction():
            store.enqueue([first], 100)
        dispatcher = Dispatcher(store, sink, InProcessMetrics())
        assert dispatcher.dispatch_once() == 1
        dispatcher.clock = lambda: now() + timedelta(seconds=60)
        assert dispatcher.dispatch_once() == 1
        assert sink.events == [first]
        # A duplicate, reordered arrival is also harmless to the consumer.
        sink.emit(first)
        assert sink.events == [first]
    finally:
        database.close()


def test_backoff_validation_and_jitter_bounds() -> None:
    for kwargs in ({"base": 0}, {"cap": 0}, {"base": 3, "cap": 2}, {"max_attempts": 0}):
        with pytest.raises(ValueError, match="invalid_outbox_backoff"):
            OutboxBackoff(**kwargs)
    backoff = OutboxBackoff(base=1, cap=8)
    assert [backoff.delay(i, lambda: 1) for i in range(1, 6)] == [1, 2, 4, 8, 8]
    assert backoff.delay(1000, lambda: 0) == 4
