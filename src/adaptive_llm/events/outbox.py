"""Transactional outbox boundary and asynchronous, at-least-once delivery."""

import asyncio
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Literal, Protocol

from opentelemetry import trace
from opentelemetry.trace import Tracer

from adaptive_llm.contracts import DeletionRequest, Event, now
from adaptive_llm.events import EventSink
from adaptive_llm.metrics import Metrics


def interaction_key(event: Event) -> str | None:
    if isinstance(event.data, DeletionRequest):
        return event.data.target_id if event.data.scope == "interaction" else None
    return getattr(event.data, "interaction_id", None)


@dataclass(frozen=True)
class OutboxRow:
    event_id: str
    tenant_id: str
    trace_id: str
    interaction_id: str | None
    event_type: str
    envelope: str
    attempts: int


@dataclass(frozen=True)
class OutboxStats:
    pending: int
    dead: int
    lag_seconds: float


class OutboxStore(Protocol):
    # Must share the metadata transaction. Returns the count dropped in that transaction.
    def enqueue(self, events: Sequence[Event], pending_limit: int) -> int: ...

    def next_due(self, at: datetime) -> OutboxRow | None: ...

    def finish(
        self,
        event_id: str,
        state: Literal["pending", "delivered", "dead"],
        attempts: int,
        next_attempt_at: datetime,
        error: Literal["sink_unavailable", "invalid_event"] | None,
    ) -> None: ...

    def stats(self, at: datetime) -> OutboxStats: ...

    def dead_letters(self, tenant_id: str) -> list[tuple[str, str, str]]: ...

    def redeliver(self, event_id: str, at: datetime) -> bool: ...


def publish_metrics(store: OutboxStore, metrics: Metrics, at: datetime) -> OutboxStats:
    stats = store.stats(at)
    metrics.gauge("outbox_pending", stats.pending)
    metrics.gauge("outbox_dead", stats.dead)
    metrics.gauge("dispatcher_lag_seconds", stats.lag_seconds)
    return stats


@dataclass(frozen=True)
class OutboxBackoff:
    base: float = 0.1
    cap: float = 30.0
    max_attempts: int = 8

    def __post_init__(self) -> None:
        if not 0 < self.base <= self.cap or not math.isfinite(self.cap) or self.max_attempts < 1:
            raise ValueError("invalid_outbox_backoff")

    def delay(self, attempts: int, jitter: Callable[[], float]) -> float:
        # Equal jitter keeps retries positive and inside the configured cap.
        ceiling = min(self.cap, self.base * 2.0 ** min(attempts - 1, 30))
        return ceiling * (0.5 + 0.5 * jitter())


class Dispatcher:
    """One dispatcher per local database; sinks must deduplicate event_id after a crash.

    Sink calls run outside the database lock and off the serving event loop. Concurrent
    dispatch_once calls on this dispatcher serialize; multiple serving processes are deferred.
    """

    def __init__(
        self,
        store: OutboxStore,
        sink: EventSink,
        metrics: Metrics,
        *,
        backoff: OutboxBackoff | None = None,
        clock: Callable[[], datetime] = now,
        jitter: Callable[[], float] = random.random,
        tracer: Tracer | None = None,
    ) -> None:
        self.store, self.sink, self.metrics = store, sink, metrics
        self.backoff, self.clock, self.jitter = backoff or OutboxBackoff(), clock, jitter
        self.tracer = tracer if tracer is not None else trace.get_tracer("adaptive_llm.outbox")
        self._lock = Lock()
        self._stop = asyncio.Event()

    def refresh_metrics(self) -> OutboxStats:
        return publish_metrics(self.store, self.metrics, self.clock())

    def dispatch_once(self, limit: int = 256) -> int:
        processed = 0
        with self._lock:
            for _ in range(limit):
                row = self.store.next_due(self.clock())
                if row is None:
                    break
                attempts = row.attempts + 1
                try:
                    event = Event.model_validate_json(row.envelope)
                    if (
                        event.event_id != row.event_id
                        or event.tenant_id != row.tenant_id
                        or event.trace_id != row.trace_id
                        or event.event_type != row.event_type
                        or interaction_key(event) != row.interaction_id
                    ):
                        raise ValueError("invalid_event")
                except Exception:
                    self.store.finish(row.event_id, "dead", attempts, self.clock(), "invalid_event")
                    self.metrics.increment("dead_letter")
                else:
                    try:
                        with self.tracer.start_as_current_span(
                            "event_emission",
                            attributes={"trace_id": event.trace_id, "schema_version": "1.0"},
                            record_exception=False,
                            set_status_on_exception=False,
                        ):
                            self.sink.emit(event)
                    except Exception:
                        if attempts >= self.backoff.max_attempts:
                            self.store.finish(
                                row.event_id, "dead", attempts, self.clock(), "sink_unavailable"
                            )
                            self.metrics.increment("dead_letter")
                        else:
                            retry_at = self.clock() + timedelta(
                                seconds=self.backoff.delay(attempts, self.jitter)
                            )
                            self.store.finish(
                                row.event_id, "pending", attempts, retry_at, "sink_unavailable"
                            )
                            self.metrics.increment("outbox_retried")
                    else:
                        self.store.finish(row.event_id, "delivered", attempts, self.clock(), None)
                        self.metrics.increment("outbox_delivered")
                processed += 1
            self.refresh_metrics()
        return processed

    async def run(self, interval: float = 0.05) -> None:
        while not self._stop.is_set():
            try:
                count = await asyncio.to_thread(self.dispatch_once)
            except Exception:
                # Storage errors also stay out of serving, without recording their bodies.
                self.metrics.increment("dispatcher_failures")
                count = 0
            if count == 256:
                continue
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()
