"""Non-blocking, bounded, in-memory collection of content-free event contracts."""

from threading import Lock
from typing import Protocol

from adaptive_llm.contracts import Event


class EventSink(Protocol):
    # Return only after accepting the event. Raise on rejection; deduplicate by event_id.
    def emit(self, event: Event) -> None: ...


class InMemoryEventSink:
    """Bounded local consumer; reject overflow so the durable dispatcher can retry."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("invalid_event_capacity")
        self._capacity = capacity
        self._events: dict[str, Event] = {}
        self._dropped = 0
        self._lock = Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            if event.event_id in self._events:
                return
            if len(self._events) >= self._capacity:
                self._dropped += 1
                raise RuntimeError("event_sink_full")
            self._events[event.event_id] = event.model_copy(deep=True)

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped

    def events_for_trace(self, trace_id: str) -> list[Event]:
        with self._lock:
            return [
                e.model_copy(deep=True) for e in self._events.values() if e.trace_id == trace_id
            ]

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return [e.model_copy(deep=True) for e in self._events.values()]
