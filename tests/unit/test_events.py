from adaptive_llm.contracts import Event, Started, uid
from adaptive_llm.events import InMemoryEventSink


def test_bounded_idempotent_queue_and_trace_readback() -> None:
    sink = InMemoryEventSink(capacity=1)
    event = Event(
        event_type="interaction.started.v1",
        tenant_id="synthetic-a",
        trace_id=uid(),
        data=Started(
            interaction_id=uid(), application_id="support-assistant", policy_version="synthetic-1"
        ),
    )
    sink.emit(event)
    sink.emit(event)
    assert sink.events_for_trace(event.trace_id) == [event]
    assert sink.events_for_trace("other") == []
    assert sink.dropped_events == 0
    sink.emit(event.model_copy(update={"event_id": uid()}))
    assert sink.dropped_events == 1
    sink.emit(event)
    assert sink.dropped_events == 1
    assert sink.events == [event]
