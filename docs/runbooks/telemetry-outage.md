# Telemetry outage

Serving commits metadata and content-free envelopes to SQLite before returning. A sink failure
leaves pending rows with `sink_unavailable`; it does not change the inference response. The
lifespan dispatcher retries with equal jitter and capped exponential backoff. Defaults are
0.1 seconds base, 30 seconds cap and 8 attempts including the first delivery. Dead rows require
operator redelivery after the consumer recovers.

A storage failure takes a separate degraded path: `persistence_failures` increments and the
persistence worker emits the same content-free envelopes directly to the configured `EventSink`.
`degraded_emissions` counts each attempted fallback emission, including rejected attempts.
Sink errors are swallowed independently for each envelope. No prompt, output, retrieved text
or error body is buffered. Fallback events are **not durable**, are not retried, and may be
duplicated once the store recovers; consumers must deduplicate by `event_id`.

This also applies to deletion-request events. Storage failures still fail the deletion operation;
a fallback request event is not confirmation that deletion committed. Known interaction trace
ids are retained. If the store cannot resolve an interaction, the event has the requested
scope/target and a new trace id. Normal serving waits only for its persistence transaction;
the exceptional fallback attempts the sink directly in the worker before returning.

Inspect `GET /healthz`: expected `status: ok`, `stage: milestone-1-local`,
`inference_enabled: true`, and increasing `outbox_pending` while delivery is unavailable. `dead_letters` counts exhausted or invalid rows;
readiness remains true. In-process metrics expose pending/lag gauges and retry/dead/drop counters.
There is no metrics HTTP endpoint or exporter in this slice.

At 100,000 pending rows by default, started/retrieval/route events are dropped on enqueue and
counted. Generation, completion and deletion events remain durable. This threshold is a
telemetry degradation control, not a limit on billing/privacy evidence or disk usage.

Known local cost: `SQLiteOutboxStore.enqueue` runs `count(*) WHERE state = 'pending'` on each
save. Even with the pending-state index, this scans the pending entries, up to and beyond the
100,000-row default threshold. This is acceptable for the local slice; replace this counting
strategy as part of future multi-process transport work.

After stopping the gateway's dispatcher, run a bounded local batch:

```sh
make dispatch-once DATA_DIR=.local
make dead-letters DATA_DIR=.local TENANT=synthetic-a
```

Expected batch output: `processed=<0..256> pending=<count> dead=<count>`. Repeat after retry
backoff elapses until pending is zero. Only one dispatcher may operate on a local database.
The CLI uses the in-memory consumer and discards its process memory on exit; the delivered
envelopes remain in SQLite. Tests can inject an `EventSink` into the same CLI entry point.
An external consumer must deduplicate event ids and tolerate reordering. Delivery is at least
once; the consumer can receive duplicates after an acknowledgement or process failure.

Run the measured outage drill and full correlation test:

```sh
uv run --locked pytest tests/drills/test_resilience.py -k telemetry_outage -s
uv run --locked pytest tests/load/test_event_correlation.py -s
```

Measured locally on 2026-09-23: **200 requests, 25 failed deliveries, 1,000 unique events
accepted, p95 overhead 2.487 ms (<50 ms), elapsed 0.712 s**. Correlation under a flaky consumer,
including lost acknowledgements: **1,000/1,000 interactions (100.000%), 5,000 unique events,
4.634 s**. Each interaction had the complete ordered event set and matching trace, interaction,
retrieval, route and attempt ids. These are local synthetic measurements, not staging results.
