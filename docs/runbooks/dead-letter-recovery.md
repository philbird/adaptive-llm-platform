# Dead-letter recovery

The dispatcher validates envelope contracts and row identity bindings before delivering.
Malformed JSON, schema mismatches and binding mismatches go directly to `dead` with
`invalid_event`, without retry. Consumer failures become `dead` after the configured maximum
attempts with `sink_unavailable`. Error bodies are never stored. `dead_letter` counts transitions
into dead; `outbox_dead` and the health `dead_letters` field report current rows.

Stop automatic dispatch before manual recovery. List one tenant's dead rows:

```sh
make dead-letters DATA_DIR=.local TENANT=synthetic-a
```

Each output line contains only `<event_id> <event_type> <last_error_code>`. No rows means no
output. Restore the consumer before redelivering sink failures:

```sh
make redeliver DATA_DIR=.local EVENT_ID=<event_id>
make dispatch-once DATA_DIR=.local
```

Expected output: `redelivered=1`, then `processed=<count> pending=<count> dead=<count>`.
Missing or non-dead ids report `redelivered=0`. Redelivery preserves `event_id`, resets attempts
to zero, clears the code and schedules immediate dispatch. Invalid envelopes will be quarantined
again; redelivery does not bypass validation. Diagnose schema/producer defects using ids and
contract definitions. This slice does not expose an envelope editor or print corrupt payloads.

Earlier pending rows block later events of the same interaction. Dead rows do not block, so
recovered rows may arrive after later events. Consumers deduplicate event ids and accept reordering.

```sh
uv run --locked pytest tests/drills/test_resilience.py -k dead_letter_recovery -s
```

Measured locally on 2026-09-23: **5 rows forced dead, 5 reset through the CLI, 5 delivered,
0 pending and 0 dead, elapsed 0.012 s**. Unit coverage also corrupts JSON and envelope bindings,
verifies immediate quarantine, and simulates acceptance followed by a lost acknowledgement.
