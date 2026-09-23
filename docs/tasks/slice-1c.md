# Task: slice 1c — resilience: outbox, retry, dead letter and drills

Implement milestone 1 slice 1c as described in `docs/status.md` and the specification
(`docs/spec/`, sections 7.1, 7.8, 9.4, 14.2, 15.4, 19 milestone 1 exit criteria, 21.3, 21.4).
Read `AGENTS.md` and `docs/runbooks/interaction-walkthrough.md` first. Slices 1a and 1b are on
`main`; build on them without changing their behaviour except where this brief says so.

## Deliverables

1. **Durable outbox.** Events that must correspond to committed operational state
   (`interaction.started`, `retrieval.completed`, `route.decided`, `generation.*`,
   `interaction.completed`, `privacy.deletion.requested`) are written to an `outbox` table in
   the same transaction as the interaction graph (migration `0003_outbox.sql`), then delivered
   asynchronously. The in-memory sink from slice 1a becomes one `EventSink` implementation
   among others; the gateway no longer emits directly to a sink during the request. Rows carry
   `event_id`, `tenant_id`, `trace_id`, `event_type`, the envelope JSON, `attempts`,
   `next_attempt_at`, `state` (`pending`, `delivered`, `dead`) and a content-free
   `last_error_code`. Serving must never wait on delivery.
2. **Dispatcher.** A background task started in the lifespan (and a `make dispatch-once`
   CLI for tests and drills) reads pending rows in `next_attempt_at` order, delivers each to
   the configured `EventSink`, marks `delivered`, and on failure schedules a retry with bounded
   exponential backoff plus jitter from `Settings.outbox_backoff` (base, cap, max attempts).
   After the maximum, the row moves to `dead` and a `dead_letter` counter increments. Delivery
   is idempotent on `event_id`; the sink must tolerate duplicates and reordering. Preserve
   per-interaction order where feasible: deliver rows of one `interaction_id` in `rowid` order
   and do not deliver a later row while an earlier one of the same interaction is pending.
3. **Quarantine.** A row whose envelope fails contract validation at dispatch time (simulate
   by corrupting the JSON in a test) moves to `dead` with `last_error_code="invalid_event"`
   without retries. `make dead-letters TENANT=<id>` lists dead rows by id, type and error code
   only, and `make redeliver EVENT_ID=<id>` resets one row to pending.
4. **Telemetry-degradation behaviour.** With the outbox in place, a sink outage only grows
   the pending backlog. Add `Settings.outbox_pending_limit`; when exceeded, new
   non-billing events (`interaction.started`, `retrieval.completed`, `route.decided`) are
   dropped with a `dropped_events` counter, while `generation.*`, `interaction.completed` and
   `privacy.deletion.requested` are always written. Serving continues regardless. This
   implements spec 7.1: emit a counter, never retain unredacted content as a fallback.
5. **Metrics.** Add a minimal `Metrics` Protocol with an in-process implementation exposing
   counters and gauges (no exporter yet): requests by status class, fallback-free attempts,
   outbox pending, delivered, retried, dead, dropped, persistence failures, replay hits,
   dispatcher lag seconds (now minus oldest pending `next_attempt_at`). Labels are bounded:
   tenant id is allowed, request/interaction/trace ids are not. `GET /healthz` gains
   `outbox_pending` and `dead_letters` fields; readiness stays true (degradation is not
   unhealthy).
6. **Key rotation.** `PayloadCipher` accepts a keyring of `key_version -> key` with one
   current version; decrypt selects by the blob's `key_version`. Add `make rotate-key
   NEW_KEY_VERSION=<v>` which re-encrypts every live blob for a tenant under the current key
   in batches inside transactions, and a test proving old blobs decrypt before and after and
   that a blob missing from the keyring fails with a fixed code.
7. **Backup and restore.** `make backup DATA_DIR=<d> OUT=<file>` uses SQLite's online backup
   API to a file; `make restore` refuses to overwrite an existing database unless
   `FORCE=1`. Test: back up, mutate, restore, verify the restored database replays the
   original request and that migrations recognise its version.
8. **Drills as tests** under `tests/drills/` (marked `@pytest.mark.drill`, run by
   `make drills`), each printing a one-line measured result:
   - telemetry outage: sink raises for the first N deliveries; serving p95 stays under the
     50 ms overhead target and every event is eventually delivered exactly once;
   - dead-letter recovery: force rows dead, `redeliver`, assert delivery;
   - retention and deletion under load: 200 interactions, sweep, subject deletion, assert
     no blobs and no plaintext in the file;
   - backup and restore round trip;
   - key rotation with live traffic.
9. **Exit criterion of milestone 1.** A `tests/load` test creating 1 000 interactions with a
   flaky sink asserts at least 99.9 % valid event correlation after dispatch drains: every
   interaction has its full event set delivered with matching ids. Print the ratio.
10. **Docs.** Runbooks under `docs/runbooks/`: `telemetry-outage.md`, `dead-letter-recovery.md`,
    `key-rotation.md`, `backup-restore.md`, `retention-deletion.md`, each with the command,
    expected output and the measured result from the drill. Update `docs/status.md` milestone 1
    exit criteria as met or not met with the numbers. Update the walkthrough for the outbox.

## Out of scope

Real message transport (Kafka etc.), metrics exporter, Postgres, streaming, feedback endpoint,
real providers, task classifier. Do not modify `docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None.
