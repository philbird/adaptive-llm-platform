# Local development and stopping the in-memory service

Run `make dev`, then `curl http://127.0.0.1:8000/healthz`. Expected response:

```json
{"status":"ok","stage":"slice-1a","inference_enabled":true}
```

`make check integration` checks contracts and application startup. No database migration,
provider account, model download or Docker daemon is required. `make contracts` refreshes
checked-in schemas; review the diff before committing contract changes.

Follow [the interaction walkthrough](interaction-walkthrough.md) for a synthetic authenticated
inference request, event correlation, replay and failure examples. Additional security and load
checks run with `uv run --locked pytest tests/security tests/load -s`.

Stop with Ctrl-C. Replay responses and collected events exist only in memory and disappear when
the process stops. There is no durable user-data store or external resource to roll back.

Slice 1c adds the telemetry outage, dead-letter
recovery, key rotation, retention/deletion and backup/restore drills. Before live
specialists, add promotion, kill-switch and rollback drills with measured recovery times.
