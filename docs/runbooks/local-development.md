# Local development and scaffold rollback

Run `make dev`, then `curl http://127.0.0.1:8000/healthz`. Expected response:

```json
{"status":"ok","stage":"scaffold","inference_enabled":false}
```

`make check integration` checks contracts and application startup. No database migration,
provider account, model download or Docker daemon is required. `make contracts` refreshes
checked-in schemas; review the diff before committing contract changes.

Stop with Ctrl-C. This checkpoint creates only the Python environment and development caches;
it creates no user-data store or external resources. No deployment rollback is needed.

Slice 1a adds the interaction walkthrough. Slice 1c adds the telemetry outage, dead-letter
recovery, key rotation, retention/deletion and backup/restore drills. Before live
specialists, add promotion, kill-switch and rollback drills with measured recovery times.

