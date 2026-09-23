# Local development and persistence

Run `make dev`, then `curl http://127.0.0.1:8000/healthz`. The existing health contract remains:

```json
{"status":"ok","stage":"slice-1a","inference_enabled":true}
```

`Settings(inference_enabled=False)` still omits inference and reports false. Enabled local
applications create `.local/local.sqlite3` on startup. `Settings.data_dir` selects the directory;
`Settings.environment` selects the filename (`local`, `development`, `staging`, `production`).
Authenticated environment must match the application environment. The directory is git-ignored.
No provider account, model download or Docker daemon is required. Only `cryptography` was added
for AES-256-GCM; SQLite is from the standard library.
`create_app()` and imports create no database or directories. Lifespan startup opens the database
and runs migrations, and shutdown closes it. Tests using HTTPX's ASGI transport must explicitly
enter `app.router.lifespan_context(app)`; `TestClient` does this as a context manager.

Numbered migrations run transactionally by default (`Settings.migrate_on_startup=True`). To apply
them explicitly, or sweep an existing database:

```sh
make migrate
make retention-sweep TENANT=synthetic-a
make retention-sweep TENANT=synthetic-b
make migrate DATA_DIR=/private/tmp/synthetic-data ENVIRONMENT=local
make retention-sweep DATA_DIR=/private/tmp/synthetic-data ENVIRONMENT=local TENANT=synthetic-a
```

The equivalent CLI is `uv run --locked python -m adaptive_llm.storage migrate --data-dir .local
--environment local`, or `retention-sweep --data-dir .local --environment local --tenant synthetic-a`.
Sweeps require an explicit tenant and an already migrated database. No automatic sweep scheduler
exists. Set `migrate_on_startup=False` only after applying migrations with the same data directory.

Policy retention defaults to 3600 seconds for both synthetic tenants. `Settings.retention_seconds`
defaults to `None` (use policy); a positive override is available locally. Replay TTL defaults to
86,400 seconds and is capped by retention. Expiry immediately prevents content reads/replay;
sweeps physically remove payloads, clear refs and mark metadata expired. Privacy deletion
immediately removes payloads and records permanent tombstones. See
[the interaction walkthrough](interaction-walkthrough.md) for request/deletion commands and the
contents of each stored record, including redacted replay as required by specification 9.1.
`Settings.replay_capacity` independently bounds in-flight requests in memory and completed
replay rows per tenant in SQL. Completed entries are never loaded into a memory cache.

`Settings.payload_key` accepts exactly 32 bytes. Its local-only default is derived from the
identity secret; changing either key breaks old payload decryption. Non-local environments
require an explicit key (production uses KMS). Preserve keys and the data directory across
restarts when testing durable replay. Injected `MetadataStore` and `PayloadStore` implementations
must share the metadata transaction's atomic unit of work. The local implementation uses one
SQLite connection with foreign keys, secure deletion and DELETE journaling.
Persistence writes run in worker threads and retain the SQLite `RLock` transaction boundary.
Emitted and stored retrieval/attempt records share persistence-path hashes; only refs may be
added at commit time. Failed redaction stages have null hashes, never hashes of an empty string.

Run the review checks:

```sh
make contracts
make check integration
uv run --locked pytest tests/security tests/load -s
```

`make contracts` refreshes checked-in schemas and OpenAPI. Tests use isolated temporary data
directories, sharing one only for explicit restart tests. No test writes request data to `.local`.
`application.state.persistence.failures` reports failed writes; event failures and drops retain
their existing counters. No durable telemetry retry is included yet.

Stop with Ctrl-C. The database and completed replay entries survive; in-memory events and
in-flight reservations do not. Use one gateway process for local in-flight exclusion. Subject
deletion accepts `{"subject":"<raw>"}` at `POST /v1/privacy/subjects/deletion-requests`, never in a
URL. Its subject tombstone prevents late-arriving interactions for the pseudonym from persisting.
`make dev` retains disabled access logging; request bodies must never be logged.

Slice 1c adds telemetry outage/dead-letter recovery, key rotation and backup/restore drills.
Streaming, real providers, the feedback endpoint and Postgres remain outside slice 1b.
