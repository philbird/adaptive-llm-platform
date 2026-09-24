# PostgreSQL operation

SQLite remains the default. Select PostgreSQL explicitly for both gateway and CLI commands:

```sh
export STORAGE_BACKEND=postgres
export DATABASE_URL='postgresql://user@127.0.0.1/adaptive'
make migrate
make dev
```

Supply passwords through the operator's secret mechanism; do not commit connection URLs.
Non-local environments also require independent payload keys and signing verification keys.
The environment selects separate tenant/control schemas within the database. Schema migrations
are transactional and recorded separately in `schema_migrations`. Selecting PostgreSQL opens
its own schemas; it does not import existing SQLite data. Startup applies the same
numbered files as `make migrate`, with PostgreSQL dialect overrides. The application role
needs schema/table creation for migration; an operator may migrate first and run applications
with `migrate_on_startup=False` and appropriately reduced privileges. Tenant filtering is
application-enforced. TLS, database disk encryption, production roles and platform deployment
remain operator/platform decisions.

Multiple gateway processes can dispatch the same outbox concurrently. Each dispatcher keeps
one lazily opened claim connection per store and thread, reused across claims and closed
on store/database shutdown. Each claim has its own transaction and keeps
a row lock until delivery bookkeeping commits; ordered interaction streams cannot overtake
a pending predecessor. Training workers claim distinct trusted submitter rows, so cancellation
and checkpoint updates can commit while training runs. Processes must share the corresponding
artifact filesystem. PostgreSQL releases claims on connection loss; pending/running jobs
resume from authenticated checkpoints. Configure PostgreSQL connection keepalives/timeouts
for the chosen environment. The training claim holds a transaction for the whole training run:
for the worker role, `idle_in_transaction_session_timeout` must be disabled (`0`) or exceed
`Settings.training_time_limit_seconds`. A dropped claim connection releases its row lock and
leaves the job restartable from its last authenticated checkpoint. Coverage:
`test_worker_claims_are_distinct_recoverable_and_allow_progress` in
`tests/integration/test_postgres.py` checks claim recovery; the backend-parametrised
`test_restart_resumes_running_and_restarts_without_checkpoint` in
`tests/integration/test_training_queue.py` checks that a restarted worker resumes at the next
checkpoint step. External sinks still require idempotent event-id handling after
a crash between successful delivery and the database acknowledgement.

Install matching `pg_dump` and `pg_restore` client utilities on the operational host:

```sh
make backup OUT=/secure/backups/pair-001
# Stop gateways, dispatchers and training workers before restore.
make restore OUT=/secure/backups/pair-001 FORCE=1
```

Backup creates a new directory containing `pair.dump` and `pair.json`, using one repeatable-read
exported snapshot across both schemas, and publishes it by atomic directory rename. Restore
checks the archive hash and schema names, restores into a disposable database, validates both
migration ledgers, then restores the live pair in one transaction. The restore role requires
CREATEDB and permissions to replace both schemas. `FORCE=1` is required for existing schemas.
Failed validation leaves both live schemas unchanged. Preserve the artifact directories,
public signing ring, required legacy MAC secret and payload keys separately. Reconcile newer
deletions and sweep expired content before resuming traffic after any restore.

Tests require a cached fixed image. Provision it explicitly on the reviewer host if absent:

```sh
docker pull postgres:17.6
REQUIRE_POSTGRES=1 make ci
```

Tests use `docker run --pull=never`, bind a random loopback port, create isolated databases,
and clean up their container. The default gate never fetches an image or needs external
network access. All unavailable cases use exactly:
`PostgreSQL requires Docker and cached postgres:17.6 with accessible local networking`.
The reviewer runs `REQUIRE_POSTGRES=1 make ci` so an unavailable PostgreSQL fixture fails
with that same fixed reason instead of skipping. Contributors without Docker retain the
default skip behaviour. Tests never pull images automatically in either mode.
Backup tests run the container's own client binaries; operational commands use host binaries.
The concurrent dispatcher test counts 1,000 actual sink calls with two independent connections,
checks uniqueness and stream ordering, and verifies recovery after an abandoned claim.
