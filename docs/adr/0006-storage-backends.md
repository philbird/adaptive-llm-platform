# ADR 0006: PostgreSQL with shared storage behavior and separate control schema

Status: accepted for local implementation, 2026-09-24. No platform vendor decision implied.

SQLite remains the default. `Settings.storage_backend="postgres"` selects psycopg 3 and requires
`database_url`. Tenant and control data occupy `adaptive_<environment>_tenant` and
`adaptive_<environment>_control` schemas in one PostgreSQL database. Each has its own
`schema_migrations` ledger. Numbered SQL is reused where compatible; PostgreSQL overrides
provide bytea, generated outbox sequences and JSON-expression indexes.

Repositories retain shared, explicit tenant predicates and the existing MetadataStore,
PayloadStore, OutboxStore and control/registry protocols. A typed Database unit of work
adapts bound parameters and JSON extraction. Write transactions acquire a schema-scoped
PostgreSQL advisory transaction lock to preserve existing check-then-write semantics for
privacy tombstones, replay capacity, approvals, baseline locks and lifecycle transitions.
Read transactions use repeatable-read snapshots. This deliberately conservative writer
serialization is a correctness baseline, not a throughput claim or database-enforced RLS.

Dispatch claims use independent transactions and `FOR UPDATE SKIP LOCKED`; the lock remains
held through sink acknowledgement and state update. Training workers similarly lock trusted
submitter rows, allowing concurrent job progress/cancellation updates. A disconnected worker
releases its claim and running jobs remain restartable. SQLite retains the local worker lease
and single-dispatcher operating model. Delivery remains at-least-once after a sink succeeds
but the database commit/acknowledgement fails; sinks must deduplicate event ids.

One custom-format pg_dump archive includes both schemas at the same exported snapshot.
Restore first validates the archive in a disposable database, then replaces the pair with
one `pg_restore --single-transaction`. This requires an offline restore operator with CREATEDB
and matching PostgreSQL client utilities. Filesystem artifacts and keys are separate backups.
See [PostgreSQL operations](../runbooks/postgresql.md).

The default tests never pull an image. PostgreSQL tests start the cached `postgres:17.6`
container and skip with one fixed reason if Docker, the image or local networking is absent.
The reviewer's host runs those cases before merging; a sandbox skip is not a database pass.
