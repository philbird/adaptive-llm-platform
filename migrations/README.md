# SQLite migrations

`0001_persistence.sql` creates tenant-scoped interactions, started records, retrieval runs,
route decisions, generation attempts, feedback, deletion tombstones, replay entries and opaque
encrypted payloads. Metadata stores canonical contract JSON plus ids, expiry and lifecycle state.
Payloads store ciphertext, nonce, key version and binding metadata, never plaintext content.
Composite foreign keys include tenant ids. `0002_subject_tombstones.sql` adds tenant-scoped
subject tombstones and the index for SQL replay eviction in reservation order.
`0004_dataset_manifests.sql` stores content-free dataset manifests.
`0005_interaction_started_at.sql` backfills start times from interaction JSON (normalising UTC
`Z` to `+00:00` to match storage writes) and indexes `(tenant_id, started_at)`. Dataset source
windows use an inclusive SQL lower bound and exclusive upper bound, preserving microseconds.
`schema_migrations` is a database-wide ledger with only `version` and UTC `applied_at`; it is not
tenant data. The runner transactionally removes the legacy ledger's `tenant_id` column when
upgrading an existing slice-1b database, preserving its applied versions and timestamps.

Add migrations as numbered `NNNN_description.sql` files; do not edit an applied version.
Use complete SQL statements terminated with semicolons, with each statement ending on its own
line. Do not include transaction controls. The runner applies all pending files in numeric order
inside one `BEGIN IMMEDIATE` transaction, recording version and UTC applied-at timestamps in
the same transaction. Re-running is a no-op; any failure rolls back the entire pending batch,
including schema changes and version records. An existing applied version remains unchanged.

Local startup migrates by default. Run `make migrate` explicitly before starting with
`Settings.migrate_on_startup=False`. `DATA_DIR` (default `.local`) and `ENVIRONMENT` (default
`local`) select `<data-dir>/<environment>.sqlite3`. Do not manually change operational tables.

`make retention-sweep TENANT=synthetic-a` deletes expired payloads and replay entries, clears
references and marks graph rows expired. Tombstones are permanent and refuse later writes for
the deleted interaction id or subject pseudonym. A completed replay cap is enforced per tenant
inside `put_replay`, deleting excess oldest entries and their payloads in the same transaction.
SQLite uses `secure_delete=ON` and DELETE journals; expiry alone
does not physically remove a blob until a sweep runs. Privacy deletion removes it immediately.

Unit/integration/security tests cover fresh application, repeat application, failed-batch rollback,
legacy-ledger upgrades, graph/payload transaction rollback, tenant isolation, ciphertext
integrity, restart replay, per-tenant SQL eviction, retention, and atomic subject tombstones.
Outbox, retry/dead letter, key rotation and backup/restore belong to 1c.


Slice 3a separates migration streams. Root migrations apply only to tenant databases.
`0007_control_separation.sql` removes empty legacy evaluation tables and refuses to drop
populated ones. Applied tenant ledgers may retain version 6 from slice 2b. Control migrations
live in `control/`: 0003 creates its outbox, 0006 creates evaluation reports/baselines, and
0007 creates jobs, model versions, transitions and deployments while removing empty tenant
tables inherited from the old control store. That upgrade preserves control rows and outbox
events. The runner receives a directory; it contains no special-case control table logic.

`make migrate` opens both databases. The control store is `<data-dir>/control/<environment>.sqlite3`.
Startup moves the old `evaluations/control/` directory when the new one is absent; ambiguous
paths fail closed. Backups and restore validate both migration streams; see the
[paired backup runbook](../docs/runbooks/backup-restore.md).
