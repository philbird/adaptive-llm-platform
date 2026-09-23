# SQLite backup and restore

`make backup` publishes an atomic **directory containing a matched database pair**. It covers
tenant metadata/content/replays/tombstones and the shared control database's evaluations,
baseline locks, training jobs, models, deployments, lifecycle history and both outboxes.
Key material and filesystem dataset/model/report artifacts must be preserved separately.

```sh
make backup DATA_DIR=.local OUT=/private/tmp/adaptive-backup-pair
```

Expected output: `backup_complete`. The destination must not exist. The command acquires
tenant then control write fences, backs up tenant first and control second with SQLite's
online backup API, and publishes `tenant.sqlite3`, `control.sqlite3` and `pair.json` by one
directory rename. Both copies contain the same UUIDv7 backup version and distinct role; the
manifest records file hashes. SQLite writers cannot change either database during this
snapshot. No live database file is copied directly.

Stop the gateway and all operational writers before restoring:

```sh
make restore DATA_DIR=.local OUT=/private/tmp/adaptive-backup-pair
make restore DATA_DIR=.local OUT=/private/tmp/adaptive-backup-pair FORCE=1
```

An existing destination requires `FORCE=1`; otherwise the command returns
`restore_destination_exists`. Restore verifies hashes, integrity, roles, matching backup
versions and each database's migration set before writing. Both databases are restored in one
SQLite ATTACH transaction using DELETE journals and FULL synchronous mode. SQLite's
multi-database journal commits the pair atomically. A failed second-database restore rolls
back the first too. Success prints `database_restored`; invalid/mismatched pairs return
`restore_pair_failed` and preserve existing databases. The paired CLI requires slice-3a
migration sets and does not accept legacy single-file backups.

Tenant migrations are 1–5 and 7 (old ledgers may retain 6); control migrations are 3, 6 and 7.
Use the original HMAC secret and all required payload key versions. Preserve artifact
directories at the matching point; digest/MAC checks refuse missing or changed artifacts.
Pending events resume independently in the two streams. The in-memory consumer does not
reconstruct already delivered events.

Restore rewinds tombstones. Reconcile later deletion requests and sweep expired content
before serving. Automated external privacy-ledger reconciliation remains outside this slice.

```sh
uv run --locked pytest tests/drills/test_resilience.py -k backup_restore_round_trip -s
uv run --locked pytest tests/unit/test_control_storage.py -s
```

The drill restores the original replay, a control training job and five pending tenant events,
and checks both migration ledgers. Unit tests cover mixed generations even with adjusted
hashes, failure in the second restore, repeated backup/restore, separate migration streams
and preservation of legacy evaluation records.
