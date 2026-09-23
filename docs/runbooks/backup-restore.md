# SQLite backup and restore

Back up the database through SQLite's online backup API. The backup includes migration history,
metadata, encrypted payloads, replay entries, tombstones and all outbox states. Key material is
external and must remain available independently of the database backup.

```sh
make backup DATA_DIR=.local OUT=/private/tmp/adaptive-backup.sqlite3
```

Expected output: `backup_complete`. Use a new output path; an existing destination returns
`backup_destination_exists`. The source connection is locked during the consistent copy;
SQLite coordinates a separately running writer. No copying of a live database file is used.

Stop the gateway and operational writers before restoring. Restore validates a temporary copy,
recognises/applies numbered migrations and publishes it atomically:

```sh
make restore DATA_DIR=.local OUT=/private/tmp/adaptive-backup.sqlite3
make restore DATA_DIR=.local OUT=/private/tmp/adaptive-backup.sqlite3 FORCE=1
```

The first command refuses an existing database with `restore_destination_exists`. With an
absent database, or explicit `FORCE=1`, expected output is `database_restored`. A malformed
backup fails with `restore_failed` and leaves the existing database untouched. Restart using
the same subject/content HMAC secret and every payload key version needed by the backup.
Pending events resume dispatch; previously accepted duplicates remain the consumer's
responsibility. The default consumer is in memory and does not reconstruct previously
delivered events on startup.

A restore rewinds tombstones too. Before serving a restored backup, reapply deletion requests
made since its snapshot and sweep expired content. Automated reconciliation with an external
privacy ledger is outside this local slice.

```sh
uv run --locked pytest tests/drills/test_resilience.py -k backup_restore_round_trip -s
```

Measured locally on 2026-09-23: **backup, delete the live interaction, refuse unforced
overwrite, force restore, replay original response exactly; migrations 1–3 recognised,
5 pending events recovered, elapsed 0.018 s**. Additional unit tests reject malformed backups
without replacing the destination. `OUT` names the backup file for both commands.
