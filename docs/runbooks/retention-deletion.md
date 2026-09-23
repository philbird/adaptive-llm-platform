# Retention and deletion

Retention bounds payload and replay reads immediately. The tenant-scoped sweep physically
removes expired blobs/replay, clears graph refs and marks metadata expired:

```sh
make retention-sweep DATA_DIR=.local TENANT=synthetic-a
```

Expected output is `expired_interactions=<count>`; repeating the sweep reports zero. There
is no background retention scheduler. Metadata, tombstones and content-free outbox envelopes
remain as operational evidence. Historical event refs cannot resolve removed blobs.

Delete an interaction or pseudonymous subject through the authenticated endpoints:

```sh
curl -X DELETE http://127.0.0.1:8000/v1/privacy/interactions/INTERACTION_ID \
  -H 'Authorization: Bearer synthetic-key-a'
curl -X POST http://127.0.0.1:8000/v1/privacy/subjects/deletion-requests \
  -H 'Authorization: Bearer synthetic-key-a' -H 'Content-Type: application/json' \
  -d '{"subject":"synthetic-drill-subject"}'
```

Expected results: 204 for an interaction; `{"deleted":<count>}` for the subject. Other-tenant
or absent interaction ids return 404. The subject is supplied in the body, HMAC-pseudonymised,
and never appears in an event or URL. Deletion commits tombstones, ref clearing, removal of
all payloads/replay, and privacy outbox rows together. A transaction failure rolls all of them
back. Repeating deletion reuses its event id. Subject tombstones prevent subsequent persistence
for that tenant/subject, even if an in-flight request still serves successfully. Privacy events
are never dropped by backlog pressure.

```sh
uv run --locked pytest tests/drills/test_resilience.py -k retention_deletion_under_load -s
```

Measured locally on 2026-09-23: **200 interactions, 100 expired interactions swept while
new traffic runs, 200 subject interactions deleted (including expired metadata), zero blobs,
zero replay rows, zero synthetic plaintext/subject matches in the database, elapsed 0.521 s**.
SQLite secure deletion and DELETE journaling retain the slice-1b behavior. Backups need their
own retention/deletion handling; see [backup and restore](backup-restore.md).
