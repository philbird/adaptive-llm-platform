# Task: slice 1b — persistence

Implement milestone 1 slice 1b as described in `docs/status.md`, ADR 0003 and the
specification (`docs/spec/`, sections 6.3, 7.2, 7.8, 8, 9.1, 10.2, 21.3). Read `AGENTS.md` and
`docs/runbooks/interaction-walkthrough.md` first. Slice 1a is on `main`; build on it without
changing its behaviour except where this brief says so.

## Deliverables

1. **Storage boundary** in `src/adaptive_llm/storage/`: Protocols `MetadataStore` (interactions,
   retrieval runs, route decisions, attempts, feedback, deletion tombstones, replay entries) and
   `PayloadStore` (opaque encrypted blobs by reference). Local implementations on SQLite via the
   standard library `sqlite3` only, one file per environment under a data directory from
   `Settings` (default `.local/`, already git-ignored). No ORM. Every table carries `tenant_id`
   and every read is filtered by it; write a negative test that a tenant cannot read another
   tenant's rows through the store API.
2. **Migrations** in `migrations/`: numbered SQL files applied in order inside a transaction,
   with a `schema_migrations` table recording version and applied-at. `Settings` exposes
   `migrate_on_startup` (default true locally). Tests: fresh database applies all; re-running is
   a no-op; a failing migration rolls back and leaves the version unchanged.
3. **Persistence-path redaction** in `src/adaptive_llm/policy/`: a second redactor, separately
   versioned, that applies the tenant's full PII rules (emails, phone numbers, postal codes,
   names are out of scope; use emails, phones and the processing-path classes). It runs on the
   processing-path output immediately before any durable write and records counts under
   `persistence_redaction_counts`. Redaction failure fails closed for persistence only: the
   request still succeeds, nothing content-bearing is written, and the interaction records
   `error_code="persistence_redaction_failed"` with a content-free reason. Test both paths.
4. **Encrypted payload refs**: AES-256-GCM via the `cryptography` package (allowed dependency,
   add it back with `uv lock`). Associated data is `tenant_id|interaction_id|field` so a blob
   cannot be moved between rows or fields. Key from `Settings.payload_key` (32 bytes; default
   derived from the identity secret for local only, with a comment saying production uses a
   KMS). Nonce per blob, never reused; store `key_version` with each blob. Write payloads only
   when the tenant policy allows: `messages_ref`/`query_ref` require
   `content_logging_allowed`, `output_ref` requires `content_logging_allowed`. When not
   allowed, refs stay null and only hashes are stored. Tests: round trip, AAD mismatch fails,
   tampered ciphertext fails, refs null when logging disallowed.
5. **Persist the interaction graph** from the gateway after the response is produced, in one
   transaction per interaction, without blocking the response path beyond the write (no queue
   yet; that is slice 1c). Persist the same five contracts the events carry. The events
   themselves continue to go to the in-memory sink unchanged.
6. **Durable idempotent replay**: replay entries move from the in-memory cache to
   `MetadataStore`, scoped to `(tenant_id, application_id, request_id)` with the fingerprint,
   the stored response (encrypted as a payload, AAD field `replay`), and `expires_at` set from
   `Settings.replay_ttl_seconds` (default 86_400). In-flight reservation stays in memory. Replay
   after process restart returns the original response with `replayed: true`. Keep the
   existing replay tests green and add a restart test.
7. **Retention and deletion**: `Settings.retention_seconds` default from policy
   (`PolicyDecision.retention_seconds`). A `retention.sweep()` function deletes expired payload
   blobs and marks metadata rows `expired`; it is callable from a `make retention-sweep` target
   and tested with a frozen clock. `DELETE /v1/privacy/interactions/{interaction_id}` (tenant
   scoped, 404 for other tenants) writes a tombstone, deletes the payload blobs immediately,
   nulls the refs, and emits `privacy.deletion.requested.v1`. A later write for a tombstoned
   interaction is refused. `POST /v1/privacy/subjects/{subject_id}/deletion-requests` takes the
   raw subject, pseudonymises it the same way the gateway does, tombstones every interaction of
   that pseudonym for the tenant and emits one event per interaction.
8. **Privacy tests** in `tests/security/`: no plaintext content in the SQLite file (open the
   file bytes and assert the synthetic prompt, output and subject are absent); cross-tenant
   read and delete isolation; deletion removes blobs; redaction-failure fails closed for
   persistence only.
9. **Docs**: extend the walkthrough with the persistence steps and what the database contains
   for the walkthrough request; update `docs/runbooks/local-development.md` with the data
   directory, migration and sweep commands; update `migrations/README.md`.

## Out of scope

Outbox, retry, dead letter, telemetry outage drill, key rotation, backup/restore (slice 1c).
Feedback endpoint. Streaming. Real providers. Postgres. Do not modify `docs/spec/`.

## Allowed dependency additions

`cryptography` only. Nothing else. `sqlite3` is standard library.
