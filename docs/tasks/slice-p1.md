# Task: slice P1 — production readiness without owner decisions: signing, observability export, PostgreSQL storage

First production-readiness slice. It contains only work the specification requires that needs no
owner decision (`docs/decisions.md` stays provisional). Read `AGENTS.md`, `docs/status.md`,
`docs/adr/`, `docs/runbooks/backup-restore.md` and `docs/runbooks/key-rotation.md` first.
Milestones 1–6 are on `main` (PRs #1–#12). Everything remains locally verifiable:
PostgreSQL runs in a local container started by the tests when Docker is available and is
skipped with a clear reason otherwise; the SQLite implementations remain the default.

## Deliverables

1. **Asymmetric signing replaces local MACs for artifacts, manifests and reports** (spec 7.12
   "deploy signed artifacts only", 10.2 "verify signatures at deployment", 8.6 "signed dataset
   manifest"). Add a `Signer`/`Verifier` Protocol pair with an Ed25519 implementation from the
   `cryptography` package already in the lockfile: the signer holds a private key loaded from
   `Settings.signing_key_path` (PEM, never in the repo; a test key is generated per test run),
   the verifier holds only public keys keyed by `key_id`. Dataset manifests, dataset approvals,
   model manifests, evaluation reports, benchmark reports and checkpoints gain
   `signature`, `signature_key_id` and `signature_version="ed25519-v1"`; the existing MAC fields
   remain for legacy records and are verified by their recorded version, exactly as the v1/v2
   MAC scheme works today. Verification at load and at every promotion uses the verifier only,
   so a deployment host never holds the signing key. `make sign-rotate` adds a new key id and
   marks the old one verify-only; a test signs with key A, rotates to B, and proves records
   signed by A still verify and new records use B.
2. **OpenTelemetry export and a metrics endpoint** (spec 7.8, 14.1, 14.2, 17). Add an OTLP
   exporter for spans configured by `Settings.otlp_endpoint` (off by default; when off, spans
   stay in-process as now) and a `GET /metrics` endpoint in Prometheus text format for every
   counter and gauge in `Metrics`, with the same bounded labels. The `opentelemetry-sdk`,
   `opentelemetry-exporter-otlp-proto-http` and `prometheus-client` packages are locked and
   present; use them. Tests: the metrics endpoint emits every known
   metric name, never a high-cardinality label, and requires no authentication but is bound to
   loopback by default (`Settings.metrics_bind`).
3. **PostgreSQL implementations of `MetadataStore`, `PayloadStore`, `OutboxStore`, the control
   stores and the registry** (spec 6.3, 17), selected by `Settings.storage_backend =
   "sqlite" | "postgres"` and `Settings.database_url`. `psycopg` 3 (with the binary extra) is locked and
   present. Migrations: reuse the numbered
   SQL files where the dialect is identical and add a `migrations/postgres/` and
   `migrations/postgres/control/` set where it is not; the runner records versions in the same
   `schema_migrations` table. Transactions map to real database transactions; the row-level
   tenant filters, tombstone checks, outbox ordering query and replay cap must behave
   identically, proven by running the existing storage, persistence, outbox, registry and
   privacy test modules parametrised over both backends. The backup and restore commands gain
   a `pg_dump`/`pg_restore` path with the same pair semantics. Tests start PostgreSQL through
   `docker run` with a fixed image tag and skip with a fixed reason when Docker is unavailable;
   never require network in the default gate.
4. **Multi-process dispatch** (the known local cost from slice 1c): with PostgreSQL, the
   outbox dispatcher and the training worker claim rows with `SELECT ... FOR UPDATE SKIP
   LOCKED` so several processes can run safely; with SQLite the single-process lease stays.
   Test with two dispatcher instances against PostgreSQL delivering 1,000 events exactly once.
5. **Docs**: ADR 0005 (signing), ADR 0006 (storage backends), runbooks for signing key
   rotation and PostgreSQL operation, `docs/status.md` production-readiness table listing each
   spec requirement in sections 6.3, 10.2, 14 and 17 as met locally, met with PostgreSQL, or
   still open with the owner decision it waits on.

## Out of scope

Identity provider integration (waits on the tenancy decision), real provider adapters (wait on
the provider decision), KMS integration (waits on the platform decision), Kubernetes and
infrastructure modules. Do not modify `docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None by you. The reviewer has already added and locked these on this branch (audited, no known
vulnerabilities): `opentelemetry-sdk` 1.44.0, `opentelemetry-exporter-otlp-proto-http` 1.44.0,
`prometheus-client` 0.26.0 and `psycopg[binary]` 3.3.6, all as core dependencies. Use them
directly; do not add anything else and do not edit `pyproject.toml` or `uv.lock`. Docker 29.4.0
is installed on the reviewer's host, so the PostgreSQL tests will run in the merge gate; they
must still skip cleanly with a fixed reason where Docker is absent.
