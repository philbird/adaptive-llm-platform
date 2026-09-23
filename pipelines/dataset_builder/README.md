# Milestone 2: local dataset factory (slice 2a)

Implementation lives in `src/adaptive_llm/datasets/`. User feedback and authorised correction
refs are recorded through authenticated, transactional gateway endpoints. The factory
re-evaluates current purpose policy, deletion, retention, licences and negative labels;
verifies and redacts decrypted refs and exact-version RAG sources; selects labelled targets;
deduplicates exact/near matches; and decontaminates against 20 synthetic golden items.

Splits use connected components over document families and subject pseudonyms jointly,
with optional time cutoffs. Each build creates immutable tenant-bound encrypted JSONL,
a content-free manifest with pending approval, a local HMAC (`manifest.mac`) and a data card.
Production replaces the MAC with an asymmetric signature from a key the builder does not hold.
Manifest persistence and one `dataset.built.v1` per tenant share the publish transaction.

Use `POST /v1/datasets/builds` with an operator key or
`make dataset-build SPEC=<file>` (optional `DATA_DIR` and `POLICY`). See
[the dataset runbook](../../docs/runbooks/dataset-build.md) for a complete synthetic demo,
artifact format, reproducibility tolerances, deletion behavior and tests.

The factory snapshots metadata and encrypted blobs in a short read transaction, computes
without storage locks and rechecks deletion tombstones in a short publish transaction. Shard
regeneration after a deletion also runs outside the lock. Migration 0005 indexes source-window
selection, and an inverted shingle index prunes near-dedup comparisons. External artifact
lifecycle management, evaluation/baseline locking, human review, training, approval APIs
and registry integrations remain later slices.
