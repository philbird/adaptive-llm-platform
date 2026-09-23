# Test coverage by milestone

Current tests cover initial contract constraints, UUID/time helpers, event discrimination,
schema drift and health-only application lifecycle. They do not establish production readiness.

Milestone 1 covers token/cost accounting, policy/redaction, fake-provider conformance, tenant
RAG ACL/provenance, validation, event retries/dead letters, privacy, local load and recovery
drills. Dataset tests cover eligibility, deletion, deduplication, grouped/time splits,
encrypted manifests, source provenance, reproducibility and publication races.

Slice 2b adds `tests/unit/test_evaluation.py`, `tests/integration/test_evaluations.py` and
`tests/security/test_evaluation_privacy.py`. They exercise the five suites through the fake
serving pipeline, paired bootstrap/sample-size arithmetic, bounded contracts, blinded judge
ordering and disagreements, nontrivial retrieval ranks, failed-outcome costs, baseline
lock/replacement/idempotency, CLI/API parity, transactional outbox/artifact rollback, concurrent
locks and serving, operator scope, artifact tampering, content-free reports and exclusion of
evaluation interactions from datasets. Exact tenant operational row counts remain unchanged
after evaluation, including payloads, replay entries and outbox rows. Tests verify private
in-memory scratch cleanup on success and failure, isolated completion-event delivery, a shared
candidate/baseline event loop, runtime judge/rubric registration and null baseline event versions.
An identical candidate passes; a citation-dropping
wrapper fails the citation and safety segments. Missing/undersized samples, CI boundary
contact, absent coverage and a changed baseline manifest fail closed.

Synthetic fixtures live in `tests/fixtures/golden`, `tests/fixtures/safety` and
`tests/fixtures/retrieval`. The evaluation fixture seeds one train and eight test examples
with exact source provenance; it does not manufacture shards outside the dataset builder.
The migration and backup/restore tests now verify migrations 1–6.

Run `make check integration`, `uv run --locked pytest tests/security tests/load -s`, and
`make drills`. Run `make contracts` when contracts change. See the
[evaluation runbook](../docs/runbooks/evaluation.md) for an executable local baseline demo.
CPU training, registry transitions, real-provider evaluation, human review and accelerator
tests remain future work. No empty test is presented as evidence of those controls.
