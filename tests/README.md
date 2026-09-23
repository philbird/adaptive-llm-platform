# Test coverage by milestone

Current tests cover initial contract constraints, UUID/time helpers, event discrimination,
schema drift and health-only application lifecycle. They do not establish production readiness.

Serving implementation must add token/cost accounting, policy/redaction, fake-provider
conformance, tenant RAG ACL/provenance, validation/fallback, event duplicate/reorder/retry/dead
letter, secret leakage, prompt injection/tool abuse, load and dependency outage tests.

Dataset/training increments must add eligibility/deletion/deduplication/grouped-split/leakage,
manifest/config reproducibility, CPU training/evaluation smoke, registry state-machine and
separately marked accelerator tests. No empty test is presented as evidence of those controls.
