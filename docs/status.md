# Status and gap analysis

The user supplied engineering specification v1.0 on 2026-09-23, then requested a pause
after scaffolding. This checkpoint deliberately stops before milestone 1 implementation.

Initial inspection found no existing repository, instructions, CI, deployment configuration,
identity integration or data infrastructure in the new target directory. A neighbouring
Python project uses uv/FastAPI; it provides tooling precedent, not shared infrastructure.

| Area | This checkpoint | Remaining work |
| --- | --- | --- |
| Discovery | Gap analysis, ADRs, owner decision list | Confirm real workload and obtain design approvals |
| Contracts | Initial serving/event models and generated schemas | Version compatibility policy, field classification coverage, full manifest contracts |
| Gateway | Health-only service | Authentication, tenant resolution, limits, deadlines, idempotency, SSE |
| Privacy | Safe defaults documented; no content collected | Purpose policy, redaction, encrypted storage, retention, deletion propagation |
| RAG | Synthetic fixture and evidence contracts | ACL enforcement, retrieval, exact-version instrumentation |
| Telemetry | Event envelope contract | Operational migrations, outbox, queue, retry, dead letter, metrics/traces |
| Evaluation | Future suite configuration | Fake provider, vertical slice, golden/held-out/safety/retrieval/performance baseline |
| Datasets | Planned boundary | Eligibility, provenance, deduplication, grouped splits, signed manifests |
| Training/registry | Planned boundary | Approved datasets, CPU smoke, LoRA, lineage, evaluation and promotion gates |
| Routing/deployment | Foundation-only configuration intent | Constraints, shadow, bounded fallback, canary, rollback |
| Research | Disabled configuration intent | Isolated activation/pruning work after earlier milestones |

## Next increment: milestone 1 vertical slice

1. Validate contracts against provider/storage/retriever/policy interfaces.
2. Implement the deterministic fake foundation provider with conformance tests.
3. Authenticate and resolve a synthetic tenant; enforce policy before persistence.
4. Retrieve synthetic tenant-scoped chunks and record exact source versions.
5. Generate and validate a foundation response; record tokens and integer-micro costs.
6. Emit correlated metadata events through bounded asynchronous collection.
7. Produce an offline evaluation record, metrics and a trace walkthrough.
8. Add privacy, isolation, retry/dead-letter, failure and load tests plus runbooks.

Only then broaden to milestone 2 datasets/evaluation, milestone 3 LoRA, milestone 4 routing,
milestone 5 distillation, and optional milestone 6 research. Each remains a separate reviewable
increment. No production acceptance criterion is claimed satisfied at this checkpoint.

