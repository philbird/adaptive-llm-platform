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

## Next increment: milestone 1, delivered as three reviewable slices

Milestone 1 is split so a running pipeline exists early and hardening is reviewed separately
(specification section 21.1 asks for a vertical slice before broadening).

**1a. Vertical slice (in memory). Delivered 2026-09-23 on branch `slice-1a`; see `docs/runbooks/interaction-walkthrough.md`.** Deterministic fake provider with conformance tests; static
API-key tenant resolution; policy decision and processing-path redaction; synthetic tenant-scoped
retrieval with exact source versions; foundation generation and validation with tokens and
integer-micro costs; correlated events to an in-memory collector; one end-to-end test and a
trace walkthrough. Exit: p95 overhead for logging, classification and routing under the
configured 50 ms on the local load test.

**1b. Persistence.** Numbered migrations for SQLite metadata; persistence-path redaction;
AES-GCM encrypted payload refs bound to tenant and interaction ids; idempotent replay of
`request_id`; retention and deletion tombstones; privacy and isolation tests.

**1c. Resilience.** Durable outbox, retry, dead letter and quarantine; telemetry outage drill
proving serving is unaffected; dead-letter recovery, key rotation, retention/deletion and
backup/restore drills and runbooks. Exit: ≥99.9% valid event correlation under local load.

Only then broaden to milestone 2 datasets/evaluation, milestone 3 LoRA, milestone 4 routing,
milestone 5 distillation, and optional milestone 6 research. Each remains a separate reviewable
increment. No production acceptance criterion is claimed satisfied at this checkpoint.

