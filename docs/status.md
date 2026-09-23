# Status and gap analysis

The engineering specification v1.0 was supplied on 2026-09-23. Milestone 1 slices 1a–1c
are now implemented and verified locally with synthetic data. Staging and production
acceptance remain separate; no external provider, transport or exporter is configured.

Initial inspection found no existing repository, instructions, CI, deployment configuration,
identity integration or data infrastructure in the new target directory. A neighbouring
Python project uses uv/FastAPI; it provides tooling precedent, not shared infrastructure.

| Area | This checkpoint | Remaining work |
| --- | --- | --- |
| Discovery | Gap analysis, ADRs, owner decision list | Confirm real workload and obtain design approvals |
| Contracts | Initial serving/event models and generated schemas | Version compatibility policy, field classification coverage, full manifest contracts |
| Gateway | Authenticated foundation inference, deadlines, replay, truthful health | Production identity/quotas, SSE |
| Privacy | Purpose policy, two redaction passes, encrypted replay/content, retention and deletion | External deletion propagation and backup reconciliation |
| RAG | Tenant/ACL/residency filtering and exact-version synthetic evidence | Real retrieval service |
| Telemetry | SQLite outbox, retry/quarantine, idempotent sink, in-process metrics and traces | Real transport, exporter, multi-process dispatch |
| Evaluation | Fake provider conformance, privacy, load and recovery drills | Golden/held-out evaluation and real-provider baseline |
| Datasets | Planned boundary | Eligibility, provenance, deduplication, grouped splits, signed manifests |
| Training/registry | Planned boundary | Approved datasets, CPU smoke, LoRA, lineage, evaluation and promotion gates |
| Routing/deployment | Foundation-only routing with residency and integer-micro cost constraints | Shadow, bounded fallback, canary, rollback |
| Research | Disabled configuration intent | Isolated activation/pruning work after earlier milestones |

## Milestone 1, delivered as three reviewable slices

Milestone 1 is split so a running pipeline exists early and hardening is reviewed separately
(specification section 21.1 asks for a vertical slice before broadening).

**1a. Vertical slice (in memory). Delivered 2026-09-23 on branch `slice-1a`; see `docs/runbooks/interaction-walkthrough.md`.** Deterministic fake provider with conformance tests; static
API-key tenant resolution; policy decision and processing-path redaction; synthetic tenant-scoped
retrieval with exact source versions; foundation generation and validation with tokens and
integer-micro costs; correlated events to an in-memory collector; one end-to-end test and a
trace walkthrough. Exit: p95 overhead for logging, classification and routing under the
configured 50 ms on the local load test.

**1b. Persistence. Delivered 2026-09-23 on branch `slice-1b`.** Numbered migrations for SQLite metadata; persistence-path redaction;
AES-GCM encrypted payload refs bound to tenant and interaction ids; idempotent replay of
`request_id`; retention and deletion tombstones; privacy and isolation tests.

**1c. Resilience. Delivered 2026-09-23 on branch `slice-1c`.** Migration
0003 writes content-free events with the operational graph and privacy transactions. A lifespan
dispatcher provides ordered retry, bounded backoff/jitter, dead letters and validation quarantine.
Backlog pressure drops only optional telemetry. Metrics and health expose degradation. Versioned
payload keys, batched rotation and online backup/offline restore have CLI commands and drills.

| Milestone 1 exit criterion | Result | Evidence measured locally on 2026-09-23 |
| --- | --- | --- |
| ≥99.9% valid event correlation under the slice's local load | **Met** | 1,000/1,000 interactions, **100.000%**, 5,000 unique events, 4.634 s; flaky delivery and lost acknowledgements |
| Content policy, tenant isolation, retention and deletion | **Met** | 21 security tests; drill: 200 interactions, 100 swept, 200 deleted, zero blobs/replays/plaintext matches |
| Telemetry degradation preserves serving; p95 overhead <50 ms | **Met** | 200 outage requests, 25 failed deliveries, 1,000 eventually accepted events, **2.487 ms** p95; normal 200-request load **1.615 ms** p95 |
| Dead-letter recovery | **Met** | 5 dead rows reset and delivered through CLI entry points, 0.012 s |
| Backup/restore and migration recognition | **Met** | Original replay matched after mutation/restore, migrations 1–3, 5 pending events recovered, 0.018 s |
| Rotation with live traffic | **Met** | 100 old blobs rotated during 100 new requests; 200 current-key blobs and matching replay, 0.453 s |
| Specification's staging load / production acceptance | **Not met here** | No staging environment or real transport/exporter; local results are not production acceptance |

Run `make check integration`, `uv run --locked pytest tests/security tests/load -s` and
`make drills`. The five drill tests are registered with `pytest.mark.drill` in `tests/conftest.py`
without modifying dependency configuration. See the [walkthrough](runbooks/interaction-walkthrough.md)
and [telemetry outage](runbooks/telemetry-outage.md), [dead-letter recovery](runbooks/dead-letter-recovery.md),
[key rotation](runbooks/key-rotation.md), [backup/restore](runbooks/backup-restore.md), and
[retention/deletion](runbooks/retention-deletion.md) runbooks for commands and measured outputs.

Future increments may broaden to milestone 2 datasets/evaluation, milestone 3 LoRA, milestone 4 routing,
milestone 5 distillation, and optional milestone 6 research. Each remains a separate reviewable
increment. No production acceptance criterion is claimed satisfied at this checkpoint.

