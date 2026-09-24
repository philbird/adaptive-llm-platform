# Status and gap analysis

The engineering specification v1.0 was supplied on 2026-09-23. Milestone 1 slices 1a–1c
and milestone 2 slices 2a–2b plus milestone 3 slices 3a–3b are implemented and tested locally with synthetic data. The reviewer runs `make ci` on the host; every gate passed on 2026-09-24 with the training group installed, including the SBOM audit. Staging and production
acceptance remain separate; no external provider, transport or exporter is configured.

Initial inspection found no existing repository, instructions, CI, deployment configuration,
identity integration or data infrastructure in the new target directory. A neighbouring
Python project uses uv/FastAPI; it provides tooling precedent, not shared infrastructure.

| Area | This checkpoint | Remaining work |
| --- | --- | --- |
| Discovery | Gap analysis, ADRs, owner decision list | Confirm real workload and obtain design approvals |
| Contracts | Serving, dataset and evaluation records with generated schemas and operator APIs | Version compatibility policy, field classification coverage |
| Gateway | Authenticated foundation inference, deadlines, replay, truthful health | Production identity/quotas, SSE |
| Privacy | Purpose policy, two redaction passes, encrypted replay/content, retention and deletion | External deletion propagation and backup reconciliation |
| RAG | Tenant/ACL/residency filtering and exact-version synthetic evidence | Real retrieval service |
| Telemetry | SQLite outbox, retry/quarantine, idempotent sink, in-process metrics and traces | Real transport, exporter, multi-process dispatch |
| Evaluation (milestone 2, slice 2b, delivered 2026-09-23 on branch `slice-2b`) | Five in-process suites; blinded deterministic judge; paired bootstrap and segmented hard gates; local MACs and transactional foundation baseline lock | Real-provider baseline, human review, asymmetric signing and production promotion approvals |
| Datasets (milestone 2, slices 2a–2b) | Authenticated feedback/corrections; current-policy eligibility; exact source provenance; indexed exact/5-gram deduplication and golden decontamination; joint family/subject splits; encrypted shards, pending manifests with local MACs and data cards; operator API/CLI; deletion/reproducibility/concurrency tests; evaluation interactions excluded | Asymmetric signing, external artifact lifecycle |
| Training/registry (milestone 3, slices 3a–3b, delivered 2026-09-24 on branch `slice-3b`) | Signed approval; durable asynchronous queue/cancellation/restart; optional offline CPU LoRA; deterministic safetensors checkpoints and merged/unmerged exports; verified specialist generation; resource limits; exact-artifact evaluation and audited promotion/rollback | GPU/distributed training, external artifact lifecycle, production approvals, traffic routing (milestone 4) |
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

Slice 2a uses migration 0004 for dataset manifests and shares the existing feedback/payload/outbox
transactions. The default serving policy is unchanged; a separate synthetic demo policy opts
tenant A into logging/training and denies tenant B training. See the
[dataset-build runbook](runbooks/dataset-build.md) for commands and limitations. Rebuilds keep
stable content digests while generating new immutable versions and build-start deletion watermarks.
Builds start pending; slice 3a adds signed store-only operator approval. Builds compute outside database locks and recheck deletion tombstones
before publishing one lifecycle event per tenant. Migration 0005 backfills and indexes interaction
start times for SQL window selection. The recovery drill now verifies tenant migrations 1–5/7 and control migrations 3/6/7.

Slice 2b adds migration 0006 (`evaluation_reports`, `baselines`). Evaluation runs on a worker
thread through `InferenceService`, with authenticated tenant grants, `application_id="evaluation"`
and a fixed no-logging/no-training policy. Each evaluation owns an in-memory SQLite database,
a no-op case outbox and private metrics; its connection closes before publication or on failure.
Tenant operational tables, replay entries and the tenant outbox remain unchanged.
The test shards are MAC/hash/AAD verified;
reports contain only identifiers, aggregate metrics, keyed segment labels and per-item scores.
Publication, baseline locking and one `evaluation.completed.v1` event per dataset tenant share
a transaction in a separate evaluation control database and event stream. The local report MAC
uses its own purpose-derived key. Candidate and baseline suites share one event loop.

The local foundation baseline was re-measured after persistence isolation and locked on
**2026-09-23 at 17:54:32 UTC** in
the legacy `.local/evaluation-review-1/evaluations/control/local.sqlite3`
(moved to `.local/evaluation-review-1/control/local.sqlite3` on slice-3a startup): deployment
`fake-foundation-local-1`, dataset `synthetic-evaluation` version
`01a0cf67-a839-7019-a072-be800b15ff1e`, evaluation
`01a0cf67-a83c-77b3-8b4a-0ebeb6c51cdb`. All five suites passed: golden 20, held-out 8,
safety 20, retrieval 8, performance 40 at concurrency 4. Paired mean delta was 0, 95% CI
`[0, 0]`, n=8 overall and on each critical segment. Performance p95 was **2.264 ms**
(the earlier measurement including tenant persistence was 6.777 ms),
error rate 0, cost **42 USD micros per successful outcome**, and TTFT null.
The synthetic planning pilot `[-0.01, 0, 0.01]` has sample SD 0.01, yielding minimum n=4;
this is not a real-workload variance estimate. The lock is a local milestone artifact,
not dataset approval or production promotion. The referenced database and report are local,
git-ignored artifacts and are **not checked in**. The lock is reproducible from the
[evaluation runbook](runbooks/evaluation.md), which also describes replacement semantics,
fixture limits and commands. The demonstration leaves the nine seeded tenant interactions,
35 payloads, nine replay entries and 46 tenant outbox rows unchanged; its two lifecycle events
are stored only in the evaluation control outbox. No scratch directory remains.

Slice 3a uses `migrations/control/` for evaluation/registry tables and a separate tenant
migration to remove empty legacy control tables. The shared store is
`<data_dir>/control/<environment>.sqlite3`. Existing isolated evaluation stores move on startup;
ambiguous paths or populated misplaced tables fail closed. Dataset approval changes only
the stored manifest, with authenticated actor and readable audit reason. Training rechecks
current policy for every manifest tenant and runs off the HTTP event loop. Two signed fake
checkpoints support retry with the same job id; final versions are never overwritten.
All artifact files are digest/MAC verified before specialist loading. Candidate reports bind
registry version, artifact digest, dataset version and the locked foundation baseline.
Approval remains an explicit operator action; shadow/canary/production states do not route
traffic. Emergency rollback uses the previous version's recorded approval history and verified
artifact, independent of later baseline replacement. Adapter architecture identifiers are
extensible and checked against the trainer declaration; multi-dataset manifests are refused
until mixing is supported. Backup/restore atomically covers both database schemas, records
and outboxes;
filesystem artifacts and keys still require separate preservation. See the
[training and promotion runbook](runbooks/training-and-promotion.md).

Slice 3a pass-1 review verification on 2026-09-23: `make check integration` passed formatting, Ruff,
strict mypy (58 source files), 122 unit/contract tests and 89 integration tests. The separate
security/load run passed 43 tests: event correlation 1,000/1,000 (100%), with 5,000 events;
normal p95 overhead 1.681 ms. All five drills passed; paired backup/restore recovered both
migration ledgers, the original replay, one control job and five pending events in 0.054 s.
`pytest -m smoke` passed: CPU train → evaluate → approve → shadow completed in **0.122 s**, below
the ten-second ceiling. These are local synthetic measurements, not production acceptance.

Slice 3b adds control migration 0008 for authenticated queue submitters and a queue index. POST
returns `queued`; a lifespan worker processes jobs in creation order under filesystem leases.
Current policy and approved shards are rechecked before execution. Cancellation, graceful
shutdown and crash recovery use cooperative boundaries and authenticated checkpoints. No identity
comes from the specification. The optional stack is imported only inside `training/lora.py`;
the default fake backend remains available without it. Real bases/tokenizers are verified local
snapshots, never downloads. Only safetensors and numerical/configuration JSON enter real artifacts.
The new golden items check grounded arithmetic/instruction retention and refusal with the pinned
deterministic judge. Real-adapter failures still block promotion.

| Milestone 3 local exit criterion | Slice 3b result measured 2026-09-24 |
| --- | --- |
| Real CPU train → five suites → approve → shadow in <60 s | **2.930 s**, 500 optimizer steps, 47,500 tokens; passing synthetic no-context task, eight held-out items |
| Tiny model and memory measurement | Two layers, 32 hidden units, 259 tokens; admission estimate **844,032 bytes**; process peak RSS **391,905,280 bytes** |
| CPU determinism and checkpoint resume | Identical four-step loss curves, every published file and full artifact digests across fresh runs; interruption at step 2 resumes to the same artifact digest |
| Real generation and gates | Actual tokenizer usage, greedy bounded output; all five ordinary suites executed on a four-step adapter; failed report refuses approval |
| Queue/cancellation/restart and resource limits | 4 fake-worker queue tests plus 8 real-trainer tests pass; trainer architecture remains pinned across workers/retries; cancelled work never registers; 1-byte admission limit refuses loading; zero-second deadline saves checkpoint 0 before failure |
| Formatting, lint, strict types, unit/contract/integration | `make check integration` passes; strict mypy **59** source files; **133** unit/contract passed, **1** GPU skip; **102** integration passed |
| Security/load and recovery | **43** security/load tests and **5** drills passed; correlation **1,000/1,000**, **5,000** events; normal p95 overhead **1.921 ms** |
| CPU smoke gates | **2** passed; fake lifecycle **0.139 s**, real lifecycle **2.930 s** |
| Optional dependencies absent | Full suite with all four training imports blocked: **275 passed, 9 skipped** in **31.55 s** (8 optional CPU tests and 1 GPU placeholder skipped) |
| Required dependency-absence CI gate (review follow-up) | `make check-without-training` now runs after smoke in `make ci`: **275 passed, 9 skipped** in **31.75 s** |
| Full `make ci` | **Passed on the reviewer's host, 2026-09-24**, with the training group installed, including SBOM audit: **284** tests; real smoke approximately **3 s**, same adapter digest on two runs |

The passing real smoke intentionally learns one synthetic no-context response; its retrieval suite
has no relevant documents. It demonstrates lifecycle mechanics, not general safety, RAG quality,
catastrophic-forgetting acceptance or production readiness. Those broader golden/safety/retrieval
fixtures are exercised separately and the tiny model fails their gates. Published files and the full
artifact digest are reproducible: checkpoints live outside the signed export inventory, while wall
clock and peak RSS live only in the job's resource usage. Dependency manifests/lockfiles and serving
traffic behavior remain unchanged.
See the [training runbook](runbooks/training-and-promotion.md) for reproduction and limits.

The pass-1 follow-up also passed full `make ci`, including the SBOM audit and the newly required
dependency-absence gate. Determinism assertions now compare complete published inventories and
artifact digests across two fresh jobs and checkpoint resume. Publication recovery covers partially
restored checkpoint archives, and specialist loading is checked with a relocated export and an
explicit base-model data directory.

Future increments may broaden to milestone 4 routing,
milestone 5 distillation, and optional milestone 6 research. Each remains a separate reviewable
increment. No production acceptance criterion is claimed satisfied at this checkpoint.
