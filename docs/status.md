# Status and gap analysis

The engineering specification v1.0 was supplied on 2026-09-23. Milestone 1 slices 1a–1c
and milestone 2 slices 2a–2b plus milestone 3 slices 3a–3b, milestone 4 slices 4a–4b and
milestone 5 slice 5a and optional milestone 6 slice 6a are implemented
and tested locally with synthetic data. The reviewer runs `make ci` on the host; the slice 3b
checkpoint passed every gate on 2026-09-24 with the training group installed, including the SBOM
audit. The reviewer also ran full `make ci` for slice 4a on the host on 2026-09-24 and every
gate passed, including the SBOM audit. The reviewer ran `make ci` for slice 5a on the host on
2026-09-24: every gate passed, including the SBOM audit, with 396 tests, eight drills and three
smoke paths (student approximately 3.5 s); measurements are recorded below. Staging and production
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
| Training/registry (milestone 3, slices 3a–3b; router jobs added in 4b) | Signed approval; durable asynchronous queue/cancellation/restart; optional offline CPU LoRA; deterministic exports; verified specialist generation; plain-Python calibrated router jobs; exact-artifact evaluation and audited promotion/rollback | GPU/distributed training, external artifact lifecycle, production approvals |
| Distillation (milestone 5, slice 5a, delivered 2026-09-24 on branch `slice-5a`) | Isolated teacher curation, encrypted approved datasets, preserved evidence/folds, general/safety mix, full/LoRA tiny students, optional safetensors KL, teacher non-inferiority and signed deployment benchmarks | Real workload quality and hardware/cost acceptance; production approvals |
| Routing/deployment (milestone 4, slices 4a–4b, delivered 2026-09-24 on branch `slice-4b`) | Immutable policies; shadow comparisons; numeric encrypted router datasets; calibrated logistic/OOD admission; opt-in canary/production responses; segmented outcome costs; persistent automatic rollback and kill controls | Real workload calibration, production approvals, external transport, later bandits |
| Research (optional milestone 6, slice 6a, delivered 2026-09-24 on branch `slice-6a`) | Disabled by default; isolated aggregate studies, encrypted safetensors, physical structured pruning, full tuning, ordinary candidate registry and independent-process benchmarks | Demonstrated hardware benefit, representative quality/safety acceptance, production approval |

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
At the slice 3a checkpoint, approval remained explicit and shadow/canary/production states did
not route traffic; slice 4b now opts canary/production into live policies. Emergency rollback uses the previous version's recorded approval history and verified
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

Slice 4a adds control migration 0009 for immutable route-policy versions, an independent
active pointer, disablement controls, audit history, shadow coverage and content-free comparisons.
A bounded post-response queue loads only verified tenant-eligible shadow adapters. Attempt 0
is marked shadow, has its own optional encrypted output ref and outbox event, and cannot alter
the response, replay, final attempt or billed live cost. Comparisons use the explicitly labelled
`shadow-chunk-overlap-1` grounding proxy and paired bootstrap by critical segment; these scores
are not the evaluation rubric. Validators add five-gram grounding, citation presence,
English-script language, repetition/truncation, tool allowlist and application domain checks.
Hard checks control acceptance; heuristic/advisory failures are recorded and counted, with
validated per-application severity overrides. Token-budget completion remains a 200/`length`
response. Verified specialists use an eight-entry version/digest cache, with registry eligibility
rechecked on every access and state/digest changes evicting stale entries.

At the slice 4a checkpoint, live specialist enablement was rejected by the public contract. An injected planner tested the
bounded chain's ordering, budgets, deadlines, failure suppression and maximum attempts.
Breakers use bounded process-local sliding windows and single half-open probes. Environment,
tenant and task disablement survive restarts and propagate through a one-second pointer cache.
The local two-instance kill-switch drill measured **1.006 s** (repeat **1.014 s**) with **100/100** successful requests
under load and zero shadow starts on ten post-effect requests. These are synthetic local
measurements, not production acceptance. See the
[shadow and kill-switch runbook](runbooks/shadow-and-kill-switch.md).

Slice 4a verification on 2026-09-24: formatting, Ruff, strict mypy (63 source files),
162 unit/contract tests (one existing GPU skip), 118 integration tests, 43 security/load tests,
six drills and both smoke tests passed. Event correlation remained 1,000/1,000 with 5,000 events;
normal p95 overhead was 2.110 ms. Real CPU smoke completed in 2.970 s. The required optional-stack
absence gate passed 321 tests with nine skips. The reviewer ran `make ci` on the host and every
gate passed on 2026-09-24, including the SBOM audit: 330 tests, six drills, kill-switch effect
1.01 s. Dependency files and the specification were unchanged.

The slice 4a pass-1 corrections also passed full `make ci` on 2026-09-24, including the
SBOM audit (no known vulnerabilities). The updated suite collected 345 tests: 176 unit/contract
passed with one existing GPU skip, 119 integration passed, 43 security/load passed and six
drills passed; both smoke tests passed again. Kill-switch effect was 1.010 s. The optional-stack
absence gate passed 336 tests with nine expected skips. The new coverage checks advisory
severities and application overrides, preserved budget-limited responses, provider cache reuse
and revocation/digest eviction, and the distinct shadow proxy version.

## Milestone 4 slice 4b — calibrated live routing and rollback

Slice 4b is implemented locally without commits or dependency changes. The reviewer ran
`make ci` on the host on **2026-09-24** and every gate passed, including the SBOM audit:
**371 tests**, **eight drills**, automatic rollback **1.03 s** breach-to-foundation,
novel-task foundation routing with `out_of_distribution`, and kill-switch propagation **1.0 s**.
Router datasets use the existing builder, current-policy/deletion checks, encrypted tenant/split
shards, approval and
source-version lineage. The numeric rows preserve absent counterfactual coverage. Router jobs
use the ordinary durable orchestrator and signed registry artifacts, with independent logistic
fit, Platt calibration and test folds. The new routing suite gates false-specialist rate and
ten-bin ECE. Approved router versions enable live policies; only canary/production specialists
can answer users, and hard controls precede calibrated admission.

Control migration 0010 adds persistent per-specialist disablement, live observations and
rollback measurements. The monitor checks configured hard thresholds, including critical
segments, and atomically stores the triggering numbers, disablement audit and deployment event.
It never promotes. Shadow and canary evidence gate specialist progression; production still
requires an authenticated operator note. Cost comparisons retain failed attempts and fallback
cost, expose actual-serving cohorts separately, and exclude shadow cost from live savings.

| Milestone 4 local exit criterion | Slice 4b measured result (2026-09-24) |
| --- | --- |
| Numeric counterfactual data and reproducible calibrated router | 60 shadow observations, 40 train / 10 calibration / 10 test; identical signed artifact digests across fresh jobs; coverage gaps retained as nulls |
| Router promotion gates | Ten-item fold: false-specialist **0**, unnecessary-foundation **0**, ECE **0.00017646**; .021 false-specialist and .101 ECE block approval. This fold has zero natural OOD samples; the separate novelty drill supplies that coverage |
| Live canary non-inferiority | Five matched specialist/control outcomes; proxy mean delta **0**, 95% CI **[0, 0]**; no hard validation/error/safety failures |
| Live synthetic savings | Specialist **4** versus foundation **13** USD micros per success: **69.23%** reduction |
| Live latency and safety | Measured specialist cohort p95 **1.875 ms**, below 5,000 ms; hard validation failures, endpoint errors and critical safety incidents **0** in the passing canary |
| Business outcome with fallback | Ten successful outcomes, **12 attempts**, two fallbacks; specialist cohort total **66** micros, displayed per-success **7** (exact **6.6**) versus foundation **13**; reduction **49.23%** |
| Independent arithmetic fixture | Specialist total **400**, ten successes, per-success **40** versus **100**; reduction **60%**; shadow cost **56** separate; one extra failed outcome raises per-success to **50** |
| Automatic rollback under five seconds | Injected critical breach → persistent disablement → foundation in a second app instance: **1.012 s** (focused repeat **1.014 s**); validation-failing live wrapper also triggers automatic disablement |
| Existing kill switch and load gates | Kill-switch effect **1.001 s**, 100/100 load requests successful, zero subsequent shadows; event correlation **1,000/1,000**, 5,000 events; normal p95 overhead **1.993 ms**, below 50 ms |
| Live-path overhead after review pass 1 | **200/200** responses from one production-state specialist at **100%** assignment; full-CI p95 overhead excluding provider time **2.375 ms**, including classification, routing, logging, persistence and HTTP; p95 routing **0.169 ms**, below the **50 ms** overhead target. Focused run: overhead **2.786 ms**, routing **0.187 ms** |
| Novel task abstention | Foundation response with `out_of_distribution`; measured OOD **0.739**, policy ceiling **0.15** |
| Deterministic canary assignment | Stable SHA-256 assignment over 10,000 ids within the tested 5% range; shadow state never served live; canary capped at 5%, production at 100% |

These are local synthetic measurements. Task/language/risk features still use the existing
RAG-flag/English/medium classification. Quality uses the coarse shadow chunk-overlap proxy;
task/risk pairing is observational matching, not identical-prompt counterfactual inference.
OOD coverage is reported explicitly, including unmeasured folds. Router `approved` is the
explicit promoted state required by live policies; generative specialists retain the full
shadow/canary/production progression. Immutable artifacts and current controls remain distinct.
Review pass 1 adds snapshot-cadence admission-manifest caching, a bounded verified-router cache
keyed by version/digest, eviction on registry state/digest changes, and observable fail-closed
planner exceptions via `live_planner_failures` on `/healthz`. The internal price-comparison
reason is `not_cheapest`; request-budget errors retain `cost_limit_exceeded`.
See [canary and rollback](runbooks/canary-and-rollback.md) for commands and detailed limitations.

Following slice 5a, the next optional increment is milestone 6 research. Bandits,
real providers, streaming, semantic classification and production acceptance remain outside
this slice. The specification, `pyproject.toml` and `uv.lock` are unchanged.

Initial implementation verification commands use the already provisioned environment, with the exact prefix
`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache`:

| Command after that prefix | Result |
| --- | --- |
| `make contracts` | Generated JSON schemas and OpenAPI synchronized; schema consistency gate passed |
| `make check integration` | Formatting, Ruff, strict mypy on 67 source files; 194 unit/contract passed, one existing GPU skip; 125 integration passed |
| `make ci` | Reviewer host run passed every gate, including the SBOM audit, on 2026-09-24: 371 tests and eight drills |
| `uv run --locked pytest tests/security tests/load -s` | 43 passed; correlation and overhead figures above |
| `uv run --locked pytest tests/drills -m drill -s` | Eight passed, including rollback, novel task and the existing kill switch; backup/restore verified control migration 0010 |
| `uv run --locked pytest -m smoke -s` | Two passed; real CPU LoRA lifecycle 2.904 s, fake lifecycle 0.192 s |
| `make check-without-training` | 362 passed, nine expected optional-stack/GPU skips; the full suite runs with training imports blocked |
| `uv run --locked pytest tests/integration/test_live_routing.py -q -s --tb=short` | Six passed; measured calibration, live latency, cost and fallback evidence above |

Focused development also ran `uv run --locked ruff format src tests`,
`uv run --locked ruff check src tests --fix`, `uv run --locked mypy src/adaptive_llm`, and
`uv run --locked pytest tests/unit/test_router_model.py tests/unit/test_canary.py -q --tb=short`.
Shadow/training regressions, all-candidate admission, and both new drills were run independently
before the full gates. Early failures in shadow cost updates and breaker telemetry were fixed;
no gate was disabled. Recovery-only legacy registry tests inject passed progression evidence
so they continue to test emergency rollback independently of mutable rollout gates; new live
tests exercise the real progression checks.

Review pass 1 verification on 2026-09-24: `make contracts` and `make ci` passed end to end,
including the SBOM audit (no known vulnerabilities), formatting, lint and strict mypy.
There were **202** unit/contract passes (one existing GPU skip), **126** integration passes,
**44** security/load passes, **eight** passing drills and **two** passing smoke tests;
`make check-without-training` finished with **372 passed, nine expected skips**.
The focused planner, metric, live-routing, application and overhead tests also passed (**30**).
This full-CI run measured automatic rollback **1.013 s**, kill-switch propagation **1.006 s**,
and the live overhead numbers in the exit table. `git diff --check` and the protected-file
diff check passed. No dependency changes or commits were made.

## Milestone 5 slice 5a — local distillation and deployment benchmark

Slice 5a adds teacher curation under `src/adaptive_llm/distillation/`, through the existing
dataset build and isolated evaluation paths. The approved adapter source and generated
distillation dataset receive separate explicit approvals. Only train inputs reach the teacher;
held-out targets and grouping remain the original source's. The manifest pins teacher version,
artifact, generation settings, judge, mix and optional encrypted safetensors. Student jobs reuse
the durable queue and CPU trainer for full or LoRA updates, deterministic checkpoints and signed
exports. Distilled promotion additionally requires held-out non-inferiority to the teacher and
a MAC-verified benchmark. Control migration 0011 stores content-free benchmark records.

| Milestone 5 local exit criterion | Measured result on 2026-09-24 |
| --- | --- |
| Smaller generated model | Student: **1 layer, 16 hidden units, 10,640 parameters**; tiny teacher fixture: **2 layers, 32 hidden units, 35,168 parameters**; shared **259-token** vocabulary |
| Teacher filtering and mix | Generation/hard-validation, judge, golden overlap and redaction failures are counted and excluded. Mix test: **1 teacher + 2 general + 1 abstention + 1 escalation**, actual fraction **.8**; eight held-out targets unchanged |
| Evidence and source governance | Evidence-bearing training inputs retain exact source blocks and provenance. Unapproved sources, unapproved teacher states, current-policy denial and source deletion block use |
| Soft targets | Real approved tiny teacher produces one encrypted safetensors payload with target IDs and **259-way** per-token log distributions; full student trains with KL; altered payloads fail authentication; fake teacher has no soft output |
| Deterministic full/LoRA training and resume | Both student modes produce identical full artifact digests across fresh four-step jobs and interruption/resume at step **2**; repeated KL jobs also match |
| CPU train → five suites → teacher comparison → benchmark → approve | Final full-student smoke: **3.432 s**, **500** optimizer steps, under the **90 s** limit; eight held-out items, quality delta **0**, 95% CI **[0, 0]** |
| Warm deployment benchmark at concurrency 4 | **16/16** successful outcomes per model; student p50/p95 **159.623 / 162.553 ms**, **24.915 requests/s**; fake foundation teacher **2.170 / 2.720 ms**, **1,668.579 requests/s** |
| Peak process RSS | Final smoke: student measurement **413,089,792 bytes**, teacher measurement **413,171,712 bytes**; training **398,393,344 bytes**. These are process high-water marks, not separate model allocations |
| Configured synthetic cost improvement | Student **1** versus teacher **16 USD micros/success**, **93.75%** reduction. Student prices **1/1**, fake teacher **1000/2000** micros per 1,000 input/output tokens; actual tokenizer counts used |
| Independent quality and efficiency gates | Four-step inferior student fails teacher comparison and benchmark; a faster inferior numerical fixture fails. Missing or tampered benchmarks block approval; current thresholds are checked again at promotion |
| Repeated benchmark | Identical request-mix digest, quality CI and integer costs across two runs; warm p95 within the documented factor-five scheduling tolerance |

These numbers demonstrate a local synthetic lifecycle. The passing smoke deliberately uses
one no-context target and sets its mix fraction to zero; separate tests cover the recorded
general/safety mix and evidence retention. Its teacher is the fake foundation, so the tensor
student is slower; it passes the configured cost criterion only. The registered real tiny
teacher is separately exercised for soft targets. Production cost, broad quality and the
specification's real-workload milestone exit remain unproven. The benchmark records this scope.
See the [distillation runbook](runbooks/distillation.md) for exact endpoints, reproduction,
approval sequencing, thresholds, loss and artifact formats.

No dependencies, specification files, `pyproject.toml` or `uv.lock` were changed. GPU,
downloads, external teachers, rollout changes and milestone 6 remain outside this task.

Verification used the exact prefix
`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache` on every uv/make command:

| Command after that prefix | Result |
| --- | --- |
| `make contracts` | Generated schemas and OpenAPI synchronized; schema consistency tests pass |
| `make check integration` | Initial pass: formatting, Ruff, strict mypy, 202 unit/contract passes and one GPU skip, 136 integration passes; subsequent added coverage included below |
| `make ci` | **Passed on the reviewer's host, 2026-09-24**, including the SBOM audit: **396** tests, **8** drills and **3** smoke paths (student approximately **3.5 s**). Formatting, Ruff, strict mypy on **71** source files; **205** unit/contract passes and one GPU skip; **138** integration passes |
| `uv run --locked pytest tests/security tests/load -s` | **44 passed**; event correlation **1,000/1,000**, **5,000** events; normal/live p95 overhead **2.032 / 2.374 ms** |
| `uv run --locked pytest tests/drills -m drill -s` | **8 passed**; rollback **1.025 s**, kill switch **1.002 s**; paired restore verifies control migration **0011** |
| `uv run --locked pytest -m smoke -s` | **3 passed**; student **3.432 s**, existing real LoRA **2.867 s**, fake lifecycle **0.203 s** |
| `make check-without-training` | **382 passed, 14 expected skips**, with all optional training imports blocked |
| `uv run --locked pytest tests/integration/test_distillation.py -q --tb=short` | **12 passed** |
| `uv run --locked pytest tests/unit/test_registry.py tests/integration/test_training.py -q --tb=short` | **30 passed** |
| `uv run --locked ruff format src tests` | Formatting applied, final check clean |
| `uv run --locked ruff check src tests` | Passed |
| `uv run --locked mypy src/adaptive_llm` | Passed |

`git diff --check` and `git diff --exit-code -- docs/spec pyproject.toml uv.lock` also passed.
Early development failures in new approval serialization and duplicate pytest module naming were
fixed; no gate was disabled. Final review also bound benchmark lookup to the signed candidate and
evaluation IDs and made student resource limits independent of the ordinary adapter backend.
No commits were made.

The pass-1 corrections admit any supported student geometry, require a strictly smaller parameter
count when the registered teacher's size is known, and record the verified student architecture
and parameter count. Unknown teacher size is an explicit manifest limitation. Benchmark success
counts completed requests without errors or hard validation failures; paired quality and evaluation
gates remain independent. New dataset approvals and model manifests use explicit version-2 MACs
over all immutable fields, while version-1 records retain historical verification rules. The
runbook documents the minimum of 16 accepted teacher examples for the default 0.2 fixture mix.

Pass-1 correction verification on 2026-09-24: `make contracts` regenerated the schemas;
`make check integration` passed; full `make ci` passed every gate, including the SBOM audit
(no known vulnerabilities). The suite now collects **405** tests: **207** unit/contract passes
and **1** expected GPU skip, **145** integration passes, **44** security/load passes and **8**
drills. All **3** smoke paths passed: student **3.995 s**, real LoRA **3.302 s**, fake lifecycle
**0.193 s**. Strict mypy passed on **71** source files. The dependency-absence gate passed
**390** tests with **15** expected skips; the focused distillation/MAC regression run passed
**24** tests. The earlier reviewer measurements above are retained.

## Milestone 6 slice 6a — optional local activation and pruning research

Both the Settings and routing JSON feature flags are required. Disabled applications import no
research service, mount no research routes and create no research tables or directories. The
research operator capability defaults empty. Jobs require an allowlisted verified tiny base,
approved calibration/evaluation datasets, current training policy for all tenants and a passed
evaluation of the exact unpruned base/adapter. Research code and artifacts are isolated under
`src/adaptive_llm/research/` and `<data_dir>/research/`. No migrations or dependencies were added.

Hooks capture only per-layer/head/channel norm moments, sparsity and a gradient-based
zero-ablation sensitivity estimate. One bounded batch/backward pass produces three model-sized
arrays, encrypted with study-bound AAD and a MAC-authenticated inventory. Structured pruning
rewrites config and full weights, then uses the existing full student trainer. The standard
loader verifies the `pruned-full-v1` candidate and its complete immutable lineage. Evaluation,
benchmark MACs, baseline locks and explicit registry approval remain mandatory. Research jobs
do not activate routes or deploy models.

| Milestone 6 local exit criterion | Measured result on 2026-09-24 |
| --- | --- |
| Flag and capability isolation | All disabled flag combinations expose no routes/tables/artifacts; fresh-process test confirms no research or optional-stack imports |
| Aggregate-only artifacts | Tiny shapes: layers **[2,1,4]**, heads **[2,4,4]**, MLP channels **[2,64,4]**; no token/example dimension; encrypted bytes and MAC tampering rejected |
| Physical removal and standard loading | One of four heads per layer removed; MLP width **64 → 48**; standard Llama config and safetensors load and generate. Separate test removes one complete layer |
| Distinct calibration/evaluation lineage | Passed with separate approved dataset versions: manifest training lineage and `training.completed.v1` name calibration; evaluation and registry admission bind to the declared evaluation version and reject calibration as evaluation evidence |
| Residual sensitivity and batch admission | Both tiny layers rank first when their contribution is independently zeroed; an oversized batch is refused before any hook or forward call |
| Parameter count | Heads plus channels: **35,168 → 32,096**. Head-only GQA compaction needs KV expansion and has no parameter saving; no unsupported saving is claimed |
| Study → prune → full tune → five suites → benchmark | Initial focused smoke **6.721 s**, below **120 s**; **500** optimizer steps, **8** independent held-out cases |
| Quality and safety on the narrow smoke | Five suites passed; candidate-minus-unpruned quality mean **0**, CI **[0,0]**, zero critical failures on the synthetic smoke safety fixture |
| Warm concurrency-four benchmark | Candidate p50/p95 **167.807 / 175.327 ms**, **23.259 requests/s**; unpruned p50/p95 **168.424 / 171.534 ms**, **23.368 requests/s**; **8/8** successes each |
| Independent-process peak RSS | Candidate **357,318,656 bytes**; unpruned **363,216,896 bytes**; **1.624%** reduction, including interpreter/library overhead |
| Hardware benefit required for approval | **Not demonstrated**: p95 regressed **2.211%** and RSS reduction was below **20%**. Signed benchmark returned **no_hardware_benefit** and promotion was refused |
| Specification milestone 6 exit / production acceptance | **Not met**: local lifecycle and refusal behavior are demonstrated; hardware benefit on a representative workload and production approval remain outstanding |

The smoke intentionally learns one no-context synthetic response; its small safety fixture is
not a general red-team acceptance claim. Numerical gate fixtures independently verify passing
latency/RSS boundaries and refusal on missing/tampered benchmarks or critical safety failures.
Research baselines and jobs use private in-memory persistence; measurements use separate fresh
processes, so the source and candidate do not share a peak-RSS high-water mark. Request-scoped
research jobs do not add a durable queue or restart/resume guarantee. See the
[activation research runbook](runbooks/activation-research.md) for exact commands and limitations.

Slice 6a verification used the installed training group and the exact prefix
`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache` on each command below:

| Command after that prefix | Result |
| --- | --- |
| `make contracts` | Passed; generated schemas and disabled-by-default OpenAPI synchronized |
| `make check integration` | Final pass: formatting, Ruff, strict mypy on **79** source files; **215** unit/contract passes, **1** existing GPU skip; **158** integration passes |
| `make ci` | The reviewer ran `make ci` on the host (Python 3.12.13, uv 0.12.17) and it passed, including the SBOM audit. |
| `uv run --locked pytest tests/security tests/load -s` | **44 passed**; **1,000/1,000** correlated interactions, **5,000** events; normal/live p95 overhead **2.084 / 2.421 ms** |
| `uv run --locked pytest tests/drills -m drill -s` | **8 passed**; rollback **1.014 s**, kill-switch propagation **1.006 s**; paired restore still verifies control migration **0011** |
| `uv run --locked pytest -m smoke -s` | **4 passed**; research **6.570 s**, student **3.563 s**, real LoRA **2.883 s**, fake lifecycle **0.200 s** |
| `make check-without-training` | **398 passed, 28 expected skips** with all four training imports blocked |
| `uv run --locked pytest tests/unit/test_research_boundary.py tests/unit/test_research_gates.py tests/integration/test_mac_versions.py -x -q --tb=short` | **14 passed**, including legacy MAC compatibility |
| `uv run --locked pytest tests/unit/test_research_boundary.py tests/unit/test_research_gates.py tests/integration/test_research.py -m smoke -s --tb=short` | **1 passed**, initial research smoke measured above |
| `uv run --locked pytest tests/integration/test_research.py -x -q --tb=short` | Initial **11 passed**; final expanded **13** research tests passed in the full integration gate |
| `uv run --locked ruff format src tests` | Formatting applied; final check clean |
| `uv run --locked ruff check src tests --fix` | Passed |
| `uv run --locked mypy src/adaptive_llm` | Passed |

The final smoke again passed quality and correctly refused hardware admission: candidate/base
p95 **172.455 / 174.794 ms** (only **1.338%** improvement), RSS **358,203,392 / 358,678,528 bytes**
(**0.132%** improvement). The table above retains the initial independent-process measurement.
The new tests also reject modified summary MACs, cross-study ciphertext despite a freshly signed
envelope, separately unapproved evaluation data, and tampered benchmark evidence. Identical
studies produce identical decrypted aggregate bytes. Head-only and layer-only exports generate
through the standard provider. All non-network CI gates ran; none was disabled.

`git diff --check` and `git diff --exit-code -- docs/spec pyproject.toml uv.lock` passed.
Development fixture setup failures were corrected before the full gates. No dependencies,
specification changes, commits or pushes were made. The reviewer ran `make ci` on the host
(Python 3.12.13, uv 0.12.17) and it passed, including the SBOM audit.

Review pass 1 separates calibration training lineage from evaluation evidence, corrects layer
statistics to use the residual contribution, and adds admission estimates for the padded batch.
Source verification during benchmark pre-checks and promotion does not instantiate models.
Queued or failed job IDs return a conflict; publication and summary checks authenticate dataset
metadata and current approval/policy without decrypting shards again. Old hook summaries remain
readable, but layer pruning requires the corrected hook version. Focused regressions passed:
**16 tests** for instrumentation and the research lifecycle, plus **4 tests** for job conflicts,
metadata rechecks and old layer statistics. The measured smoke numbers above are retained.

Review pass 1 validation used the same command prefix as above:

| Command after that prefix | Result |
| --- | --- |
| `make contracts` | Passed |
| `make check integration` | Formatting, Ruff and strict mypy passed; **218** unit/contract passes, **1** existing GPU skip; **162** integration passes. These gates also passed inside `make ci` |
| `uv run --locked pytest tests/security tests/load -s` | **44 passed** |
| `uv run --locked pytest tests/drills -m drill -s` | **8 passed** |
| `uv run --locked pytest -m smoke -s` | **4 passed**; research path **6.529 s**, quality passed and hardware admission correctly refused |
| `make check-without-training` | **398 passed, 35 expected skips** |

The protected-file diff and whitespace checks passed; no dependencies, commits or pushes were added.

## Specification milestone coverage 0–6

| Milestone | Implemented and locally verified | Not claimed / still required |
| --- | --- | --- |
| 0 — Discovery and decisions | Gap analysis, ADRs, threat/policy documentation and owner decision list | Confirmed real workload and signed security/data-owner design approvals |
| 1 — Instrumented foundation | Authenticated gateway, exact RAG evidence, encryption/redaction, retention/deletion, correlated events, retry/recovery and local load/drills | Staging load acceptance, real providers/transport/exporters and production identity |
| 2 — Dataset/evaluation platform | Approved reproducible encrypted datasets, decontamination/splits/lineage, five suites, bootstrap gates and locked local baseline | Representative seed data, human-reviewed real-provider baseline, external deletion lifecycle and asymmetric signing |
| 3 — Adapter specialist | Offline CPU LoRA, durable queue/resume/cancel, authenticated registry, real tensor generation and local promotion smoke | Real specialist quality, staging endpoint acceptance, GPU/distributed training and production approval |
| 4 — Router/shadow/canary | Calibrated logistic/OOD routing, validators/fallback/breakers, shadow/canary controls, costs and automatic rollback drills | Representative live calibration, production canary evidence and external infrastructure |
| 5 — Distilled student | Teacher filtering, separately approved student data, full/LoRA/KL training, paired teacher evaluation and signed benchmark gates | Material real-world hardware/cost improvement with representative quality and safety |
| 6 — Optional pruning research | Disabled-by-default studies, encrypted aggregate instrumentation, physically compact candidates, tuning, independent-process measurements and reports | Demonstrated hardware benefit without unacceptable representative quality/safety regressions; separate production approval |

Milestones 1–6 have local synthetic implementation coverage. Their production and real-workload
exit criteria remain distinct. The slice 6a hardware benchmark deliberately remains a failed
admission result when the measured improvement is insufficient; parameter counts do not replace
that requirement. No production deployment, GPU, downloads, real open-weight model experiment
or bandit work is included.
