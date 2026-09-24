# Local calibrated routing, canary and automatic rollback (slice 4b)

Live routing is opt-in. The default gateway still serves foundation. Use a registered fake
or local LoRA specialist with a passing ordinary evaluation, and a separately approved router
with a passing `routing` evaluation. All keys, prices and measurements below are synthetic.
No external provider, streaming, bandit or production acceptance is implied.

## Build and approve counterfactual data

Start with the dataset and specialist lifecycle in [evaluation](evaluation.md) and
[training and promotion](training-and-promotion.md), then collect observations under a shadow
policy using [shadow and kill switch](shadow-and-kill-switch.md). At least one train group,
one calibration group and four test examples are needed. Subjects and document families remain
grouped; use distinct synthetic subjects for independent examples. A single connected document
family cannot be split across folds.

Post a dataset specification to the existing `/v1/datasets/builds` endpoint (or use
`make dataset-build SPEC=... POLICY=... DATA_DIR=...`):

```json
{
  "dataset_id": "synthetic-router",
  "purpose": "router_training",
  "tenant_ids": ["synthetic-a", "synthetic-b"],
  "source_dataset_id": "synthetic-evaluation",
  "source_dataset_version": "SOURCE_DATASET_VERSION",
  "source_window": {"start": "2026-09-24T00:00:00Z", "end": "2026-09-25T00:00:00Z"},
  "eligibility_policy_version": "synthetic-dataset-policy-1",
  "seed": 23
}
```

Replace the source version, window, tenant grants and policy version with the reviewed local
values. Current training permission, retention and deletion checks apply before publication.
The source reference is immutable lineage. Eligible source test interactions and shadow
opportunities in the window contribute one row each, including opportunities dropped by the
queue or kill switch. Every observed specialist has a column; absent observations are null.
Signed held-out per-item observations take precedence over shadow scores. Older scores without
per-item validation/cost/latency retain those measurements as unknown and cannot certify suitability.

Rows contain task/language/risk, chunk count/top score/index id, input tokens, context flag,
deployment ids, observed quality/validation/cost/latency and lineage identifiers. There are no
messages, targets, retrieved chunks, provider bodies or subject strings. The existing AES-GCM
tenant/split shards, MAC-verified manifest, pending approval, data card, build event and deletion
recheck apply. Feature-identical interactions are retained; text deduplication is inapplicable.
The default grouping split is 80/10/10; explicit `time_split` cutoffs are also supported.

Approve the returned dataset with the existing operator endpoint:

```sh
curl -X POST http://127.0.0.1:8000/v1/datasets/synthetic-router/versions/ROUTING_DATASET_VERSION/approval \
  -H 'Authorization: Bearer synthetic-operator-key' -H 'Content-Type: application/json' \
  -d '{"reason":"SYNTHETIC reviewed numeric router dataset"}'
```

## Train, evaluate and promote the router

Submit to `/v1/training/jobs` and poll the returned job id, or run the existing training CLI:

```json
{
  "job_type": "router",
  "registry_id": "synthetic-router",
  "dataset_id": "synthetic-router",
  "dataset_version": "ROUTING_DATASET_VERSION",
  "seed": 23
}
```

The normal durable queue, authenticated submitter, current-policy checks, cancellation and
immutable signed export apply. Router jobs always select the plain-Python backend, regardless
of the adapter backend. Interrupted jobs deterministically recompute the small model; no torch
or new dependency is required. Repeated jobs on the same dataset have identical artifact digests.

`router.json` uses six categorical groups: task, language, risk, token bin (edges 32, 128,
512, 2048, 8192), context flag and retrieval-score bin (edges .25, .5, .75). Each group has an
unknown bucket. For each specialist, 600 full-batch gradient steps fit logistic suitability on
train, then 600 steps fit an intercept/slope sigmoid on validation logits (Platt calibration).
Suitability labels require observed validation, quality ≥ .9 and quality within .02 of foundation;
missing coverage labels abstention. Confidence is validation reliability, `1 − ECE`, multiplied
by `min(1, observed_train_examples / 4)`. Test rows never fit either model.

OOD is RMS distance from the train centroid after coordinate standardisation (standard-deviation
floor .25), transformed as `distance / (1 + distance)`. New categories activate unknown buckets.
This is a conservative local novelty heuristic; correlated categorical features and sparse
coverage can cause abstention. The serving task classifier still derives `general` versus
`question_answering` from the RAG flag, language `en`, and risk `medium`; semantic task, language
and risk inference is not added by this slice.

Post a separate evaluation request:

```json
{
  "candidate_deployment_id": "ROUTER_MODEL_VERSION",
  "baseline_deployment_id": null,
  "dataset_id": "synthetic-router",
  "dataset_version": "ROUTING_DATASET_VERSION",
  "suites": ["routing"],
  "minimum_sample_size": 4
}
```

The routing suite reports false-specialist selections divided by all specialist selections,
unnecessary-foundation selections divided by rows with a suitable specialist, ten-bin ECE and
OOD detection among naturally unseen test categories. Zero `ood_samples` means unmeasured OOD
coverage. Router approval requires the configured minimum, false-specialist rate ≤ the target
in `configs/evaluation/initial-targets.json` (.02), and ECE ≤ .1. Empty/incomplete folds fail.
This evaluation has no generative foundation baseline; signed observations provide its labels.
The ordinary five-suite foundation lock and specialist promotion checks remain unchanged.

Promote the router to `approved`, supplying the routing evaluation id and operator reason.
For a router, this explicit registry approval is the promoted version usable by a route policy;
the router itself does not generate responses. Dataset version/digest and artifact digest must
match the report when approval is checked. The specialist still follows shadow → canary → production.

## Activate live canary and expand

Create and activate a route policy using the existing operator API:

```json
{
  "eligible_specialist_versions": ["SPECIALIST_VERSION"],
  "foundation_fallback": "fake-foundation-local-1",
  "router_version": "ROUTER_MODEL_VERSION",
  "live_specialists_allowed": true,
  "shadow_enabled": true,
  "tenant_enabled": {"synthetic-a": true},
  "task_enabled": {"general": true},
  "quality_threshold": 0.9,
  "router_confidence_threshold": 0.85,
  "ood_threshold_max": 0.15,
  "canary": {"traffic_fraction": 0.05, "tenant_allowlist": []},
  "rollback": {
    "interval_seconds": 1,
    "window_seconds": 300,
    "minimum_samples": 4,
    "non_inferiority_margin": 0.02,
    "validation_failure_rate_max": 0,
    "error_rate_max": 0,
    "p95_latency_ms_max": 5000,
    "cost_ratio_max": 1
  }
}
```

For the internal step, fill `tenant_allowlist` with `Keyring.pseudonym(tenant_id, "canary-tenant")`
values derived using the deployment's configured keyring. An empty list admits all policy-enabled
tenants. Both the allowlist and fraction apply. Assignment is
`int(SHA256(interaction_id + ":" + policy_id)) / 2**256 < traffic_fraction`. Replays reuse the
original response and create no new observations. A new policy id creates a new assignment.

Continue collecting shadow comparisons under this active policy while the specialist is still
`shadow`; it cannot answer users. Promotion to `canary` requires this policy, fraction ≤ .05,
and passing non-inferior shadow comparisons for this exact specialist, overall and in each
observed critical segment, with at least the configured minimum. Promotion to `production`
requires an operator note and a passing canary report. Unmeasured or unmatched segments block
promotion. Once production is approved, a new immutable policy can increase the fraction to 1.
Automatic disablement persists across policy changes, so activation cannot bypass rollback.

Hard checks cover trusted tenant/task controls, registry state, residency, text modality,
context capacity (input plus requested output), breakers and budget. This ingress has no tools;
its tool allowlist is empty and any generated tool call is a hard safety failure. The cheapest
qualifying specialist is selected, with foundation retained as the fallback. Every considered
candidate records suitability, confidence, OOD or its hard rejection reason. `foundation` mode
always uses foundation; `specialist` requests cannot bypass policy, canary or admission checks.
Specialists priced above foundation receive the internal candidate/fallback reason
`not_cheapest`. The client error `cost_limit_exceeded` remains reserved for the request budget.
Model token prices are signed in the training specification/manifest; defaults preserve the
existing synthetic prices. Lower specialist quotes can fit a budget that excludes foundation.

Live admission manifests refresh with the policy store's one-second snapshot cadence, including
explicit control invalidations. Verified routers are cached by version and artifact digest in
a bounded cache; a changed registry state or digest evicts the entry. Cold verification rechecks
the registry before publishing a cached router. Tenant membership and hard constraints still
run for every request, and the specialist loader retains its registry eligibility checks.
Planner exceptions clear its caches, serve foundation with `policy_uncertainty`, and increment
the content-free `live_planner_failures` counter exposed beside `kill_switch` on `/healthz`.

## Reports and rollback

```sh
curl --get http://127.0.0.1:8000/v1/canary/reports \
  -H 'Authorization: Bearer synthetic-operator-key' \
  --data-urlencode 'policy=POLICY_ID' \
  --data-urlencode 'since=2026-09-24T00:00:00Z'
```

Reports have an explicit end time, overall/per-specialist aggregates and task+risk, language,
context and keyed-tenant segments. The live quality proxy is the same chunk-overlap judge used
for shadow. Pairing matches chronological observations within tenant, task, risk, language and
context bins; it is observational matching, not repeated evaluation of identical prompts.
The seeded paired bootstrap uses 2,000 resamples. Unmatched items do not enter its CI and the
paired sample count is reported. Serving/report records contain no text.

`specialist` and `foundation` are attempt cohorts: a specialist interaction includes its failed
attempts and foundation fallback, even if foundation delivered the answer. `specialist_served`
and `foundation_served` additionally show actual response cohorts, with their own
`served_cost_delta_micros` paired bootstrap and `served_cost_reduction_fraction`.
`total_cost_micros` sums all
live attempts; `shadow_cost_micros` is separate. Per-success costs divide all cohort cost by its
successful outcomes and round up to integer USD micros for display. Reduction uses unrounded
totals. Bootstrap cost samples allocate failed-outcome costs across successes within each bin.
Unknown provider charges are unavailable; local token accounting is synthetic and excludes
unavailable retrieval/infrastructure costs. No successes produces null per-success cost.

The lifespan monitor checks every configured interval. After minimum volume it disables a
canary/production specialist on any validation, endpoint-error, latency or cost-ratio breach,
including a breach in a critical segment. A critical safety failure has no sample minimum.
Tool-allowlist and forbidden-text failures are marked critical; injected validators may mark
additional checks. No automatic promotion occurs. Control migration 0010 stores live numbers,
per-environment disabled versions and the complete triggering aggregate/segment names.
The measurement, audit and `deployment.changed.v1` event commit atomically. The next process
observes the persistent control within the existing one-second cache interval.

After reviewing the cause, re-enable with a nonblank operator reason:

```sh
curl -X DELETE http://127.0.0.1:8000/v1/route-policies/kill-switch \
  -H 'Authorization: Bearer synthetic-operator-key' -H 'Content-Type: application/json' \
  -d '{"specialist_version":"SPECIALIST_VERSION","reason":"SYNTHETIC reviewed recovery"}'
```

The monitor retains the offending window; re-enabling while it still breaches can disable the
version again. Review the report and wait for that window to clear before recovery. Global,
tenant and task switches remain independently effective. Already running generations finish
at their cooperative boundary; the drill measures the next request. Control-store failure
abstains to foundation. Tenant graph and control observations use separate transactions;
an observation write failure reduces coverage and cannot create a passing report without samples.

## Reproduce the local exit checks

```sh
uv run --locked pytest tests/integration/test_live_routing.py -s
uv run --locked pytest tests/drills/test_canary_rollback.py tests/drills/test_routing_kill_switch.py -m drill -s
make contracts
make check integration
make ci
```

The fixture runs real public dataset, training, evaluation and promotion endpoints with a
registered fake specialist, sixty shadow observations and 40/10/10 router folds. Selection
drills choose interaction ids meeting the real hash rule; a separate 10,000-id test checks the
5% allocation. See [status](../status.md) for the measured exit table and gate results.
In the restricted sandbox prefix commands with
`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache`. The SBOM audit requires network;
if it fails there, run the remaining security/load, drill, smoke and dependency-absence gates
explicitly as described in the shadow runbook.
