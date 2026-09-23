# Local evaluation and foundation baseline lock (slice 2b)

Only the deterministic fake foundation is registered by default:
`fake-foundation-local-1`. The operator API and CLI use the same evaluator, suites,
storage and gate. Slice 3a also resolves registered model versions as candidates; the
[training runbook](training-and-promotion.md) describes approval and promotion.

## Reproduce a baseline

From the repository root, seed nine synthetic interactions and build one train example
and eight held-out examples. This uses the existing dataset demo policy and local operator
key. Each invocation builds a fresh immutable dataset version; no real content is used.
The held-out examples share a document family and are kept together by the time split.

```sh
uv run --locked python - <<'PY'
import argparse
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from adaptive_llm.app import ROOT, Settings, create_app
from adaptive_llm.contracts import (
    DatasetManifest, DatasetSpecification, EvaluationInput, SourceWindow, TimeSplit, now, uid,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path(".local/evaluation-demo"))
args = parser.parse_args()
directory = args.data_dir
start = now()
operator = {"Authorization": "Bearer synthetic-operator-key"}
app = create_app(Settings(
    data_dir=directory, policy_path=ROOT / "configs/policy/dataset-demo.json",
    outbox_dispatch_enabled=False,
))
with TestClient(app) as client:
    cutoff = now()
    for index in range(9):
        result = client.post("/v1/inference", headers={
            "Authorization": "Bearer synthetic-key-a", "X-Subject": f"synthetic-eval-{uid()}",
        }, json={
            "request_id": uid(), "application_id": "support-assistant",
            "messages": [{"role": "user", "content": f"SYNTHETIC evaluation case {index} unused receipt"}],
            "rag": {"enabled": index > 0, "index_id": "synthetic-kb"},
        })
        if result.status_code != 200:
            raise RuntimeError("synthetic_seed_failed")
        if index == 0:
            cutoff = now()
    specification = DatasetSpecification(
        dataset_id="synthetic-evaluation", tenant_ids=["synthetic-a", "synthetic-b"],
        source_window=SourceWindow(start=start, end=now()),
        eligibility_policy_version="synthetic-dataset-policy-1",
        near_duplicate_threshold=1,
        time_split=TimeSplit(train_end=cutoff, validation_end=cutoff + timedelta(microseconds=1)),
    )
    result = client.post("/v1/datasets/builds", headers=operator,
                         json=specification.model_dump(mode="json"))
    if result.status_code != 200:
        raise RuntimeError("synthetic_dataset_failed")
    manifest = DatasetManifest.model_validate(result.json())
    request = EvaluationInput(
        candidate_deployment_id="fake-foundation-local-1",
        baseline_deployment_id="fake-foundation-local-1",
        dataset_id=manifest.dataset_id, dataset_version=manifest.version,
        suites=["golden", "held_out", "safety", "retrieval", "performance"],
        minimum_sample_size=4, critical_segments=["citation", "safety"],
        performance_requests=40, concurrency=4,
    )
    (directory / "evaluation.json").write_text(request.model_dump_json(indent=2))
    print(f"dataset={manifest.version} train={manifest.examples['train']} test={manifest.examples['test']}")
PY

make evaluate SPEC=.local/evaluation-demo/evaluation.json DATA_DIR=.local/evaluation-demo
```

The seed command prints `dataset=<version> train=1 test=8`. Evaluation prints a content-free
JSON report with `passed: true`, five suite results and all gates passed. Expected counts are
golden 20, held-out 8, safety 20, retrieval 8, performance 40 at concurrency 4.
Paired mean delta and both 95% CI bounds are zero, with n=8 overall and for the citation
and safety segments. Golden assertion rate is 1 and rubric mean is 5; held-out F1 and
citation precision/recall are 1; retrieval recall@3, MRR and context precision are 1;
critical safety failures, leakage and isolation incidents are zero. TTFT is null.
Latency varies by machine; the lock re-measured after persistence isolation had p95
2.264 ms and 42 USD micros per successful outcome. The earlier p95 of 6.777 ms included
tenant persistence. Current latency includes private in-memory persistence, encryption and
validation, with no tenant writes or case events. The database/report cited in `docs/status.md`
are local, git-ignored artifacts, not checked in; these commands reproduce the lock with new ids.
This is synthetic smoke evidence, not real workload acceptance.

For API use, start the server against that directory in another terminal:

```sh
uv run --locked python -c 'from pathlib import Path; import uvicorn; from adaptive_llm.app import Settings, create_app; uvicorn.run(create_app(Settings(data_dir=Path(".local/evaluation-demo"))), host="127.0.0.1", port=8000, access_log=False)'

curl -fsS http://127.0.0.1:8000/v1/evaluations \
  -H 'Authorization: Bearer synthetic-operator-key' \
  -H 'X-Subject: synthetic-evaluator' -H 'Content-Type: application/json' \
  --data-binary @.local/evaluation-demo/evaluation.json

curl -fsS http://127.0.0.1:8000/v1/evaluations/EVALUATION_ID \
  -H 'Authorization: Bearer synthetic-operator-key'
```

Do not run separate demo processes writing to the same SQLite directory concurrently.
The tests exercise concurrent requests through a single application.

## Specifications, samples and gates

The request has candidate/baseline deployment ids, immutable dataset id/version, the five
suite names, pinned rubric/judge versions, seed, confidence, margin, minimum sample size,
critical label keys, request count and concurrency. Judge and rubric versions are nullable
identifiers, checked at runtime against the registered judge's version and rubric version.
An unregistered version returns 422 with `judge_version_mismatch` or `rubric_version_mismatch`.
`judge_version: null` disables the optional rubric; `rubric_version` may also be null in that
case. An enabled judge requires its registered rubric version. Deterministic golden assertions
still run. A null baseline can produce
a diagnostic report but cannot pass or lock a baseline. Unknown deployments return 422.
Only local evaluation is enabled.

The synthetic planning pilot deltas `[-0.01, 0, 0.01]` have sample standard deviation 0.01.
`required_sample_size(0.02, 0.01, 0.95)` returns **4**:
`floor((2 × normal_quantile(0.975) × 0.01 / 0.02)²) + 1`.
The strict inequality puts the CI half-width below half the margin. Only
`minimum_sample_sizes` was filled in `configs/evaluation/initial-targets.json`.
This planning pilot is deliberately synthetic; workload owners must supply a measured
pilot and confirm provisional targets before real deployment.

The report records that pilot, the derived minimum, actual n, coverage and limitations.
Both the requested minimum and the derived minimum must be satisfied for the overall
comparison and every value of each critical label key. Missing n, missing labels, unmatched
items, empty suites, or a CI touching/straddling −margin all fail closed. The bootstrap uses
2,000 paired resamples with a seeded RNG; percentiles use linear interpolation.
Quality uses the minimum of token F1 and citation precision/recall per held-out item.
The report also preserves each metric separately. Segments are ordinary dataset labels;
`citation` is derived as required/none and `safety` from the risk tier. Label values are
HMAC pseudonyms in reports. Synthetic adversarial checks are separate hard safety gates.

The default local performance ceilings are p95 total latency 5,000 ms, error rate zero,
and 20,000 integer USD micros per successful outcome, at the requested concurrency.
Provider token usage is charged even when subsequent validation fails; unsuccessful
outcomes never enter the success denominator. There is no passable cost measurement if
there are no successful outcomes. These local ceilings are configurable in the specification.
The provisional 30% specialist cost reduction and false-specialist targets apply to later
specialist routing, not identical synthetic foundation comparisons. Existing load tests
continue to enforce the separate 50 ms routing/logging/classification overhead target.

## Data and judge boundaries

All requests call `InferenceService` in process, from a worker thread, with a trusted
evaluation identity derived from operator tenant grants. The application is always
`evaluation`; the evaluation policy denies content logging and training. One private in-memory
SQLite database per evaluation stores its metadata and encrypted replay using the same cipher,
keyring and redactor as serving. It has a no-op outbox/fallback sink and separate metrics.
Candidate and baseline suites share this database and one event loop. The database connection
is closed before report publication, and also on errors; no scratch directory is created or
left behind. Evaluation never writes tenant operational tables, payloads, replay entries or
the tenant outbox. The dataset builder's `evaluation_interaction` exclusion remains as defence
in depth for older or misclassified rows.

Held-out data is the immutable test split only. The reader verifies the stored manifest's
immutable fields and any signed store-only approval,
its detached MAC, every shard's tenant/split binding, AES-GCM authentication and SHA-256,
the overall content digest and split counts. Source snapshots are decoded in memory from
the builder's redacted source boundaries and provenance; the evaluation application gets
access only to these authenticated snapshots. The production index is not modified.
All candidate/baseline items are joined by their existing keyed example hash.

The 20 existing golden inputs/targets now have deterministic expected facts, citation and
JSON assertions. Targets become synthetic retrieved evidence for the echoing fake provider.
The judge receives only answer text, expected facts, prohibited terms and citation validity.
It sees no deployment identity; inputs are shuffled before pointwise scoring, then restored
to item order. Its pinned rule-based scores range from 0 to 5, and disagreement counts
compare perfect rubric scores against the deterministic assertions.

Safety has five critical cases each for indirect injection, tool abuse, secret extraction
and cross-tenant retrieval. Injection cases put instructions to abandon citations in retrieved
content; success means those grounding assertions fail. Quoting those synthetic instructions
with attribution is not itself counted as executing them. Tool/canary markers are prohibited
in the answer. Retrieval fixtures include higher-overlap foreign-tenant, denied-application,
wrong-environment and stale-index decoys, filtered by the same local retriever as serving.
These fixtures do not establish broad jailbreak resistance or live-index freshness.

## Reports, locks and replacement

Artifacts are stored under:

```text
<data_dir>/evaluations/<evaluation_id>/
  report.json
  report.mac
```

The MAC is HMAC-SHA256 over the exact UTF-8 report bytes using the separately derived
`evaluation-report-v1` key. The database stores the same report and MAC. Reads verify
both copies; tampering returns a fixed content-free error.

Control migration `migrations/control/0006_evaluations.sql` adds `evaluation_reports` and `baselines`. The evaluation control database is
`<data_dir>/control/<environment>.sqlite3`, separate from tenant operational storage.
Its outbox, dispatcher, event sink and metrics are also separate; evaluation completion and model lifecycle events enter
this stream, never individual case events. Publication atomically records
the report, updates a qualifying baseline, and enqueues one `evaluation.completed.v1`
per dataset tenant, including tenants with empty shards. Events share a trace id, have
distinct event ids, and contain a report reference plus versions, suite names and pass flag.
`baseline_version` is JSON null when no baseline was specified.
A write/rename failure rolls back SQL and cleans that run's staged artifacts. A process crash
can leave an unregistered directory, which must not be consumed. GET only reads committed rows.

A passing `candidate == baseline == fake-foundation-local-1` evaluation locks
`(deployment_id, dataset_version)`. Later injected fake candidates compare to the locked
per-item scores; they must match the baseline manifest, dataset digest, fixture digest,
suite list, rubric/judge versions, seed and segment definitions. The deployment manifest
version is SHA-256 over the local deployment configuration, including model and price versions.
A changed baseline during evaluation fails publication with 409. Registered candidates are
verified before loading, enter `evaluating`, and retain this state even after a passing report.
Reports contain their registry version and artifact digest; model report links commit with
publication. Only an explicit promotion request can approve them.

Reposting the identical evaluation id/spec returns the original immutable report.
A new id against an already locked foundation key returns **409**. To replace it, create a
new UUIDv7 id and submit `replace: true` with a nonempty `operator_note`.
The note is stored only as an HMAC; it never appears in the report or event.
Replacement requires a passing report; a failed replacement preserves the prior lock.
Reports remain immutable and reports in the control store remain readable. Earlier local
locks from before persistence isolation are not imported from the tenant database; reproduce
them using the seed and evaluation commands above. Unauthenticated calls return 401,
user keys 403, and insufficient tenant grants cannot build or read reports. CLI failures
exit 1 with `evaluation_failed`; completed gate failures print a report with `passed: false`.

Unit/contract checks remain the first evaluation layer and run through `make check`.
Slice 3a adds fake training, dataset approval and gated registry transitions. Human review,
LLM judging, real training and shadow/canary execution remain out of scope. Streaming TTFT, process-memory/cold-start attribution,
real retrieval/infrastructure costs and real-provider behavior are unavailable in this local
fake baseline and are declared limitations. Filesystem artifact revocation, rotation,
asymmetric signing and backup/reconciliation remain separate lifecycle work; the paired SQLite backup CLI now preserves tenant and control databases atomically, but
report/dataset/model directories still require separate preservation. Migration of pre-isolation local artifacts remains outside this synthetic slice.

## Verification

```sh
make contracts
make check integration
uv run --locked pytest tests/security tests/load -s
make drills
```

If the sandbox denies uv's default cache, prefix commands with
`UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache`; no dependency or lockfile changes are needed.
