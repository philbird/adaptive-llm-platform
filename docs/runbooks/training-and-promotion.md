# Local training and promotion (slice 3a)

The deterministic fake CPU trainer produces authenticated artifacts and proves approval,
lineage, checkpoint recovery, evaluation gates and registry transitions. It creates no tensors
and enables no specialist traffic. Slice 3b adds real LoRA through the injected `Trainer`
interface and an optional dependency group.

First run the seed and baseline commands in [evaluation.md](evaluation.md), using
`.local/evaluation-demo`, then stop that demo server. Its dataset includes both synthetic
tenants, including empty tenant-B shards. Training requires current permission for **every**
manifest tenant. The explicitly selected `configs/policy/training-demo.json` grants both tenants
training. Default serving and dataset demo policies remain unchanged.

Approve the signed dataset and write the training specification:

```sh
uv run --locked python - <<'PY'
from pathlib import Path
from fastapi.testclient import TestClient
from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import EvaluationInput, TrainingJobSpecification

directory = Path('.local/evaluation-demo')
evaluation = EvaluationInput.model_validate_json((directory / 'evaluation.json').read_text())
app = create_app(Settings(data_dir=directory, outbox_dispatch_enabled=False))
with TestClient(app) as client:
    response = client.post(
        f'/v1/datasets/{evaluation.dataset_id}/versions/{evaluation.dataset_version}/approval',
        headers={'Authorization': 'Bearer synthetic-operator-key', 'X-Subject': 'synthetic-reviewer'},
        json={'reason': 'SYNTHETIC CPU lifecycle demonstration'},
    )
    assert response.status_code == 200
spec = TrainingJobSpecification(dataset_id=evaluation.dataset_id,
                                dataset_version=evaluation.dataset_version)
(directory / 'training.json').write_text(spec.model_dump_json(indent=2))
PY

make train SPEC=.local/evaluation-demo/training.json DATA_DIR=.local/evaluation-demo POLICY=configs/policy/training-demo.json
make models DATA_DIR=.local/evaluation-demo
```

The job response contains `state="succeeded"`, a new `model_version`, checkpoint refs,
artifact digest, resource counters and server-filled Git revision. Copy the version into
`MODEL_VERSION` below. For HTTP, start the app with the training demo policy and POST the
same specification to `/v1/training/jobs`; GET `/v1/training/jobs/{job_id}` reads durable status.
POST waits on a worker thread, keeping inference/status available. A filesystem lease rejects
concurrent runs of the same job id. Distributed queues and cancellation APIs remain future work.
Slice 3b will change submission to return `queued` immediately, run real LoRA in a background
worker, and use `GET /v1/training/jobs/{job_id}` for status polling.

```sh
uv run --locked python - <<'PY'
from pathlib import Path
from adaptive_llm.contracts import EvaluationInput, uid
directory = Path('.local/evaluation-demo')
baseline = EvaluationInput.model_validate_json((directory / 'evaluation.json').read_text())
candidate = baseline.model_copy(update={
    'evaluation_id': uid(), 'candidate_deployment_id': 'MODEL_VERSION',
})
(directory / 'candidate.json').write_text(candidate.model_dump_json(indent=2))
PY

make evaluate SPEC=.local/evaluation-demo/candidate.json DATA_DIR=.local/evaluation-demo
make promote MODEL=MODEL_VERSION TO=approved EVALUATION_ID=EVALUATION_ID NOTE='SYNTHETIC reviewed five suites' DATA_DIR=.local/evaluation-demo
make promote MODEL=MODEL_VERSION TO=shadow NOTE='SYNTHETIC shadow approval' DATA_DIR=.local/evaluation-demo
```

Evaluation verifies artifacts, moves candidates to `evaluating`, and runs all five suites
against the locked foundation baseline. Publication links report ids and actual pass/fail to
the manifest in the same control transaction. Passing does not approve a model automatically.
Reports identify the registry version, artifact digest and provider model version containing
that digest. Substitute the returned evaluation id for `EVALUATION_ID` above.

Promotion requires an operator credential and nonblank `reason` (CLI `NOTE`). Its optional
body `actor` must match the authenticated pseudonym. Without `X-Subject`, the operator key
supplies a stable pseudonymous actor. Reasons are readable audit text, bounded to 2,000
characters, in stored approvals, model history, transition records and deployment events.
Operators must not paste user content into reasons. Evaluation `operator_note` remains hashed.
The HTTP body is `{"model_version":"...","target_state":"approved",
"reason":"...","evaluation_id":"..."}` at `/v1/models/{model_version}/promotion-requests`.

Ordinary edges are `candidate → evaluating → approved → shadow → canary → production → deprecated`.
Every non-revoked state can reach `revoked` with a reason; revoked is terminal. Approval checks
the passed report's exact model, artifact and dataset version against the **current** baseline
lock. Replacing that lock makes old reports insufficient for new approval requests. Identical
promotion retries with the same actor, note and evaluation id emit nothing further. Each
actual transition, including initial candidate registration, emits one `deployment.changed.v1`
per dataset tenant, including empty shards. Training terminal outcomes similarly emit
`training.completed.v1` per tenant; failed attempts retain fixed failure codes and checkpoints.

These are control states only: shadow execution, canary routing and production specialist
serving remain milestone 4. Promoting a second version under the same registry id to production
atomically deprecates the incumbent and preserves it as the previous version. The registry id
is also the local deployment id.

```sh
make rollback DEPLOYMENT=synthetic-specialist NOTE='SYNTHETIC operator rollback' DATA_DIR=.local/evaluation-demo
```

Rollback requires the deployment's recorded previous version to be deprecated, to have an
`evaluating → approved` transition in its own lifecycle history, and to have verified artifacts.
It does not re-evaluate approval or consult the current baseline lock; replacing a baseline
cannot disable emergency rollback. It deprecates the current version and restores the previous
version to production in one transaction, emitting two transitions per tenant. This is the
explicit recovery exception to forward-only edges. Missing, revoked, unapproved or tampered versions
are refused. HTTP uses `/v1/deployments/{deployment_id}/rollback` with `{"reason":"..."}`.
Every API enforces operator tenant grants. `make models` prints only registry ids, versions,
states and evaluation ids. CLI failures contain fixed codes.

## Files, signatures and recovery

Approval updates only the stored dataset manifest approval, with a MAC bound to the full
approved record. Immutable manifest/MAC and encrypted shard files remain unchanged. Training
and evaluation authenticate both records, every tenant/split binding, shard encryption,
plaintext hash, count and overall digest. Ordered train rows are HMACed before deriving fake
weight bytes. No examples or arbitrary notes appear in model files.

Files under `<data_dir>/models/<registry_id>/<version>/` are `adapter_config.json`,
`adapter_weights.bin`, `training_report.json`, and two `checkpoint-N/` directories containing
weights, a training report and `checkpoint.mac`. The model manifest in the control database
records SHA-256 hashes of every file and a separate `model-artifact-v1` MAC over the complete
inventory and immutable lineage. Added/missing files, symlinks or tampering fail verification.
The specialist uses foundation behavior; its digest marker appears only in model metadata.
Local MACs are not asymmetric production signatures.

The manifest's adapter architecture is an extensible identifier. Trainers declare their
architecture, which registration checks against the manifest; the fake trainer declares
`deterministic-fake-adapter-v1`, and slice 3b can declare `lora-peft-v1`. Registration and approval
currently require exactly one dataset lineage entry and otherwise return `single_dataset_required`.
Dataset mixing remains unsupported until its evaluation gates are implemented.

Interrupted work remains in `models/<registry_id>/.<version>.training/`. Retry the identical
job specification/id: authenticated checkpoints resume without repeating completed steps.
Changed configurations/code revisions or damaged checkpoints are refused. Completed versions
are never overwritten. Publication failures roll back registration/events and return files to
the working directory. A process crash after final rename can leave an unregistered directory;
retries fail closed and require manual reconciliation. Reusing a successful job id returns the
original status after rechecking approval and policy.

Different job ids with the same specification produce byte-identical artifacts and stable
manifest fields. UUIDs, timestamps, UUID-containing paths and the MAC binding them differ.
Worker CPU time is measured; peak memory is unavailable and null. Loss and step/example
counts are synthetic, not evidence of learned model quality.

The shared control database is `<data_dir>/control/<environment>.sqlite3`, migrated from
`migrations/control/`. It holds jobs/models/history/deployments/evaluations/baselines/outbox.
Tenant storage holds interactions, content and datasets. Startup moves existing slice-2b
`evaluations/control/` directories, preserving reports/locks and removing empty inherited tenant
tables. Ambiguous old/new directories and populated misplaced tables fail closed.

[Backup/restore](backup-restore.md) covers the databases atomically. Filesystem datasets,
models, reports and keys must be preserved separately. External deletion/revocation, file key
rotation, distributed recovery and production approval workflows remain outside this slice.

```sh
make contracts
make check integration
uv run --locked pytest tests/security tests/load -s
make drills
uv run --locked pytest tests/integration/test_training.py -m smoke -s
```

Prefix with `UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache` if the default uv cache is denied.
