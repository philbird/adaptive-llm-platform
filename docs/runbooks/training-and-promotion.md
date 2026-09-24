# Local training and promotion (slices 3a–3b)

The deterministic fake CPU trainer produces authenticated artifacts and proves approval,
lineage, checkpoint recovery, evaluation gates and registry transitions. It creates no tensors
and enables no specialist traffic. Slice 3b adds offline CPU LoRA through the injected `Trainer`
interface. The default backend remains `fake`, allowing the platform to run without the optional
`training` extra. Select `Settings(training_backend="lora")` or CLI `--backend lora` explicitly.
Only `training/lora.py` imports the optional stack, lazily; absent dependencies fail a real job
with `training_dependencies_unavailable`. The dependency and lock files are unchanged.

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
POST returns the new job in `queued` after approval/policy checks, without waiting for training.
The lifespan worker takes one durable job at a time, in creation order; GET
`/v1/training/jobs/{job_id}` polls. CLI `train` submits to the same queue and waits for completion.
Control migration 0008 stores the authenticated submitter separately from the specification;
the worker rechecks current dataset approval, artifacts and every tenant's policy before running.
Filesystem worker and job leases serialize local app/CLI processes. Retrying an active or
successful job returns its status; retrying a failed job requeues the same version and checkpoints.
The server also pins the trainer architecture in the queued job: fake workers skip real jobs and
vice versa. Start a worker with the matching backend to resume. Different specifications or trainer
architectures under the same id are refused. CLI terminal failures exit nonzero after printing the
content-free job status and a fixed error code.

POST `/v1/training/jobs/{job_id}/cancel` with the operator credential cancels queued work immediately
and requests cancellation of running work at the next optimizer-step boundary. Cancellation is
durable, idempotent and terminal, and emits one outcome per dataset tenant. Real training saves a
checkpoint before stopping. Graceful shutdown checkpoints and requeues; after a crash, `running`
jobs resume verified checkpoints or restart if none exist. Legacy slice-3a jobs have no stored
submitter and require authenticated resubmission. Distributed queues remain future work.

## Real CPU LoRA and local base models

Run the download-free tests with the already installed optional extra:

```sh
uv run --locked pytest tests/integration/test_lora.py -m accelerator_free -s
uv run --locked pytest tests/integration/test_lora.py -m smoke -s
make check-without-training
```

The fixture generates two random-initialized Llama layers, 32 hidden units and a 259-token byte
tokenizer under `tests/fixtures/base-models/generated/tiny/seed-17/`, then copies them into isolated
test directories. Nothing is downloaded. The passing smoke deliberately learns one no-context
response and evaluates eight held-out inputs plus small golden/safety/retrieval/performance suites.
Measured on 2026-09-24: **2.930 seconds** for real training → evaluation → approval → shadow,
with 47,500 tokens seen, admission estimate 844,032 bytes and process peak RSS 391,905,280 bytes.
It uses 500 CPU steps, rank 8, alpha 16, learning rate 0.005 and target modules
`q_proj`, `v_proj`, `lm_head`. This measures real tensor training and lifecycle mechanics; it is
not general-capability or safety acceptance. A separate test runs the normal five suites (including
the new catastrophic-forgetting and refusal golden items) against a four-step adapter and verifies
that failure blocks approval. Gate thresholds and promotion rules are unchanged.

For a local demo, first approve the dataset as above, then generate a base and update its spec:

```sh
uv run --locked python -c 'from pathlib import Path; from adaptive_llm.training.lora import generate_tiny_base; generate_tiny_base(Path(".local/evaluation-demo/base-models/tiny/seed-17"))'
uv run --locked python - <<'PY'
from pathlib import Path
from adaptive_llm.contracts import TrainingJobSpecification, uid
directory = Path('.local/evaluation-demo')
spec = TrainingJobSpecification.model_validate_json((directory / 'training.json').read_text())
spec = spec.model_copy(update={
    'job_id': uid(), 'base_model_id': 'tiny', 'base_model_revision': 'seed-17',
    'base_model_licence': 'CC0-1.0', 'tokenizer_id': 'tiny-byte-v1',
    'chat_template_version': 'tiny-chat-v1', 'steps': 20, 'batch_size': 1,
    'max_sequence_length': 128, 'checkpoint_every': 5,
})
(directory / 'lora-training.json').write_text(spec.model_dump_json(indent=2))
PY
make train BACKEND=lora SPEC=.local/evaluation-demo/lora-training.json DATA_DIR=.local/evaluation-demo POLICY=configs/policy/training-demo.json
```

The normal demo dataset/suites demand grounded answers; this tiny adapter is expected to fail
their gates. Use the smoke command for the passing synthetic approval/shadow demonstration.
For HTTP, construct the app with `Settings(data_dir=..., training_backend="lora", policy_path=...)`;
submit the same JSON and poll the returned job id.

Place a real, licensed open-weight base locally at
`<data_dir>/base-models/<base_model_id>/<base_model_revision>/`. Both identifiers are single path
components. Include `config.json`, safetensors weights (optionally sharded), local tokenizer assets,
its chat template, and `LICENSE.txt`. Supply `manifest.json` with the following metadata and a
SHA-256 hash of every other file, keyed by relative path:

```json
{
  "model_id": "local-base",
  "revision": "pinned-revision",
  "licence": "licence-identifier",
  "tokenizer_id": "pinned-tokenizer",
  "chat_template_version": "pinned-template",
  "context_limit": 4096,
  "files": {"config.json": "sha256-hex", "model.safetensors": "sha256-hex", "LICENSE.txt": "sha256-hex"}
}
```

Include all tokenizer/template files in `files` too. The job's five identity/licence fields must
match this manifest. The local operator is responsible for the original provenance and licence;
the inventory detects subsequent changes and is pinned into signed training artifacts. Loading
uses a private snapshot of verified bytes, `local_files_only=True`, safetensors only and no remote
code. Missing/extra/tampered files, symlinks, path traversal and changed base revisions fail closed.

Canonical training uses signed shard messages, source blocks with exact-version boundaries,
permitted tool results when present, and the approved target through the pinned chat template.
Current serving shards contain user/assistant roles only; training also understands trusted
system/tool roles without widening the serving request contract. Loss masks prompt and padding
tokens. Oldest prompt tokens are truncated first, preserving target tokens up to the configured
sequence bound. Sequence length cannot exceed the base manifest's context limit.

`Settings.training_memory_limit_bytes` defaults to 2,000,000,000. Before loading weights, admission
estimates `tensor parameters × precision bytes × 4 × 1.5`; the factor 4 budgets weights, gradients
and Adam moments, and 1.5 is a safety allowance. This is an estimate, not a hard process memory cap.
`Settings.training_time_limit_seconds` defaults to 300 per attempt; expiration checkpoints at a
step boundary and records `training_interrupted`. Time spent loading counts, but an individual load
or optimizer step is not forcibly killed. CPU work is serialized, single-threaded, seeded and uses
deterministic Torch operations. Precision, accumulation, batch size and LoRA hyperparameters come
from the specification. Peak RSS is the process high-water mark, not isolated worker allocation.

Real published artifacts contain only `adapter.safetensors`, `merged.safetensors`,
`adapter_config.json` and `training_report.json`. The report contains steps, examples, loss curve,
tokens seen, base manifest digest, binding and adapter digest. Wall time (`wall_seconds`) and peak
RSS (`peak_memory_bytes`) live only in `TrainingJob.resource_usage`; `artifact_bytes` counts the
published files. Each checkpoint includes adapter and Adam/RNG
safetensors, a numerical report, and a JSON digest inventory authenticated by the existing artifact
MAC key. Checkpoints bind the specification, dataset digest and base manifest digest; no pickle,
token ids, decoded examples or arbitrary notes are persisted. Complete checkpoint directories are
published by rename and recovered even if the process died before saving their database reference.
Before publishing a successful model, the orchestrator moves checkpoint directories into
`models/<registry_id>/.checkpoints/<version>/`, outside the hashed artifact inventory. Job checkpoint
references name directories there after success. Interrupted training retains its working
checkpoints; failed publication can leave checkpoints archived, and a retry restores them before
training. Partial archive/restore moves recover under the existing job lease.
The merged export is the complete merged model state, loaded with the pinned base configuration.
The unmerged export uses PEFT state-dict keys ([PEFT format](https://huggingface.co/docs/peft/developer_guides/checkpoint)).

Same CPU specification/data produces byte-identical published files, loss curves and full artifact
digests, including after resume. Evaluation reports bind that complete artifact digest.
The specialist loads verified adapter bytes onto the verified base, generates greedily within
the context/output budget, parses citations, and reports actual tokenizer token counts and
`stop`/`length`. Its local price list remains synthetic; no cost saving is claimed.

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

For the fake backend, files under `<data_dir>/models/<registry_id>/<version>/` are `adapter_config.json`,
`adapter_weights.bin` and `training_report.json`. The two `checkpoint-N/` directories containing
weights, a training report and `checkpoint.mac` move to the same separate checkpoint archive as
real checkpoints. The model manifest in the control database
records SHA-256 hashes of every file and a separate `model-artifact-v1` MAC over the complete
inventory and immutable lineage. Added/missing files, symlinks or tampering fail verification.
The specialist uses foundation behavior; its digest marker appears only in model metadata.
Local MACs are not asymmetric production signatures.

The manifest's adapter architecture is an extensible identifier. Trainers declare their
architecture, which registration checks against the manifest; the fake trainer declares
`deterministic-fake-adapter-v1`, and the real trainer declares `lora-peft-v1`. Registration and approval
currently require exactly one dataset lineage entry and otherwise return `single_dataset_required`.
Dataset mixing remains unsupported until its evaluation gates are implemented.

Interrupted work remains in `models/<registry_id>/.<version>.training/`. Retry the identical
job specification/id: authenticated checkpoints resume without repeating completed steps.
Changed configurations/code revisions or damaged checkpoints are refused. Completed versions
are never overwritten. Publication failures roll back registration/events and return exports to
the working directory; retries restore any archived checkpoints before resuming. A process crash
after final rename can leave an unregistered directory;
retries fail closed and require manual reconciliation. Reusing a successful job id returns the
original status after rechecking approval and policy.

For the fake backend, different job ids with the same specification produce byte-identical artifacts and stable
manifest fields. UUIDs, timestamps, UUID-containing paths and the MAC binding them differ.
Worker CPU and wall time are measured; fake peak memory is unavailable and null. Fake loss and step/example
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
