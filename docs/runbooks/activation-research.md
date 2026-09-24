# Local activation analysis and structured pruning (slice 6a)

This optional research path runs only with both `Settings.pruning_research_enabled=True` and
`pruning_research_enabled: true` in the server's routing JSON. Both default to false. Disabled
apps have no research routes in OpenAPI, no research service imports and no research tables or
artifact directories. No migrations or dependencies are added. The optional `training` group
must already be installed. Only locally generated tiny Llama geometries are supported; there
are no model downloads, GPU jobs or production deployments in this slice.

Copy `configs/routing/local.json` and `configs/identity/local.json` to local configuration files.
Set the routing flag in the copy and add `"capabilities": ["research"]` to a dedicated operator
key with grants covering **both** datasets. Construct `Settings` with those paths and the setting
flag. Existing operator keys have no capabilities by default. A body cannot supply identity or
capabilities. Research endpoints, plaintext aggregate downloads and benchmark submissions require
the research capability. Promotion also rechecks the source with this capability.

Use the [training runbook](training-and-promotion.md) to generate the tiny base, approve datasets,
and, optionally, train a registered adapter. `configs/research/licences.json` currently allows
only `CC0-1.0`; the verified base manifest must match. Calibration and evaluation datasets must
be approved `adapter_training` versions with authentic manifests/shards, and every tenant must
have current training permission. The narrower adapter-dataset scope keeps distillation and
router datasets out of research. Calibration defaults to the **validation** split; explicitly
choose `train` when appropriate. The test split is never available to instrumentation or tuning.

Lock the ordinary foundation baseline for the evaluation dataset. Evaluate the unpruned
registered adapter through `/v1/evaluations` and require a passed report bound to its exact
version and artifact digest. For a base without an adapter, POST `/v1/research/baselines`:

```json
{
  "base": {
    "base_model_id": "tiny",
    "base_model_revision": "seed-17",
    "base_model_licence": "CC0-1.0",
    "tokenizer_id": "tiny-byte-v1",
    "chat_template_version": "tiny-chat-v1"
  },
  "evaluation": {
    "candidate_deployment_id": "server-replaces-this-with-verified-base",
    "baseline_deployment_id": "fake-foundation-local-1",
    "dataset_id": "synthetic-evaluation",
    "dataset_version": "APPROVED_VERSION",
    "suites": ["golden", "held_out", "safety", "retrieval", "performance"],
    "critical_segments": ["citation", "safety"],
    "minimum_sample_size": 4,
    "performance_requests": 12
  }
}
```

The server binds the candidate to the verified base digest and uses the existing isolated
five-suite evaluator. A random tiny base normally fails; that is a precondition failure, not a
reason to waive the gate. Base evaluation creates no registry model or serving deployment.
Reports are MAC authenticated by the ordinary evaluation store. An unrelated foundation report,
a report for different data, or a failed report cannot authorize a study.

POST `/v1/research/jobs` with the research operator credential:

```json
{
  "job_type": "activation_study",
  "base": {"adapter_version": "UNPRUNED_REGISTERED_VERSION"},
  "calibration_dataset_id": "synthetic-evaluation",
  "calibration_dataset_version": "APPROVED_VERSION",
  "evaluation_dataset_id": "synthetic-evaluation",
  "evaluation_dataset_version": "APPROVED_VERSION",
  "baseline_evaluation_id": "PASSED_UNPRUNED_REPORT_ID",
  "calibration_split": "train",
  "sample_size": 64,
  "max_sequence_length": 128,
  "seed": 23
}
```

The bounded job runs off the HTTP event loop and returns its completed summary. Research jobs
are request-scoped local work, not an additional durable queue. Each job opens and closes its
own in-memory persistence, as evaluation does. No case events, replay records, tenant metrics
or tenant operational rows are written. Submission verifies all dataset shards; checks before
publication and summary reads authenticate manifests and current approval/policy without
decrypting the shards again. A repeated study ID with identical specifications returns the
existing study; changed specifications fail. Jobs fail with fixed codes, never library error bodies.

Sampling is seeded, without replacement, and uses at most the available rows. The maximum is
512 examples and 512 tokens per example. Before registering hooks, the memory check estimates
parameter/gradient/optimizer storage, saved activations, logits/loss workspace and eager attention
from the geometry and actual padded batch size, with a 1.5 safety allowance. A batch exceeding
`memory_limit_bytes` fails with `training_memory_limit`; this is admission estimation, not an OS
memory limit. Hooks observe each layer's residual contribution (`h_out − h_in`), attention output
before `o_proj`, and MLP output before `down_proj`. One backward pass on masked target-token
calibration loss computes a first-order zero-ablation estimate, `abs(activation · loss_gradient)`.
For layers this is `abs((h_out − h_in) · ∂L/∂h_out)`, including contribution-based norm, variance
and sparsity statistics. Ranking is by this sensitivity with deterministic index tie breaks.
This is an approximation to ablation
loss, not a claim of causal importance or measured hardware contribution.

The three safetensors keys are `layers`, `attention_heads`, and `mlp_channels`. Shapes are
`[layers, 1, 4]`, `[layers, heads, 4]`, and `[layers, intermediate_width, 4]`. The last dimension
contains norm mean, population variance, near-zero sparsity fraction and mean sensitivity.
Padding is excluded. No example, token, vocabulary, prompt or response dimension is saved.
Hooks and gradients are removed on completion or failure. The corrected hook version is
`aggregate-residual-taylor-v2`; older summaries remain readable, but layer pruning requires a
new study using this version.

Artifacts live under `<data_dir>/research/studies/<study_id>/`. Aggregates are AES-GCM encrypted
as `aggregates.safetensors.enc`, using the payload key version and AAD bound to
`research/<study_id>`. `summary.json` authenticates the summary, ciphertext digest, nonce and
key version with a purpose-separated MAC. Publication uses an atomic directory rename.
The directories are private. The content-free summary includes shapes and ranked structure
indices, provenance, actual sample count, seed, hook version and elapsed wall time. `report.md`
has the same summary, including linked candidate/evaluation/benchmark IDs after subsequent jobs.
The API never trusts the markdown as evidence.

GET `/v1/research/studies/STUDY_ID` reads the summary; GET the same path plus `/aggregates`
returns authenticated decrypted safetensors bytes only to an authorized research operator.
Copying ciphertext between studies, modifying tensors, or modifying signed JSON fails closed.
Preserve the research tree and payload keys separately alongside ordinary database backups.

To prune, POST to the same jobs endpoint:

```json
{
  "job_type": "structured_prune",
  "study_id": "STUDY_ID",
  "plan": {
    "structures": ["attention_heads", "mlp_channels"],
    "maximum_fraction": 0.25,
    "ranking_rule": "ablation_sensitivity"
  },
  "training": {
    "registry_id": "synthetic-pruned",
    "dataset_id": "synthetic-evaluation",
    "dataset_version": "APPROVED_CALIBRATION_VERSION",
    "base_model_id": "tiny",
    "base_model_revision": "seed-17",
    "base_model_licence": "CC0-1.0",
    "tokenizer_id": "tiny-byte-v1",
    "chat_template_version": "tiny-chat-v1",
    "steps": 500,
    "checkpoint_every": 100,
    "max_sequence_length": 128,
    "adapter_config": {"learning_rate": 0.005}
  }
}
```

The maximum fraction applies separately to each selected structure type, rounding down and
retaining at least one structure. It is capped at one half; default one quarter. Equal head
and channel counts are removed per retained layer so one standard Llama config can describe
the result. Layer removal renumbers all state-dict keys. Removing one of two layers therefore
requires `maximum_fraction: 0.5`. A plan that removes nothing is refused. Magnitude or any other
ranking rule returns `magnitude_ranking_unsupported`.

The tiny base has four query heads and two shared key/value heads. Removing one query head
requires expanding the retained shared KV weights into three independent heads to preserve
their mapping with the standard loader. This removes query/output work, but head-only pruning
has **no parameter reduction** on this fixture. Combining it with 25% MLP channel removal
reduces parameters from 35,168 to 32,096. Counts are measured from actual compact tensors.
No zero masks or unstructured sparsity substitute for physical removal.

The existing full student trainer fine-tunes the compact snapshot on the calibration dataset's
train split, with its usual deterministic CPU lock, memory admission, deadline and safetensors
checkpoint machinery. Temporary checkpoints are removed after request-scoped work; research
jobs do not add restart/resume promises. Exports contain the rewritten config, tokenizer assets,
full `model.safetensors` and numerical training report under `research/models/<version>/`.
The ordinary `SpecialistProvider` verifies the standard artifact MAC and loads this self-contained
`pruned-full-v1` export. The registry stores a `candidate`, complete study/plan/removed-index
lineage, both dataset versions, source adapter digest, and before/after counts. `datasets` and the
training completion event name the calibration dataset used for tuning; evaluation and promotion
bind to `pruning.evaluation_dataset`, which may be a different approved version. No route changes.

Evaluate the candidate on the approved evaluation dataset through `/v1/evaluations`, then POST
the normal `/v1/benchmarks/jobs` specification with its candidate version and evaluation ID.
The benchmark runs the candidate and exact unpruned base plus the same adapter on identical
held-out requests. The unpruned adapter is merged for instrumentation and benchmarking; its
registered version and artifact digest remain the reference binding. Each deployment runs in
a fresh process with private in-memory persistence;
inputs travel only through local memory/IPC. Warm p50/p95, throughput, process peak RSS,
parameter count, independent paired quality and cost accounting are recorded. RSS includes
the interpreter and libraries, but never the other model. Process startup is outside warm
latency measurement. No configured cost savings can satisfy the pruning hardware gate.

`initial-targets.json` requires either 20% p95 latency reduction or 20% peak RSS reduction,
plus adequate independent quality samples, successful outcomes, the full ordinary evaluation
gates, and the completed safety suite with zero critical failures. Failure to demonstrate
hardware improvement gives `no_hardware_benefit`. Promotion re-verifies the benchmark MAC,
exact model/source/data/evaluation bindings, current thresholds and the foundation baseline.
Source identification verifies base and adapter artifact bytes without instantiating or merging
models; only study, tuning, baseline evaluation and benchmark measurement load them.
A passing evaluation or smaller parameter count alone cannot approve a pruned candidate.

Reproduce the synthetic checks without downloading anything:

```sh
UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache uv run --locked pytest tests/unit/test_research_boundary.py tests/unit/test_research_gates.py tests/unit/test_research_tensors.py tests/integration/test_research.py -s
UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache uv run --locked pytest tests/integration/test_research.py -m 'smoke and accelerator_free' -s
UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache make contracts
UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache make ci
```

The smoke learns one synthetic no-context response with 500 tuning steps. Its initial measured
study → prune → tune → evaluate → benchmark path was **6.72 s**, below 120 s. Quality passed,
but hardware improvement did not: promotion was correctly refused. Numerical gate fixtures
separately prove that adequate latency or RSS improvement permits approval, while failed safety
or tampered reports block it. These measurements do not establish general capability retention,
real-workload ablation value, hardware savings, human safety review or production acceptance.
