# Local teacher distillation and student admission (slice 5a)

Distillation uses two explicit dataset approvals. Approve an `adapter_training` source, build
a `distillation` dataset through `/v1/datasets/builds`, inspect its counts and lineage, then
approve that immutable version before submitting a `distillation` training job. This separates
teacher curation from the durable student queue so generated targets receive the same operator
approval as other training data. No approval is inferred from the teacher's registry state.
The teacher must be the configured foundation or an `approved`, `canary`, or `production`
generative specialist. An evaluating, shadow, deprecated, revoked, or router version is refused.

Start with the approved source in [training and promotion](training-and-promotion.md). POST:

```json
{
  "dataset_id": "synthetic-distillation",
  "purpose": "distillation",
  "tenant_ids": ["synthetic-a", "synthetic-b"],
  "source_dataset_id": "synthetic-evaluation",
  "source_dataset_version": "APPROVED_SOURCE_VERSION",
  "source_window": {"start": "2026-09-24T00:00:00Z", "end": "2026-09-25T00:00:00Z"},
  "eligibility_policy_version": "synthetic-dataset-policy-1",
  "teacher_deployment_id": "fake-foundation-local-1",
  "teacher_minimum_score": 0.9,
  "teacher_max_output_tokens": 256,
  "general_safety_fraction": 0.2,
  "soft_targets": false,
  "seed": 23
}
```

Use the source's reviewed window, tenant set and current policy version. The source's grouping
and folds are retained. Only train inputs reach the teacher; validation and test keep their
original targets. Exact source boundaries, document versions, messages, subject grouping and
original target provenance are preserved. Teacher targets have separate source-target lineage;
they are not attributed to the original response attempt or correction.

Teacher generation runs through the existing inference evaluation runner with the server's
evaluation identity, an isolated in-memory database, private metrics and discarded case events.
Hard validation failures, generation failures and incomplete outputs are rejected. The pinned
deterministic judge checks the original approved target as an expected fact and citation validity,
scoring on a five-point scale normalized to [0, 1]. This deliberately conservative lexical judge
can reject correct paraphrases; it is not a human-calibrated measure. Golden decontamination checks
both whole examples and targets alone with the existing normalized 5-gram similarity. Targets
changed by persistence redaction, or whose redaction fails, are excluded. The manifest contains
only counts, hashes and teacher/judge/generation lineage; content remains in encrypted shards.

`general_safety_fraction` is the requested fraction of the final train split. For N accepted
teacher rows, the builder adds `round(N * fraction / (1 - fraction))` examples, cycling through
four dedicated synthetic fixtures. Two cover general capability; the others cover abstention
and escalation/refusal. They live in `tests/fixtures/{golden,safety}/distillation-training.jsonl`
and are separate from the evaluation files `synthetic.jsonl`. The builder rejects a positive mix
too small to cover all four fixtures; at fraction .2, use at least 16 accepted teacher rows.
The manifest records actual category counts, requested fraction and fixture digest. The tiny
one-example lifecycle demonstration explicitly sets the fraction to zero; it does not demonstrate
general-capability preservation. Mixing and evidence retention have separate tests.

All manifest tenants must have current training permission, including empty shards. Source
approval, grants, source deletion/retention and per-application current policy are checked again
when the queued student runs and before publication. Dataset publication rechecks deletions;
removed examples and their tensor payloads cannot survive a rebuild of the staging directory.
Published immutable versions are never silently rewritten.

Approve the returned dataset with the existing operator endpoint and a nonblank reason. Generate
the smaller base locally, using the already installed optional training group:

```sh
uv run --locked python -c 'from pathlib import Path; from adaptive_llm.training.lora import generate_tiny_base; generate_tiny_base(Path(".local/evaluation-demo/base-models/tiny-student/seed-17"), student=True)'
```

This student fixture has one Llama layer, 16 hidden units and the same 259-token byte tokenizer.
The earlier tiny teacher fixture has two layers and 32 hidden units. Student bases may use any
supported geometry. For registered tensor teachers, the student's parameter count must be
strictly smaller than the verified teacher base inventory. The fake foundation has no tensor
parameter count, so size cannot restrict admission; lineage records `teacher_parameter_count=null`
and the model manifest records the limitation `teacher size unknown; size reduction not verified`.
No model assets are downloaded.

Submit this specification to `/v1/training/jobs`, or save it as `student.json` and run
`make train SPEC=student.json DATA_DIR=.local/evaluation-demo POLICY=configs/policy/training-demo.json`:

```json
{
  "job_type": "distillation",
  "registry_id": "synthetic-student",
  "dataset_id": "synthetic-distillation",
  "dataset_version": "APPROVED_DISTILLATION_VERSION",
  "base_model_id": "tiny-student",
  "base_model_revision": "seed-17",
  "base_model_licence": "CC0-1.0",
  "tokenizer_id": "tiny-byte-v1",
  "chat_template_version": "tiny-chat-v1",
  "student_training": "full",
  "soft_target_weight": 0.5,
  "max_sequence_length": 128,
  "steps": 500,
  "checkpoint_every": 100,
  "adapter_config": {"learning_rate": 0.005},
  "input_micros_per_1000_tokens": 1,
  "output_micros_per_1000_tokens": 1
}
```

The prices are explicitly synthetic USD micros per 1,000 tokens, not measured production costs.
Distillation selects the CPU student trainer even when the ordinary adapter backend is `fake`.
`student_training="full"` exports `student-full-v1`; `"lora"` exports `lora-peft-v1` on the student
base. Both derive `student_architecture` from the verified base configuration (for example,
`llama-1x16-v1`) and record `student_parameter_count` so size reduction is auditable. They use the
existing memory/time limits, cancellation, checkpoints, durable queue, Adam/RNG safetensors and
signed export inventory.
Same data/specification produces identical complete artifact digests, including after resume.
`SpecialistProvider` verifies and loads both formats. Without the optional group, student jobs
fail with `training_dependencies_unavailable`; ordinary serving, datasets and fake training work.

New dataset approvals record `approval_mac_version="2"` and authenticate the full manifest JSON
with the MAC itself cleared. New model manifests record `manifest_mac_version="2"` and
authenticate all immutable metadata, including null/default fields; the artifact MAC and mutable
registry lifecycle fields remain outside that signature. Records without a version default to
version `"1"` and retain their historical verification rules. No database migration is needed.

When `soft_targets=true`, real teachers provide target-token log distributions by a teacher-forced
pass over the accepted response and its canonical context. A fake teacher produces hard targets
only. The distributions and target-token alignment are serialized as safetensors, then encrypted
as binary `*.safetensors.enc` files with tenant/dataset/version/filename AAD. Manifest MACs bind
the hashes and key versions; the dataset digest includes tensor hashes. There are no JSON token
arrays or distributions. Tokenizer/template compatibility, target IDs, vocabulary dimensions,
normalization and finiteness are checked before KL training. Targets exceeding the teacher's
context limit fail the build. Student loss combines hard cross-entropy and target-position KL,
with `soft_target_weight` applied only to batches with soft observations. Optional imports remain
inside `training/lora.py`.

Lock a foundation baseline for the distilled dataset, then evaluate the student using the same
five suites and exact dataset version. See [evaluation](evaluation.md). Distilled reports add
`segment_comparisons.distillation`, a seeded paired bootstrap of student-minus-teacher held-out
quality, plus teacher comparisons on every requested critical segment. Too few samples, missing
measurements or a confidence lower bound at/below the negative margin fail the additional gate.
Ordinary golden, safety, retrieval, performance and foundation non-inferiority gates still apply.
The teacher version/digest, judge and lineage must match the approved distillation record.

Before promotion, run the benchmark job:

```sh
curl -X POST http://127.0.0.1:8000/v1/benchmarks/jobs \
  -H 'Authorization: Bearer synthetic-operator-key' -H 'Content-Type: application/json' \
  -d '{"candidate_version":"STUDENT_VERSION","evaluation_id":"EVALUATION_ID","requests":40}'
```

This bounded local job runs on a worker thread and returns a `BenchmarkReport`. GET
`/v1/benchmarks/jobs/BENCHMARK_ID` retrieves it; repeating the same id/specification is idempotent.
The control database stores the record with a purpose-separated MAC in migration 0011. Ordinary
paired database backup/restore covers it. The report binds the exact student, teacher, evaluation,
dataset, request mix and both price lists. Both models receive the same held-out input sequence
at concurrency four after warmup. Latency p50/p95, throughput, process peak RSS, total cost and
integer cost per successful outcome are measured. Success means completion without an error or
hard validation failure; assertion results and quality scores belong to the paired comparison and
evaluation gates. All attempts contribute cost; no successes means no per-success cost. Repeated
requests do not inflate the independent quality sample count.

`configs/evaluation/initial-targets.json` requires at least four independent cases, a quality CI
lower bound above -.02, at least one successful outcome per model, and either at least 20% p95
latency reduction or 30% cost reduction. A faster inferior student fails. Promotion to `approved`
re-verifies benchmark
MACs and exact bindings, current thresholds and the current foundation baseline; a passing
evaluation alone is insufficient. There is no automatic promotion. Later shadow/canary/production
steps use the existing [canary and rollback](canary-and-rollback.md) workflow unchanged.

Peak RSS is a process high-water mark including loaded teachers, students and prior work; it is
not isolated model allocation. CPU tensor execution is serialized by the deterministic CPU lock.
Timing varies with host scheduling; tests compare repeated warm measurements within a factor of
five, and require exact quality, request-mix and cost results. A fake teacher is much faster than
the tensor student; the smoke passes through its explicitly configured cost reduction. Neither
this benchmark nor the narrow passing smoke claims general quality or production acceptance.

```sh
uv run --locked pytest tests/unit/test_distillation_gates.py tests/integration/test_distillation.py -s
uv run --locked pytest tests/integration/test_distillation.py -m smoke -s
make contracts
make check integration
make ci
```

In the restricted sandbox prefix commands with
`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache` to preserve the installed optional
group. If the network-only SBOM audit stops `make ci`, still run all remaining gates:

```sh
uv run --locked pytest tests/security tests/load -s
uv run --locked pytest tests/drills -m drill -s
uv run --locked pytest -m smoke -s
make check-without-training
```
