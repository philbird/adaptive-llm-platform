# Training lifecycle

Slice 3a lives in `src/adaptive_llm/training/` and `src/adaptive_llm/registry/`. It implements
operator dataset approval, current-policy checks, deterministic fake CPU training, authenticated
checkpoints/artifacts, resume, complete lineage, candidate evaluation, gated promotions and
transactional rollback. The CPU smoke test exercises train → evaluate → approve → shadow.
See the [training runbook](../../docs/runbooks/training-and-promotion.md).

Slice 3b implements offline CPU LoRA in `src/adaptive_llm/training/lora.py`, the only importer
of the optional Torch/Transformers/PEFT/safetensors stack. Select `BACKEND=lora` with `make train`
or `Settings(training_backend="lora")`; the default fake backend works without that group.
HTTP submission returns `queued`, a durable lifespan worker processes one job at a time, GET
polls, and POST `/v1/training/jobs/{job_id}/cancel` requests cooperative cancellation. Restart
recovery preserves authenticated submitters, current policy checks and signed checkpoints.

Base models must already exist under `<data_dir>/base-models/<id>/<revision>/` with an exact
digest manifest, pinned tokenizer/template and licence file; the platform never downloads models.
Tests generate a two-layer random tiny model and tokenizer in `tests/fixtures/base-models/`.
Real checkpoints contain safetensors for adapters, Adam state and RNG plus authenticated JSON;
exports include both merged and unmerged tensors. Published reports record deterministic losses,
token counts and lineage; wall time and peak RSS stay in job resource usage. Completed checkpoints
move to `models/<registry>/.checkpoints/<version>/` outside the hashed export inventory, so full
artifact digests match across identical fresh and resumed runs.
Memory admission and step-boundary time limits are configurable in `Settings`.

Run `uv run --locked pytest tests/integration/test_lora.py -s` for CPU tensor tests and
`make check-without-training` to simulate the group being absent. `make ci` runs this gate after
the smoke step.
Real tests are marked `accelerator_free`; the CPU smoke is also marked `smoke` and must finish
within sixty seconds. GPU coverage is marked `accelerator` and skipped, as it is out of scope.
The passing smoke uses a deliberately narrow no-context task; the ordinary five suites also run
against a tiny adapter and verify rejection of failed quality gates. See the runbook for measured
timings, placement rules, formats and commands.

Slice 5a adds `src/adaptive_llm/distillation/`: approved-source teacher curation through isolated
evaluation, an explicitly approved distilled dataset, and queued full/LoRA training of a one-layer,
16-unit student. Synthetic general/safety training partitions remain separate from evaluation.
Optional teacher log distributions use encrypted safetensors and target-position KL loss.
The existing five suites gain a paired held-out teacher comparison. A MAC-authenticated deployment
benchmark with non-inferior quality and material latency or cost improvement is also required for
student approval. See the [distillation runbook](../../docs/runbooks/distillation.md) for commands,
formats, thresholds and measurement limitations. The new smoke is marked `accelerator_free` and
`smoke`, and must finish within ninety seconds. No dependencies were added.

Milestone 4 controls specialist/shadow/canary traffic. Registry production state alone does not
enable specialist serving; slice 5a leaves those controls unchanged.
