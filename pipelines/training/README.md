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

Milestone 4 adds specialist/shadow/canary traffic; milestone 5 adds distillation. Registry
production state alone does not enable specialist serving. No dependencies were added here.
