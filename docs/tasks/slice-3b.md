# Task: slice 3b — real LoRA training behind an optional dependency group

Second half of milestone 3. Replace the fake trainer with a real LoRA trainer as described in the
specification (`docs/spec/`, sections 7.10, 11.1, 11.3, 11.6, 20.3, 21.3, 21.4 CPU-safe smoke
and separately marked accelerator tests) while keeping every lifecycle guarantee from slice 3a.
Read `AGENTS.md`, `docs/status.md`, `docs/runbooks/training-and-promotion.md` and
`pipelines/training/README.md` first.

**Dependencies are already installed and locked by the reviewer** in the optional group
`training` (`torch` CPU build, `transformers`, `peft`, `safetensors`). Import them only inside
`src/adaptive_llm/training/lora.py` and guard the import so the rest of the platform, the
default test suite and CI without the group keep working; tests that need them are marked
`accelerator_free` and skipped with a clear reason when the group is absent. Do not touch
`pyproject.toml` or `uv.lock`.

## Deliverables

1. **Asynchronous job submission.** `POST /v1/training/jobs` returns the `TrainingJob` in state
   `queued` immediately; a background worker task started in the lifespan takes one job at a
   time from a durable queue (the `training_jobs` table with state `queued`, ordered by
   creation), runs it under the existing lease, and persists state transitions and checkpoints
   as before. `GET /v1/training/jobs/{job_id}` polls. Add `POST /v1/training/jobs/{job_id}/cancel`
   that requests cooperative cancellation between steps and records `cancelled`. The worker
   survives a restart: a job found `running` at startup with checkpoints resumes; one without
   checkpoints restarts. Tests with the fake trainer.
2. **Real trainer** `LoraTrainer` implementing the `Trainer` Protocol with
   `architecture = "lora-peft-v1"`: loads the base model and tokenizer from a local path only
   (never downloads; `base_model_id` must resolve under `<data_dir>/base-models/<id>/<revision>/`
   with a manifest of file digests verified before load), builds the canonical example text per
   spec 11.1 from the shard rows with the chat template, tokenises within `context_limit`,
   applies PEFT LoRA with the spec's `AdapterConfig` (target modules, rank, alpha, dropout),
   trains with the seeded optimiser for `steps` from the job specification (add `steps`,
   `batch_size`, `max_sequence_length` to `TrainingJobSpecification`), writes a checkpoint every
   `checkpoint_every` steps as safetensors plus the training report (loss curve, tokens seen,
   steps, wall clock, peak RSS), and exports both merged and unmerged adapters as safetensors.
   Determinism: same specification and dataset produce identical loss curves and adapter
   digests on CPU with fixed seeds and single-threaded execution; test it.
3. **Tiny base model fixture**: generate a deterministic tiny causal LM (two layers, small
   vocabulary, random init from a fixed seed) and tokenizer into `tests/fixtures/base-models/`
   at test time, with its digest manifest, so the CPU smoke test needs no download and no
   licence question. Document how a real open-weight base model is placed under
   `base-models/` with its licence file and revision.
4. **Specialist provider for real adapters**: `SpecialistProvider` loads the verified
   safetensors adapter onto the tiny base model and generates greedily with a token budget,
   honouring `max_output_tokens`, returning `Usage` with `source="provider_reported"` and the
   real tokenizer id, citations parsed from the output with the existing pattern, and finish
   reasons. Deterministic on CPU.
5. **Evaluation of a real adapter**: the existing five suites run against the trained tiny
   adapter; the report records the adapter digest; the gate behaves exactly as before. Add
   catastrophic-forgetting and refusal checks from spec 11.3 as two new golden items scored
   by the deterministic judge.
6. **Resource limits**: training refuses to start if the estimated memory (parameters ×
   precision × 4 with a documented safety factor) exceeds `Settings.training_memory_limit_bytes`,
   and stops with `training_interrupted` if wall clock exceeds `Settings.training_time_limit_seconds`,
   checkpointing first. Tests with tiny limits.
7. **Tests**: the CPU smoke path train (real, tiny) → evaluate → approve → shadow in under
   sixty seconds, marked `smoke` and `accelerator_free`; determinism; resume after interrupt
   with real checkpoints; cancellation; restart recovery; memory and time limits; no plaintext
   example content in any artifact or checkpoint (assert on bytes; safetensors and JSON only);
   the default suite passes with the `training` group uninstalled (simulate by making the
   import fail).
8. **Docs**: update the training runbook with the real trainer commands, the base-model
   placement rule and the measured smoke timing; `pipelines/training/README.md`;
   `docs/status.md` milestone 3 exit criteria table with numbers.

## Out of scope

GPU tests (mark `accelerator` and skip), distributed training, distillation, router training,
shadow and canary traffic, any model download. Do not modify `docs/spec/`, `pyproject.toml`
or `uv.lock`.

## Allowed dependency additions

None beyond the already-locked `training` group.
