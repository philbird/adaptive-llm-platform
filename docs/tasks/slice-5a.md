# Task: slice 5a — distillation into a smaller student through the same gates

Milestone 5, as described in the specification (`docs/spec/`, sections 4 Phase C, 7.10,
11.4, 11.6, 12.3, 12.4, 19 milestone 5, 20.3, 21.2 step 12, 23 "training on incorrect model
outputs" and "judge-model bias"). Read `AGENTS.md`, `docs/status.md`,
`docs/runbooks/training-and-promotion.md` and `docs/runbooks/canary-and-rollback.md` first.
Milestones 1–4 are on `main`. Everything remains local; the teacher is a registered specialist
or the foundation, the student is a smaller tiny base model, and the optional `training` extra
is installed.

## Deliverables

1. **Teacher target generation** in `src/adaptive_llm/distillation/`: a job type
   `distillation` in the training orchestrator that, for an approved adapter-training dataset
   version, runs a registered teacher (a `canary`/`production`/`approved` specialist version or
   the foundation) over the train split's inputs in process through the existing evaluation
   runner path (evaluation identity, isolated persistence, no tenant telemetry), producing
   candidate targets. Targets pass the existing validator (hard checks) and the deterministic
   judge at a configured minimum score, and are decontaminated against the golden set, before
   they enter the distillation dataset. Rejected targets are counted by reason. Record teacher
   version, artifact digest, generation parameters and judge version in the manifest lineage
   (spec 11.4). Optionally include the teacher's per-token log-probabilities as soft targets
   only when the provider can supply them (the fake cannot; the tiny real model can) and the
   dataset specification's `soft_targets` flag is set; store them as safetensors, never JSON
   text.
2. **Distillation dataset**: purpose `distillation`, built through the existing dataset
   builder path with a `source_dataset_id`/`version` lineage like router datasets, keeping the
   original example's provenance and the RAG context so the student learns to use evidence
   (spec 11.4), mixing in a controlled general-capability and safety set from
   `tests/fixtures/golden` and `tests/fixtures/safety` at a configured fraction with
   abstention and escalation examples, and recording the mix in the manifest. Approval and
   current-policy checks apply unchanged.
3. **Student trainer**: reuse `LoraTrainer` machinery to fine-tune a *smaller* tiny base
   (generate a second fixture, one layer, 16 hidden units) either fully or with LoRA per the
   job specification, with a `student_architecture` recorded in the manifest, optional soft-
   target KL loss when soft targets exist, and the same determinism, checkpoints, limits and
   signed exports as slice 3b. `adapter_architecture` for a full fine-tune is
   `student-full-v1`; for LoRA on the student `lora-peft-v1` with `base_model_id` naming the
   student base.
4. **Same gates**: the student is evaluated by the existing five suites against the locked
   baseline and promoted only through the existing state machine; add a `distillation`
   segment to the evaluation report comparing student and teacher on the held-out fold
   (paired bootstrap) and require non-inferiority to the teacher within the margin as an
   additional gate for distilled candidates. The `SpecialistProvider` loads student artifacts
   through the same verification.
5. **Deployment benchmark** (milestone 5 exit): a benchmark job that measures the student and
   teacher on the same synthetic request mix on this machine: p50/p95 latency, throughput at
   concurrency 4, peak RSS, and cost per successful outcome using each model's price list;
   written as a `BenchmarkReport` record with a MAC, and required by the promotion gate for
   distilled candidates to show a material improvement in either latency or cost at
   non-inferior quality (configurable thresholds in `initial-targets.json`).
6. **Tests**: teacher targets are filtered and counted; no target text in manifests or logs;
   soft targets are safetensors only; distillation dataset mix recorded; student training
   deterministic and resumable; student evaluation runs the five suites plus the teacher
   comparison; a student worse than the teacher fails the new gate; a student that is faster
   but not non-inferior fails; benchmark report reproducible within tolerance; CPU smoke
   train-student → evaluate → approve in under ninety seconds, marked `smoke` and
   `accelerator_free`; suite passes without the training group.
7. **Docs**: `docs/runbooks/distillation.md`; `docs/status.md` milestone 5 exit criteria table
   with numbers; update `pipelines/training/README.md`.

## Out of scope

Real teacher models, downloads, GPU, shadow or canary changes, research milestone 6. Do not
modify `docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None.
