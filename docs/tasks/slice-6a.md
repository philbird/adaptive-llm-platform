# Task: slice 6a — isolated activation analysis and structured pruning (optional research)

Optional milestone 6, as described in the specification (`docs/spec/`, sections 3 non-goals,
4 Phase E, 16.1–16.3, 19 milestone 6, 21.2 step 13, 21.3 feature flags) and
`pipelines/activation_research/README.md`. Read `AGENTS.md`, `docs/status.md`,
`docs/runbooks/distillation.md` and `docs/runbooks/training-and-promotion.md` first.
Milestones 1–5 are on `main`. This is a research extension behind a feature flag; nothing here
may reach serving except through the standard registry, evaluation, benchmark and promotion
path. Everything remains local with the tiny generated base models and the optional
`training` extra.

## Deliverables

1. **Feature flag and isolation.** `Settings.pruning_research_enabled` (default false) and the
   existing `configs/routing/local.json` flag of the same name must both be true for any of the
   endpoints or jobs below to exist; otherwise they are absent from the OpenAPI document. All
   research code lives in `src/adaptive_llm/research/` and is imported only when the flag is
   set. Research artifacts live under `<data_dir>/research/` in their own directory tree, and
   the research job uses its own isolated in-memory persistence exactly as evaluation does. A
   test asserts that with the flag off no research route, table or import is present.
2. **Preconditions (spec 16.1)** enforced at job submission: the base model manifest carries a
   licence in an allowlist from `configs/research/licences.json` (the tiny fixture's CC0
   qualifies), the calibration and evaluation dataset versions are approved and their manifests
   verify, an unpruned baseline evaluation report for that base exists and passed, and the
   operator key class has `research` in a new `capabilities` list on `KeyIdentity` (default
   empty). Refused with fixed codes otherwise.
3. **Activation instrumentation (spec 16.2)**: a job type `activation_study` that runs the
   verified base (with an optional registered adapter) over a bounded sample of the calibration
   split (`sample_size`, default 64, maximum 512) with forward hooks capturing only aggregated
   statistics per layer, per attention head and per MLP channel: activation norm mean and
   variance, sparsity fraction, and a gradient-based sensitivity from one backward pass on the
   calibration loss. Never persist per-token tensors; the study record holds only the
   aggregates as arrays with shapes bounded by the model geometry, stored as safetensors with a
   MAC and a content-free JSON summary. Treat the aggregates as sensitive: encrypted at rest
   with the payload cipher bound to `research/<study_id>`, readable only with the research
   capability. Record base version, adapter version, dataset version, sample size, seed, hook
   version and wall clock.
4. **Structured pruning (spec 16.3)**: a job type `structured_prune` that takes a study id, a
   pruning plan (which of `attention_heads`, `mlp_channels`, `layers` to consider, a maximum
   fraction to remove per structure type, default 0.25, and a ranking rule limited to
   `ablation_sensitivity`; magnitude-only ranking is refused with `magnitude_ranking_unsupported`
   per spec 3), performs conservative structured removal that keeps the model loadable by the
   standard loader (rewrite `config.json` and the safetensors consistently; remove whole heads
   and channels, never unstructured sparsity), then fine-tunes the pruned model with the
   existing student trainer machinery for `steps` from the specification, and registers the
   result as an ordinary `ModelManifest` with `adapter_architecture="pruned-full-v1"`, full
   lineage (study id, plan, removed structure indices, parameter count before and after) and
   the standard MAC. It enters the registry in state `candidate` like any model.
5. **Hardware-level benefit (spec 16.3 steps 6–7)**: the existing benchmark job compares the
   pruned candidate against its unpruned base plus the same adapter, reporting wall-clock p50
   and p95, throughput, peak RSS and parameter count. Promotion of a pruned candidate requires,
   in addition to every existing gate, a passed benchmark with `p95_latency_reduction_min` or a
   new `peak_rss_reduction_min` from `initial-targets.json` met, and the full safety suite
   with zero critical failures. A pruned candidate that reduces parameters but not measured
   latency or memory is refused with a gate reason `no_hardware_benefit`.
6. **Research report**: `GET /v1/research/studies/{study_id}` returns the content-free summary
   (aggregate statistics shapes and ranked structure lists by sensitivity, not values from any
   single input) and, after a prune job, the before/after parameter counts, benchmark ids and
   evaluation ids. A markdown research report is written alongside with the same content.
7. **Tests**: flag off means no routes, tables or imports; preconditions each refused; study
   aggregates contain no per-token data (inspect the safetensors keys and shapes); study
   artifacts encrypted and MAC-verified; a plan removing 25 % of heads of the tiny base loads
   through the standard loader and generates; magnitude ranking refused; pruned candidate
   passes or fails promotion exactly according to the benchmark and safety gates; the smoke path
   study → prune → fine-tune → evaluate → benchmark completes in under one hundred twenty
   seconds, marked `smoke` and `accelerator_free`; the default suite passes with the flag off
   and without the training group.
8. **Docs**: `docs/runbooks/activation-research.md`; update
   `pipelines/activation_research/README.md`; `docs/status.md` milestone 6 row and exit
   criteria table; a final section in `docs/status.md` summarising the spec's milestone
   coverage 0–6 with what is and is not claimed.

## Out of scope

Any production deployment of a pruned model, GPU, downloads, real open-weight models, bandits.
Do not modify `docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None.
