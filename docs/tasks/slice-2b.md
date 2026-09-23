# Task: slice 2b — evaluation harness and locked foundation baseline

Second half of milestone 2. Implement the evaluation platform as described in the specification
(`docs/spec/`, sections 7.11, 9.3 evaluations only, 9.4 `evaluation.completed.v1`, 12.1 layers
1–7, 12.2, 12.3, 12.4, 20.4, 21.2 step 9) and the provisional targets in
`configs/evaluation/initial-targets.json`. Read `AGENTS.md`, `docs/status.md`,
`docs/runbooks/dataset-build.md` and `pipelines/dataset_builder/README.md` first. Milestone 1
and slice 2a are on `main`. Everything remains local and synthetic; the only model under test
is the deterministic fake provider, which is the "foundation baseline" for this slice.

## Deliverables

1. **Evaluation contracts** in `contracts.py` (then `make contracts`): `EvaluationSpecification`
   (evaluation id, candidate deployment id, baseline deployment id or null, dataset id and
   version for held-out, suite list from the five required suites, rubric version, judge
   version or null, seed, confidence level, non-inferiority margin, minimum sample size,
   critical segments as label keys), `SuiteResult` (suite, items, metrics as a bounded dict of
   floats, per-segment metrics, failures by reason code), `EvaluationReport` (specification,
   candidate and baseline manifests' versions, code revision, started/completed, suite
   results, paired comparison with mean delta, bootstrap CI bounds and sample size, gate
   decisions as a list of `{gate, passed, reason}`, overall `passed`, known limitations).
   Both are `Record`s. The existing `EvaluationCompleted` event carries the report id, versions,
   suites and pass flag; add `report_ref`.
2. **Suites**, each a Protocol implementation under `src/adaptive_llm/evaluation/suites/`,
   consuming a `Runner` that calls the serving pipeline in process (through
   `InferenceService`, not HTTP) with an evaluation identity so no telemetry is confused with
   production: mark those interactions with `application_id="evaluation"` and a policy that
   denies logging and training.
   - `golden`: the 20 items in `tests/fixtures/golden/`; deterministic assertions per item
     (`expect_contains`, `expect_citation`, `expect_json`) plus a rubric score from the
     deterministic judge below.
   - `held_out`: the test split of a dataset version (decrypt shards through the existing
     cipher; operator identity required), scoring target match by normalised token F1 and
     citation precision/recall against the example's provenance.
   - `safety`: a synthetic adversarial set under `tests/fixtures/safety/` (20 items: prompt
     injection in retrieved content, tool-abuse attempts, secret-extraction attempts,
     cross-tenant retrieval attempts); metrics are injection success rate, leakage rate and
     cross-tenant incidents; any critical item failing fails the suite.
   - `retrieval`: recall@k, MRR and context precision over a synthetic relevance file
     `tests/fixtures/retrieval/relevance.jsonl`, plus an ACL isolation check.
   - `performance`: p50/p95/p99 latency, time to first token (null for the fake), error rate and
     cost per successful outcome over N requests at concurrency C from the specification.
3. **Deterministic judge** in `src/adaptive_llm/evaluation/judge.py`: a `Judge` Protocol and a
   rule-based implementation with a pinned `judge_version` that scores 0–5 on presence of
   expected facts, citation validity and absence of prohibited content, with answer order
   randomised and model identity masked in its inputs even though it is rule-based, so the
   interface is honest for a later LLM judge (spec 12.4). Record disagreements between the
   judge and deterministic assertions.
4. **Statistics** in `src/adaptive_llm/evaluation/stats.py`: paired per-item deltas, mean,
   bootstrap percentile CI at the specified confidence with a seeded RNG, and a
   `required_sample_size(margin, pilot_sd, confidence)` helper per
   `configs/evaluation/initial-targets.json`. Unit tests with known inputs.
5. **Promotion gate** in `src/adaptive_llm/evaluation/gate.py`: implements spec 12.3 exactly:
   all required suites completed; non-inferiority (CI lower bound > −margin) overall and on
   every critical segment; zero critical safety, privacy or isolation regressions; latency,
   error and cost targets met; sample size ≥ minimum; report includes CI, n, coverage and
   limitations. Hard gates cannot be outweighed by cost. Output is the list of gate decisions;
   a missing sample size is a fail, not a skip.
6. **Baseline lock**: `POST /v1/evaluations` (operator key) runs an evaluation off the event
   loop, persists the report in `evaluation_reports` (migration 0006), writes the report JSON
   under `<data_dir>/evaluations/<evaluation_id>/` with a `report.mac`, emits
   `evaluation.completed.v1` per tenant, and when `candidate == baseline == foundation` and the
   gate passes, records a `baselines` row (migration 0006) keyed by deployment id and dataset
   version that later candidates compare against. Re-running against the same baseline key
   is refused with 409 unless `replace: true` and an operator note. `GET
   /v1/evaluations/{evaluation_id}` returns the report. `make evaluate SPEC=<file>` runs the
   same from the CLI.
7. **Tests**: each suite on the fake provider; a candidate that is worse by construction (a
   provider wrapper that drops citations) fails the gate on the citation and safety segments;
   a candidate identical to baseline passes; sample size below minimum fails; a CI straddling
   the margin fails; the report contains no content (assert on JSON bytes); evaluation
   interactions are not eligible for datasets (build after evaluating, assert the count is
   unchanged).
8. **Docs**: `docs/runbooks/evaluation.md` with commands and expected output; update
   `configs/evaluation/initial-targets.json` only by filling `minimum_sample_sizes` from the
   helper with a documented pilot standard deviation; update `docs/status.md` milestone 2 rows
   and record the baseline lock; update `tests/README.md`.

## Out of scope

Human review workflow, LLM judge implementation, shadow or canary analysis, training, registry
state machine, approval API. Do not modify `docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None. Bootstrap and F1 in the standard library.
