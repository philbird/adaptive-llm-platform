# Task: slice 4b — router training, calibrated live routing, canary and automatic rollback

Second half of milestone 4: specialists answer users for the first time, under every control
built in 4a, as described in the specification (`docs/spec/`, sections 7.5, 11.5, 12.2 routing
metrics, 13.1–13.4, 15.2 steps 4–7, 15.3, 19 milestone 4 exit criteria, 20.4, 20.5, 23 "router
sends novel tasks to specialist" and "apparent savings hide fallback cost"). Read `AGENTS.md`,
`docs/status.md`, `docs/runbooks/shadow-and-kill-switch.md` and `docs/runbooks/evaluation.md`
first. Milestones 1–3 and slice 4a are on `main`. Everything remains local; specialists are
registered fake or tiny real adapters.

## Deliverables

1. **Counterfactual routing dataset** (spec 11.5): a builder in `src/adaptive_llm/routing/train.py`
   that assembles, for a dataset version's held-out and shadow observations, one row per
   interaction with: task label, language, risk tier, retrieval features (chunk count, top
   score, index id), input token count, whether context was supplied, and, for each candidate
   deployment (foundation plus each shadowed specialist), the observed quality proxy (held-out
   item score where available, otherwise the shadow comparison score), validation pass, cost
   and latency. Rows contain features and numbers only, never text. Interactions with no
   specialist observation are kept with the specialist columns null, so the router learns
   abstention from coverage gaps rather than from historic selection alone. Write it as an
   encrypted shard with a manifest like datasets, purpose `router_training`, through the
   existing dataset builder path (add the purpose), so approval and lineage apply.
2. **Router model**: a rules-plus-classifier router per spec 11.5 with no new dependencies:
   deterministic hard constraints first (residency, modality, context length, tool support,
   tenant and task enablement, breaker state), then a calibrated per-specialist suitability
   estimate from a small logistic model trained by plain-Python gradient descent on the
   routing dataset (features one-hot task, language, risk tier, binned token count, context
   flag, retrieval top-score bin), with Platt-style calibration on a held-out fold and an
   out-of-distribution score defined as the distance of the feature vector from the training
   feature centroid in normalised space, scaled to [0, 1]. Training is a job type `router` in
   the existing training orchestrator (fake-trainer style, no torch) producing a signed
   artifact and a `ModelManifest` with `adapter_architecture="router-logistic-v1"`, so the
   router goes through the same registry, evaluation and promotion gates. Evaluation for a
   router: a `routing` suite that reports false-specialist rate, unnecessary-foundation rate,
   calibration error (expected calibration error over ten bins) and OOD detection rate on the
   held-out fold; the promotion gate requires false-specialist rate ≤ the configured target
   (`initial-targets.json`, 2 %) and calibration error ≤ 0.1.
3. **Live routing**: `RoutePolicy` gains `live_specialists_allowed: bool` (replacing the
   `Literal[False]`), `router_version` (a promoted router model version, required when live is
   allowed), `canary` (traffic fraction in [0, 0.05] for `canary` state, [0, 1] for
   `production`, an allowlist of tenant pseudonyms for the internal canary step, and the
   deterministic assignment rule: hash of interaction id and policy id below the fraction).
   The chain planner builds candidates from the router's calibrated estimates: select the
   cheapest candidate whose suitability ≥ quality threshold and confidence ≥ router
   confidence threshold and OOD ≤ max, else foundation. Only `canary` or `production` state
   specialists are eligible for live traffic; `shadow` state keeps shadowing. The route
   decision records every candidate with its estimate, and `RouteSummary` in the response
   gains `specialist_served: bool` and `fallback_used` semantics as before.
4. **Deployment progression** (spec 15.2): promotion to `canary` requires an active policy
   naming the version with a canary fraction ≤ 5 % and a passed shadow report over at least
   the configured minimum sample with a non-inferior score delta; promotion to `production`
   requires a passed canary report (same aggregates computed over live specialist-served
   interactions) plus operator approval. Add `GET /v1/canary/reports?policy=<id>&since=`.
5. **Automatic rollback and expansion stop** (spec 15.3): a lifespan monitor evaluates the
   canary and production aggregates every N seconds against the policy's hard thresholds
   (validation failure rate, error rate, p95 latency, cost per successful outcome versus
   foundation, and a critical safety incident count of zero); breaching any threshold
   automatically disables the specialist for the environment through the existing
   disablement table (so it survives restart and propagates within a second), emits
   `deployment.changed.v1` with the reason, and records the measurement that triggered it. It
   never promotes. Manual re-enablement requires an operator note. Test with injected
   aggregates and with a live specialist wrapper that starts failing validation.
6. **Cost accounting for the business outcome** (spec 23 "apparent savings hide fallback
   cost"): every interaction records total cost including all attempts and shadow cost
   separately; the canary report shows cost per successful outcome for specialist-served
   versus foundation-served interactions in the same window and segment, with the paired
   bootstrap where items are paired by task and risk tier bins, and the reduction as a
   fraction. The milestone exit test asserts the numbers are computed correctly on a
   synthetic mix where the specialist is cheaper but sometimes falls back.
7. **Exit drills**: a rollback drill measuring time from an injected threshold breach to the
   specialist being disabled and the next request served by foundation (target under five
   seconds); the existing kill switch drill; and a "novel task" drill showing an out-of-
   distribution request goes to foundation with reason code `out_of_distribution`.
8. **Tests**: routing dataset contains no text; router training reproducible; calibration and
   OOD math on known inputs; the routing suite metrics on a synthetic held-out fold; live
   selection picks the cheapest qualifying candidate; canary assignment deterministic and
   within the fraction; shadow-state specialists never served live; promotion gates for
   canary and production; automatic disablement propagates and requires a note to re-enable;
   the `specialist` request mode now honoured only when the policy allows; every existing
   suite green; the cost report test above.
9. **Docs**: `docs/runbooks/canary-and-rollback.md`; update `docs/status.md` with the
   milestone 4 exit criteria table and measured numbers; update the shadow runbook and
   interaction walkthrough for the live path.

## Out of scope

Bandits and reinforcement routing (spec 13.4 explicitly later), distillation (milestone 5),
research (milestone 6), real providers, streaming. Do not modify `docs/spec/`,
`pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None. Logistic regression, calibration and ECE in plain Python.
