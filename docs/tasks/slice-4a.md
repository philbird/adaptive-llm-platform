# Task: slice 4a — shadow mode, validators, bounded fallback, circuit breakers and kill switch

Milestone 4 begins. Implement the safe-routing half that does not yet expose specialist output
to users, as described in the specification (`docs/spec/`, sections 7.5, 7.7, 12.1 layer 9,
13.1–13.4, 15.2 step 3, 15.3, 15.4, 20.4, 21.2 step 11 "routing in shadow mode, then
validation/fallback"). Read `AGENTS.md`, `docs/status.md`, `docs/runbooks/interaction-walkthrough.md`
and `docs/runbooks/training-and-promotion.md` first. Milestones 1–3 are on `main`. Everything
remains local; the only specialists are registered fake or tiny real adapters.

## Deliverables

1. **Route policy contract and store**: `RoutePolicy` in `contracts.py` per spec 13.2 (policy
   id, eligible specialist versions, quality threshold, router confidence threshold, OOD
   threshold, max attempts, hard fallback reasons, foundation fallback deployment, shadow
   enabled, per-tenant and per-task enablement, kill switch flag), versioned and immutable in
   the control database (migration 0009 `route_policies` with an `active` pointer per
   environment). `POST /v1/route-policies` (operator) creates a version; `POST
   /v1/route-policies/{id}/activate` switches the pointer with an operator note and emits
   `deployment.changed.v1` with a `route_policy` deployment id, so policy rollback is
   independent of model rollback (spec 15.3). Only versions in state `shadow`, `canary` or
   `production` may be listed as eligible specialists; this slice permits `shadow` only.
2. **Shadow execution**: when the active policy has shadow enabled and lists a specialist in
   state `shadow` whose tenant set includes the request's tenant, the gateway serves the
   foundation response exactly as today and, after the response is returned, runs the
   specialist on the same redacted canonical input and approved RAG context off the request
   path (bounded queue, never blocking serving, never affecting the response). The shadow
   attempt is recorded as a `GenerationAttempt` with `attempt_number` 0 and a new
   `shadow: bool` field, persisted and emitted as `generation.completed.v1`/`failed.v1`, never
   returned to the client, never eligible for replay, and its output stored only as a payload
   ref when content logging is allowed. Shadow work is skipped, with a counter, when the
   specialist is unhealthy or the shadow queue is full. Cost of shadow attempts is accounted
   separately (`shadow_cost_micros` metric) and never charged to the interaction's response.
3. **Shadow comparison**: a `ShadowComparison` record per interaction (scores from the
   deterministic judge on both outputs, citation precision/recall of each against supplied
   chunks, token and cost deltas, latency delta, validator outcomes for the specialist output)
   written to the control database and summarised by `GET /v1/shadow/reports?policy=<id>&since=`
   (operator) as aggregate metrics only: coverage, specialist validation pass rate, mean score
   delta with paired bootstrap CI reusing `evaluation/stats.py`, per critical segment, and
   cost delta. No content in reports.
4. **Validator expansion** per spec 7.7 in `src/adaptive_llm/validation/`: schema/JSON, citation
   presence and validity, groundedness (every claim sentence must share at least one
   5-gram with a supplied chunk or be a citation-free connective, a deterministic heuristic
   marked as such), language check, repetition and truncation checks, tool-call allowlist
   (empty allowlist now), and domain deterministic tests loaded from
   `configs/validation/<application>.json`. Each check has a name and version; results go
   into `Validation.checks`.
5. **Bounded fallback chain**: `RoutingOptions.mode` gains `"specialist"` (only honoured when
   the active policy allows live specialists, which this slice does not, so it must fall back to
   foundation with reason `live_specialists_disabled`). Implement the chain machinery now so
   4b only flips the flag: attempts are numbered, each recorded, `max_attempts` from the
   policy is enforced, `hard_fallback_on` reasons trigger the next candidate, the request
   deadline is respected (no attempt starts if success before the deadline is implausible,
   using the candidate's estimated latency), and failed specialist output is never returned.
   Exercise the chain in tests through an injected policy that allows a fake "live" specialist,
   with a test proving the public configuration cannot enable it in this slice.
6. **Circuit breakers** per deployment: sliding-window error rate, validation failure rate and
   p95 latency thresholds from the policy open the breaker; open breakers remove the
   deployment from candidates and shadow selection; half-open probes after a cooldown; state
   exposed in metrics and `/healthz`. Tests drive the breaker through closed → open →
   half-open → closed.
7. **Kill switch**: `POST /v1/route-policies/kill-switch` (operator note) disables all
   specialist routing and shadow execution immediately for the environment, persisted in the
   control database and honoured by every process on its next request without restart (poll
   the pointer at most every second, cache otherwise); `DELETE` re-enables with a note. Per-tenant
   and per-task disablement through the same table. Kill switch engagement emits
   `deployment.changed.v1` and is exercised in a drill that measures time from request to
   effect (target under two seconds) and records the number in the runbook.
8. **Metrics and health**: route distribution by deployment, shadow coverage, shadow queue
   depth and drops, breaker state per deployment, kill switch state, fallback reasons. Labels
   bounded as before.
9. **Tests**: shadow never changes the response, replay or cost; shadow output never reaches
   the client or the replay store; content logging disabled means no shadow payload ref;
   shadow skipped when the breaker is open; comparison aggregates contain no content;
   validator unit tests per check including groundedness false positives on connective
   sentences; fallback chain ordering, deadline respect and max attempts; breaker state
   machine; kill switch effect within two seconds across two app instances sharing a data
   directory; policy activation and rollback independent of model state; a drill for the
   kill switch under load.
10. **Docs**: `docs/runbooks/shadow-and-kill-switch.md`; update the interaction walkthrough
    with the shadow path; `docs/status.md` milestone 4 row.

## Out of scope

Live specialist responses to users, canary traffic percentages, router training and
calibrated routing, bandits, rollback automation from thresholds (slice 4b). Do not modify
`docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None.
