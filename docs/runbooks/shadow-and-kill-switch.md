# Local shadow routing and emergency disablement (slices 4a–4b)

Foundation is the default public serving route. Slice 4b adds opt-in calibrated live routing;
see [canary and rollback](canary-and-rollback.md) for dataset, router and deployment approvals.
A registered fake or tiny real adapter
in state `shadow` can run after the HTTP response body has been sent. First create and promote
an adapter using [training and promotion](training-and-promotion.md), then start the gateway
against that same data directory. No external provider or service is required.

## Create, activate and roll back a route policy

Use the configured operator credential. Each server-generated `policy_id` identifies one
immutable policy version; creating the same id again returns 409. Tenant and task maps are
explicit allowlists: absent entries are disabled. Task labels currently come from the RAG flag
(`question_answering` or `general`), with language `en` and risk `medium`.

```sh
curl http://127.0.0.1:8000/v1/route-policies \
  -H 'Authorization: Bearer synthetic-operator-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "eligible_specialist_versions": ["MODEL_VERSION"],
    "foundation_fallback": "fake-foundation-local-1",
    "shadow_enabled": true,
    "tenant_enabled": {"synthetic-a": true},
    "task_enabled": {"question_answering": true, "general": true},
    "max_attempts": 2,
    "quality_threshold": 0.9,
    "router_confidence_threshold": 0.85,
    "ood_threshold_max": 0.15
  }'

curl -X POST http://127.0.0.1:8000/v1/route-policies/POLICY_ID/activate \
  -H 'Authorization: Bearer synthetic-operator-key' \
  -H 'Content-Type: application/json' \
  -d '{"reason":"SYNTHETIC reviewed shadow policy"}'
```

Replace `MODEL_VERSION` and `POLICY_ID` with the returned identifiers. Creation and activation
both require every listed specialist to be in `shadow`, `canary` or `production`; candidate,
approved, deprecated and revoked specialists are refused. The shadow loader only executes
`shadow` versions. The live loader only executes `canary`/`production` versions and requires an
approved router, live policy and successful calibrated admission. Both verify current
model state and tenant membership before each execution. Verified providers are cached by
`(version, artifact_digest)` in an eight-entry least-recently-used cache; artifact files and
the base/adapter are loaded on a cache miss. A changed state or digest evicts the old entry,
and state/digest are re-read after a cold load before it can enter the cache. Cached providers
use their already verified in-memory weights. The first healthy
eligible specialist in policy order runs; at most one comparison is produced per interaction.

To roll back routing, activate a previously created policy id. This changes no model states,
model deployment pointers or evaluation approvals. Migration 0009 stores immutable policies,
the environment's active pointer and independent disablement settings, an actor/note audit
history, shadow observations and comparisons in `control/<environment>.sqlite3`. Activation
and switch changes commit their audit and `deployment.changed.v1` events together; these events
use deployment id `route_policy`. Global activation/disablement needs grants for all configured
environment tenants; a tenant-specific switch needs that tenant's grant.

`live_specialists_allowed` defaults to false. True requires `router_version` naming an approved
`router-logistic-v1` model; otherwise creation fails. Without live permission, a request with
`routing.mode="specialist"` uses foundation with `live_specialists_disabled` in its route and
metrics. With live permission, `auto` and `specialist` both obey the fraction, allowlist and
admission controls. The production chain planner and its injected test boundary exercise quality/confidence/OOD
admission, residency and cumulative cost constraints, failed-output suppression, duplicate
candidate removal, hard fallback reasons, deadlines and `max_attempts`. It is not an HTTP
configuration option. A candidate's estimated latency must fit the remaining overall deadline;
skipped candidates do not consume an attempt. Invoked attempts start at 1, and the response's
live cost includes all invoked attempts. Shadow cost is always separate.

## Shadow data and reports

The worker receives the same processing-redacted messages and approved exact-version RAG
chunks used by the foundation. An ASGI background callback submits only after response delivery.
The queue holds at most `Settings.shadow_queue_capacity=32` jobs plus one active job. Submission
uses `put_nowait`; full queues, disabled controls and unavailable specialists increment drop
counters. Loading, generation, validation, judging and persistence run on a separate worker
thread. An unavailable control store fails closed for specialists and preserves foundation
serving. Queued work rechecks the active policy, kill switch, tenant/task settings, processing
permission and residency before execution. Replays create no new shadow work.

Attempts use `attempt_number=0, shadow=true`, share the original interaction/trace ids, and
emit `generation.completed.v1` or `generation.failed.v1` through the tenant outbox. They never
replace `final_attempt_id`, enter the response or replay, or become dataset production targets.
Output references require both the original and current content-logging permission, successful
persistence redaction, an active undeleted parent and unexpired retention. Payloads use the
existing cipher with AAD field `shadow_output.<attempt_id>`. Deletion/retention clears them with
the other interaction payloads. Failed redaction leaves no output ref or hash and does not
change the served response. Failure codes and events contain no provider bodies or text.

```sh
curl --get http://127.0.0.1:8000/v1/shadow/reports \
  -H 'Authorization: Bearer synthetic-operator-key' \
  --data-urlencode 'policy=POLICY_ID' \
  --data-urlencode 'since=2026-09-24T00:00:00Z'
```

Only aggregate data is returned. Coverage is completed comparisons divided by new successful
foundation interactions matching the policy's enabled tenant/task maps while shadow is enabled;
disabled/unhealthy/full-queue opportunities remain in the denominator. The filter uses UTC
observation time. Reports include validation pass rate, paired score delta and a seeded 95%
bootstrap CI using the evaluation statistics module, mean token/cost/latency deltas, citation
precision/recall, and segments for keyed tenant, task, language, risk and citation presence.
Empty segments have no invented score or CI. Generation failures have score zero and failed
validation. Comparisons record `judge_version="shadow-chunk-overlap-1"`: supplied whole chunks
are the expected facts, with blinded answer order. This coarse grounding proxy rewards echoing
and can penalise legitimate paraphrases. It is not the evaluation rubric, a human quality judgment
or offline promotion approval. The report's score deltas and intervals describe this proxy only.
Comparisons retain every hard and advisory validation outcome; the report's validation pass
rate counts only hard failures. No request/output/chunk text is stored in comparisons or reports.

Validation checks carry a name, version and severity: nonempty output, JSON object and configured key
requirements, citation presence and ids, groundedness, language, repetition, truncation, empty
tool allowlist, and domain rules from `configs/validation/<application>.json`. Groundedness is
explicitly a heuristic: each claim sentence must share a normalized five-word sequence with
supplied evidence. Only exact, citation-free connective sentences such as “However.” are exempt.
Without supplied chunks the grounding check is inapplicable. Language currently checks Latin
script for the English-only local configuration; it cannot distinguish all Latin-script languages.
Domain rules support required/forbidden text and required JSON keys. These checks are not a
general JSON Schema engine or a calibrated language/safety model.

`Validation.passed` depends only on hard checks. By default, `non_empty`, `citation_ids`,
`json_object`, `tool_allowlist`, domain `forbidden_text` and domain `required_json_key` are hard.
`citation_required`, `groundedness`, `language`, `repetition`, `truncation` and domain
`required_text` are advisory. Advisory failures remain in the attempt and comparison without
triggering fallback or opening a validation breaker. A nonempty text response that reaches the
requested token budget returns 200 with `finish_reason="length"`; malformed JSON remains a hard
failure when a JSON object was requested.

An operator can override severities for one application using the `severity` map in its
validation JSON. For example, add `"severity": {"groundedness": "hard", "truncation": "advisory"}`
to `configs/validation/support-assistant.json`. Domain keys use the emitted name,
such as `domain.synthetic_domain_marker`. Only builtin names and domain tests declared in
that application's file are accepted; misspelled/unknown names fail configuration validation.

## Breakers, metrics and kill switch

Each deployment has a process-local, bounded sliding window (up to 1,000 samples). Defaults are
60 seconds, at least five samples, error/validation failure rates of 0.5, p95 latency above
5,000 ms, and a 30-second cooldown. Configure these under the policy's `breaker` object.
Open deployments are excluded; after cooldown the next admission allows one half-open probe.
A successful timely probe closes and clears the window; a failed probe reopens it. Breakers
reset on restart; persistent global/tenant/task controls do not.

```sh
curl -X POST http://127.0.0.1:8000/v1/route-policies/kill-switch \
  -H 'Authorization: Bearer synthetic-operator-key' \
  -H 'Content-Type: application/json' \
  -d '{"reason":"SYNTHETIC emergency specialist disablement"}'

curl -X DELETE http://127.0.0.1:8000/v1/route-policies/kill-switch \
  -H 'Authorization: Bearer synthetic-operator-key' \
  -H 'Content-Type: application/json' \
  -d '{"reason":"SYNTHETIC reviewed recovery"}'
```

Add either `"tenant_id":"synthetic-a"` or `"task":"question_answering"` to the same body for
scoped disablement/re-enablement. Omitting both targets the environment. The immutable policy's
own `kill_switch=true` also disables specialists; changing that flag requires activating another
version. Clearing an environment switch does not clear scoped switches.

Slice 4b also accepts `"specialist_version":"MODEL_VERSION"` as an exclusive scope. Automatic
rollback writes that environment-wide version disablement with its triggering measurement and
event. Manual re-enablement requires a nonblank reason. Shadow reports now expose `passed` using
the policy's minimum sample/non-inferiority thresholds, overall and per critical segment. The
canary promotion gate checks the exact specialist under the active policy. A shadow-state
specialist cannot serve live even when the policy has enabled live traffic.

Every process uses a pointer cache lasting at most one second. Local mutations invalidate it
immediately; the next request in another instance refreshes after that interval. Queued work
also rechecks controls. Already running native tensor operations finish at their cooperative
boundary and cannot be forcibly interrupted. Shutdown discards queued jobs and awaits the
active worker. Shadow generation has a 30-second timeout; native loading/compute can outlast
the coroutine deadline. The queue is intentionally ephemeral; a restart may lose comparisons.
Tenant attempt/outbox persistence and control comparison publication are separate transactions,
so a control write failure can leave an attempt without a comparison; coverage exposes that gap.

`/healthz` preserves truthful `inference_enabled` and adds `circuit_breakers`, `kill_switch`
and `shadow_queue_depth`. In-process metrics add `route_distribution`, `fallback_reasons`,
`shadow_opportunities`, `shadow_completed`, `shadow_drops`, `shadow_cost_micros`,
`shadow_failures`, `shadow_coverage`, `shadow_queue_depth`, `breaker_state`, `kill_switch`
and `validation_advisory_failures`. The latter counts each failed advisory check on serving
and shadow attempts, labelled by `check_name`; replay adds no counts. Labels are bounded to
the nine builtin checks, at most 32 configured domain names across applications, and `other`.
Breaker gauge values are closed=0, open=1, half-open=2. Deployment labels are capped at 64
plus `other`; reason labels use a fixed vocabulary. No request, trace, prompt or path labels.

## Local verification and limits

```sh
make contracts
make check integration
uv run --locked pytest tests/drills/test_routing_kill_switch.py -m drill -s
make ci
```

The drill on 2026-09-24 measured **1.006 seconds** from the kill-switch request through a
second app instance's observed effect (repeat: **1.014 seconds**), while 100 concurrent-load inference requests all
succeeded. Ten subsequent requests started zero shadows. This meets the local target of two
seconds; it is not staging/production acceptance.

In a network-restricted sandbox, prefix commands with
`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/adaptive-llm-uv-cache` to use the already provisioned
environment without attempting a package build download. The SBOM audit still needs network.
If that step cannot resolve its audit service, run each later CI gate explicitly:

```sh
uv run --locked pytest tests/security tests/load -s
uv run --locked pytest tests/drills -m drill -s
uv run --locked pytest -m smoke -s
make check-without-training
```

Slice 4b implements live specialist responses, canary fractions, learned/calibrated routing and
automatic rollback; see [canary and rollback](canary-and-rollback.md). Bandits and external
transport remain later work. No dependency was added.

Initial local results and reviewer verification on 2026-09-24 (local commands used the two
environment assignments above):

| Command | Result |
| --- | --- |
| `make contracts` | Schemas and OpenAPI regenerated; schema consistency tests passed |
| `make check integration` | Formatting, Ruff, strict mypy (63 files), 162 unit/contract passed + one GPU skip, 118 integration passed |
| `make ci` | Reviewer ran it on the host on 2026-09-24: every gate passed, including SBOM audit; 330 tests, six drills, kill-switch effect 1.01 s |
| `uv run --locked pytest tests/security tests/load -s` | 43 passed; 1,000/1,000 event correlation; p95 overhead 2.110 ms |
| `uv run --locked pytest tests/drills -m drill -s` | Six passed; kill switch repeat 1.014 s |
| `uv run --locked pytest -m smoke -s` | Two passed; real CPU lifecycle 2.970 s |
| `make check-without-training` | 321 passed, nine expected optional-stack/GPU skips |

The pass-1 corrections passed full `make ci` on 2026-09-24, including the SBOM audit with no
known vulnerabilities. The updated suite collected 345 tests: 176 unit/contract passed plus one
GPU skip, 119 integration passed, 43 security/load passed and six drills passed. Both smoke tests
passed again; without the optional training stack, 336 passed with nine expected skips.
Kill-switch effect was **1.010 s**, with 100 successful concurrent-load requests and zero
post-effect shadow starts. Event correlation was 1,000/1,000 with 5,000 events; normal p95
overhead was 2.741 ms. The real CPU smoke completed in 3.832 s.

Focused development checks also used `uv run --locked ruff format src tests`,
`uv run --locked ruff check src tests --fix`, `uv run --locked mypy src/adaptive_llm`, and
`uv run --locked pytest tests/unit/test_validation.py tests/unit/test_circuit_breakers.py tests/unit/test_fallback_chain.py tests/integration/test_shadow_routing.py -q --tb=short`.
Test updates preserve privacy assertions while grounding the synthetic PII response, restore
200/`length`/one-token budget behaviour, and include the new health fields/migration. No tests or
checks were disabled. `git diff --check` passed.
