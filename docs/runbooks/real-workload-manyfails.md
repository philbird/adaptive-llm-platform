# Run ManyFails candidate triage

R1 serves the hosted foundation; it does not modify the ManyFails repository. Follow the
[integration contract](../integrations/manyfails.md). The policy permits processing,
retained logging, evaluation, human review and training for `manyfails`, with US residency
and 30-day retention. Specialist routing, shadow and canary remain off.

## Issue the caller key and configure the host

```sh
make issue-key TENANT=manyfails APP=research-sweep
```

The first output line after Make's command echo is the new key. Store it as ManyFails'
`ADAPTIVE_GATEWAY_KEY`; paste the printed JSON entry into `configs/identity/manyfails.json`,
replacing the all-zero hash placeholder. The label is not a valid key. The supplied
`synthetic-operator-key` is a local operator credential for the baseline run; replace it
before exposing a deployment. To issue an operator key, use the same command and add
`"key_class":"operator"` and `"dataset_tenants":["manyfails"]` to its hashed stanza.
Keep its entry under a different label so it does not replace the caller entry.

Set the host environment without copying the upstream key into a file in this repository:

```sh
export ADAPTIVE_IDENTITY_CONFIG=configs/identity/manyfails.json
export ADAPTIVE_POLICY_CONFIG=configs/policy/manyfails.json
export ADAPTIVE_ROUTING_CONFIG=configs/routing/manyfails.json
export ADAPTIVE_TASKS_CONFIG=configs/tasks/manyfails.json
# Supply OPENROUTER_API_KEY securely in this shell/service environment.
# Supply ADAPTIVE_SECRET securely: hex or base64 encoding of at least 32 random bytes.
# Supply ADAPTIVE_OPERATOR_KEY for a replaced operator credential.
```

The service reads `OPENROUTER_API_KEY` only from environment. Missing credentials fail
startup with `openrouter_api_key_required`. The base URL defaults to
`https://openrouter.ai/api/v1`; explicit Settings injection supports fake test transports.
No configuration JSON contains this key. Provision `ADAPTIVE_SECRET` from your secret store
for both serving and evaluation and keep it stable across restarts. It takes precedence
over optional `IdentityConfig.local_secret`; ManyFails has no file secret. Never reuse a
bearer key or provider key. Missing secrets return `adaptive_secret_required`; malformed
hex/base64 or fewer than 32 decoded bytes return `invalid_adaptive_secret`. The checked-in
synthetic default is rejected with `insecure_adaptive_secret` outside `local` or whenever
any configured key is hashed, including the ManyFails placeholder. Existing independent
payload-key and signing-key configuration still applies. Changing the HMAC secret changes
subject pseudonyms, content/replay hashes and legacy MACs; do not casually rotate it over
existing data. Explicit Settings secret injection is intended for controlled tests.

```sh
uv run --locked uvicorn adaptive_llm.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

Set ManyFails' `ADAPTIVE_GATEWAY_URL` to this reachable gateway URL. `/healthz` reports
`inference_enabled: true` only with inference mounted. Defaults store tenant/control data
under `.local`; stop the server before another process uses that same SQLite directory.

## Record on the reviewer host

The reviewer has recorded 38 entries in `tests/fixtures/cassettes/openrouter/`; preserve
them. To fill missing entries, run explicitly on the host with the approved golden fixture
and synthetic safety inputs only:

```sh
OPENROUTER_LIVE=1 make record-openrouter
```

This requires `OPENROUTER_API_KEY`, makes up to 38 real provider calls, and prints request
hashes only. Each file is named `<hash>.json` and contains canonical content, usage, finish
reason and latency. Recording preserves existing entries and resumes by filling misses.
Do not use the recorder for production traffic. Inspect canonical content before checking
in cassettes; no keys, headers, prompts or raw provider bodies belong in them.

Without this explicit command, tests and the evaluation CLI only replay and cannot call
the upstream provider. A missing test cassette skips with exactly
`openrouter cassette missing: <hash>`. The wrapper itself raises on a replay miss. A malformed
cassette fails, rather than skips. Live environment variables never enable live pytest calls.

```sh
UV_OFFLINE=1 make check integration
UV_OFFLINE=1 make check-without-training
```

## Lock the foundation baseline

With the config variables and `ADAPTIVE_SECRET` above still set, and after recording:

```sh
make evaluate SPEC=configs/evaluation/manyfails-triage.json DATA_DIR=.local/manyfails-evaluation
```

The CLI always replays OpenRouter cassettes, so this step does not need the upstream key.
It authenticates the operator, requires the `manyfails` tenant grant, the `research-sweep`
application grant and current evaluation permission, then runs 30 golden and eight critical
safety cases under `research-sweep`.
`rules-1` maps schema `triage` to `candidate_triage`, `en`, `low`. The API evaluator uses
its injected serving provider; use the CLI for reproducible cassette-based locks.

The evaluation file intentionally omits `evaluation_id`: each CLI invocation generates
UUIDv7. `dataset_version: manyfails-triage-1` names this fixture revision. Report digests
bind the actual golden/safety files and referenced prompt, including prompt-only changes.
No artificial dataset build or training data is required. Evaluation cases use private
in-memory storage with logging and training disabled; only signed aggregate reports and
the evaluation lifecycle event enter the control store.

A passing report has minimum paired sample 30, golden/safety coverage, zero schema/inference
failures and zero critical injection failures. `rubric_score` is the mean fraction of listed
fields that match (0..1); `assertion_pass_rate` is the fraction with all assertions passing.
The foundation is compared to itself; the CI should be `[0,0]`. Golden label disagreements
are reported, not forced to zero. This records foundation quality and is not specialist
promotion evidence. Existing five-suite promotion gates remain required for specialists.
For a triage specialist, set `fixture_set: "manyfails-triage"` on both its dataset build
and its five-suite evaluation, and omit `fixture_tenant_id` on the evaluation. Supply the
real dataset id/version: its held-out rows drive held-out, retrieval and performance suites;
golden/safety use the named fixture files. Lock a foundation comparison against that same
dataset and five-suite specification before evaluating the specialist. The fixture-only
lock above remains measurement evidence. Omitting `fixture_set` preserves the legacy
synthetic evaluation path; dataset builds default to the synthetic golden set.

Read `passed` and each gate in the printed report; a completed failing evaluation is still
a report and will not lock a baseline. Missing cassettes cannot produce a passing lock.
An existing lock returns 409 `baseline_already_locked` for another id. To replace it,
copy the specification to `.local`, add `"replace":true` and a nonempty `operator_note`,
and run with a fresh id. The existing lock remains until the replacement passes.

Artifacts are under `.local/manyfails-evaluation/evaluations/<id>/`; the separate control
database is `.local/manyfails-evaluation/control/local.sqlite3`. Record the report id,
fixture digest, golden rubric/assertion rates and safety counts in
[status.md](../status.md). Live latency/cost measurements belong in a separate host result;
cassette replay wall time is not hosted latency. The reviewer's first recording and locked
baseline numbers are in the status table.

## Read operational results

```sh
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:9464/metrics
```

The existing Prometheus endpoint reports bounded request status classes, route distribution,
fallback reasons, breakers and outbox health. Inspect generation records for provider-reported
usage and integer-micro costs at `openrouter-2026-09-26` (1,000 input and 5,000 output micros
per 1,000 tokens). Route records/events expose `response_schema_name` and
`response_schema_sha256`. Interaction records expose the trusted application, mapped task
and classifier version. Prompt, verdict and schema content never belong in operational
logs or metrics. Use the normal encrypted payload access path for authorized content review.

ManyFails human edits use the correction endpoint; the integration contract gives the
complete body, replay rules, error codes and its own-provider fallback on gateway 5xx.
