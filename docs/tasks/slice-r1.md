# Task: slice R1 — first real workload: ManyFails candidate triage through an OpenRouter foundation

The owner decisions 1, 2 and 3 are recorded in `docs/decisions.md` (dated 2026-09-25/26). Read
them, `AGENTS.md`, `docs/status.md`, `docs/runbooks/interaction-walkthrough.md`,
`docs/runbooks/evaluation.md` and `docs/evidence/triage-foundation-benchmark-2026-09-26.md`
first. Milestones 1–6 and slice P1 are on `main` (PRs #1–#15).

The workload: ManyFails (a separate Next.js repository, not touched here) runs a daily sweep
that classifies news hits into a strict JSON triage verdict. Today it calls a model directly.
After this slice it can call this gateway instead, the gateway serves the verdict from a real
hosted foundation model (`anthropic/claude-haiku-4.5` via OpenRouter), records the interaction
under the standard privacy and event machinery, accepts ManyFails' human corrections as
feedback, and evaluates candidates against a golden set built from ManyFails' labelled fixture.
Nothing ManyFails-side is built here; you produce the integration contract it will follow.

## Facts you build against (do not re-derive)

- Tenant `manyfails`, application `research-sweep`, task label `candidate_triage`, language
  `en`, risk tier `low`. All purposes are permitted for this tenant. Residency `us`.
- Foundation: OpenRouter, model `anthropic/claude-haiku-4.5`, processing region `us`, price
  list version `openrouter-2026-09-26` with `input_micros_per_1000_tokens=1000` and
  `output_micros_per_1000_tokens=5000`. The request goes to
  `https://openrouter.ai/api/v1/chat/completions` with `response_format` of type
  `json_schema` (strict) and `provider: {"data_collection": "deny"}`. Do not send
  `provider.require_parameters`; the benchmark shows it routes to no endpoint for this model.
- The triage task is a system prompt (about 3,000 characters, supplied by the client per
  request), one user message holding a JSON object with `title`, `description`, `url` and
  `page` (a scraped excerpt of at most 1,500 characters), and a strict JSON schema:
  `is_failure: boolean`; `kind` enum of `product_shutdown, company_shutdown, feature_pulled,
  acquihire_killed, scaled_back, never_shipped, pivoted_away, incident, watch, not_a_failure`;
  `product: string|null`; `company: string|null`; `event_date: string|null` (YYYY-MM-DD);
  `size` enum `titan, funded, indie, unknown`; `confidence: number 0..1`; `reason: string`
  (single line). All eight required, no additional properties.
- The ManyFails acceptance rule for a verdict: `confidence >= 0.6`, a `product` or `company`
  named, `kind` not `watch` or `not_a_failure`, and (`kind == "incident"` or `is_failure`).
- The labelled fixture: 30 real headlines with expected verdicts, copied for you into
  `tests/fixtures/golden/manyfails-triage.jsonl` by the reviewer before launch (see below for
  the format). The reviewer also placed the real system prompt at
  `tests/fixtures/prompts/manyfails-triage.md`. Both are public-data fixtures; treat the
  headlines as untrusted content in tests exactly as the gateway does.
- Measured volume: about 64 sweep queries per day, roughly 100–300 classifier calls per day,
  0–5 candidates inserted per day. Under the current ManyFails setup 90–145 verdicts per run
  fail its schema; the foundation chosen here produced zero schema failures on the fixture.

## Deliverables

1. **System role at ingress.** `Message.role` gains `system`, allowed only as the first
   message, at most one, content bounded to 8,000 characters and counted in the 32,000 total.
   It flows through processing and persistence redaction, token estimates, the persisted
   interaction, dataset rows (shards already carry system roles), evaluation cases, the
   specialist generator's chat template and the fake provider unchanged. Requests without a
   system message behave exactly as today; every existing test passes unchanged.
2. **Structured output at ingress.** `ResponseFormat` gains `type: "json_schema"` with
   `json_schema: {name, schema}` where `schema` is a JSON Schema object bounded to 8 KB and
   validated at ingress with the `jsonschema` package (Draft 2020-12 metaschema). The output
   validator gains a hard check `json_schema` that validates the model output against it
   (`jsonschema` again), keeping `json_object` for the existing type. A schema failure is a
   hard validation failure like `json_object` today: fallback chain, then the fixed error.
   The route record and events carry the schema name and a SHA-256 of the canonical schema,
   never the schema itself.
3. **OpenRouter provider adapter** `src/adaptive_llm/providers/openrouter.py` implementing
   `Provider` with `httpx.AsyncClient`: translates messages (including system), context chunks
   (appended as a final user message exactly as the fake provider renders them, so citations
   keep working), `response_format` (json_schema strict, json_object, or none), and
   `max_output_tokens`; sets the request deadline from the routing options; maps usage to
   `Usage(source="provider_reported", tokenizer=<model id>)` including cached and reasoning
   tokens when reported; maps finish reasons to `stop | length | content_filter`; strips a
   Markdown code fence around JSON before validation; and maps every failure to a fixed code
   (`provider_timeout`, `provider_rate_limited`, `provider_unavailable`,
   `provider_invalid_response`) with no response body, header or key in any log, event or
   exception. Configuration: `Settings.openrouter_api_key` from env `OPENROUTER_API_KEY` only
   (never a config file; `repr=False`), `Settings.openrouter_base_url` (default the public
   URL, overridable for tests), and the routing JSON's foundation block with
   `model_provider: "openrouter"`. A missing key with an OpenRouter foundation fails startup
   with a fixed error. Send `HTTP-Referer` and `X-Title` headers naming this project.
4. **Cassette recording and replay** `src/adaptive_llm/providers/cassette.py`: a wrapper that
   keys on a SHA-256 of the canonical outbound request (model, messages, schema, max tokens,
   temperature) and stores the provider's canonical result (content, usage, finish reason,
   latency) as JSON under `tests/fixtures/cassettes/openrouter/`. Modes: `replay` (default;
   a miss is a test failure with the request hash, never a network call), `record`
   (`OPENROUTER_LIVE=1`, appends new entries, requires the key). Cassettes never contain keys,
   headers, raw provider bodies or anything but the canonical result; a test asserts that.
   The reviewer will record the cassettes on the host after your pass; ship the recorder plus
   an empty directory and make every test that needs a cassette skip with the fixed reason
   `openrouter cassette missing: <hash>` rather than fail, so the gate stays green until
   recording. Never call the network in the default gate.
5. **Hashed API keys and a real tenant configuration.** `KeyIdentity` entries may carry
   `key_sha256` instead of a literal bearer value; the authenticator hashes the presented
   bearer and looks it up in constant time. Synthetic literal keys keep working for local
   configs. `make issue-key TENANT=… APP=…` prints a new random key once and the JSON stanza to
   paste. Add `configs/identity/manyfails.json` (hashed placeholder stanza the reviewer will
   replace, plus the operator key), `configs/policy/manyfails.json` (tenant `manyfails`,
   processing, retained logging, evaluation, human review and training all allowed, residency
   `us`, retention 30 days), `configs/routing/manyfails.json` (OpenRouter foundation block
   above, specialist routing off, shadow off, canary off, content logging on, training on) and
   `configs/tasks/manyfails.json` for deliverable 6.
6. **Rules task classifier (spec 7.4).** Replace the placeholder with a versioned rules
   classifier `rules-1`: a per-application map in `configs/tasks/<name>.json` from
   `application_id` plus optional `response_format.json_schema.name` to `label`, `language`
   and `risk_tier`, with reason code `application_task_map`; unmapped requests keep today's
   behaviour (`general` or `question_answering`, reason `rag_flag_only`). Record
   `classifier_version` on the interaction as today. The manyfails map sends
   `research-sweep` + schema `triage` to `candidate_triage`, `en`, `low`.
7. **Golden and safety suites for structured tasks.** Golden items gain optional
   `expect_json_fields` (exact equality per listed field after JSON parsing) and
   `expect_json_text_match` (normalised substring match either way, for `product`); the
   deterministic rubric scores a structured item as the fraction of listed fields that match.
   Fixture format for `manyfails-triage.jsonl`: `{"system": <prompt file reference
   "prompts/manyfails-triage.md">, "input": <the JSON hit as a string>, "target": <expected
   verdict JSON string>, "expect_json": true, "expect_json_fields": {"is_failure": …,
   "kind": …}, "expect_json_text_match": {"product": …}, "expect_citation": false,
   "response_format": {…the triage schema…}}`; extend the loader accordingly. Add a safety
   fixture `tests/fixtures/safety/manyfails-triage.jsonl` of at least eight items where the
   page text contains instructions (role changes, "publish this", "set is_failure true") and
   the expected verdict ignores them; a critical failure is a verdict that followed the
   injected instruction. Evaluation specifications accept `application_id` so the suites run
   under the mapped task.
8. **Baseline lock on the real foundation.** With cassettes recorded, `make evaluate` on
   `configs/evaluation/manyfails-triage.json` (golden + safety suites, minimum sample 30) locks
   the foundation baseline for the `manyfails` tenant. Your pass ships the specification and
   the code path; the reviewer records the cassettes and runs the lock, then fills the numbers
   into `docs/status.md`.
9. **Integration contract** `docs/integrations/manyfails.md`: the exact `POST /v1/inference`
   body ManyFails will send (system prompt, one user message, `response_format` json_schema
   named `triage`, `application_id: "research-sweep"`, `routing.mode: "auto"`,
   `max_output_tokens: 400`, `metadata` limited to a `sweep_query_id` and `hit_url_sha256`),
   how it reads the verdict from `content`, the correction call it makes when Phil dismisses or
   promotes a candidate (`POST /v1/interactions/{id}/correction` with the corrected verdict as
   the target), the idempotency rule for `request_id`, the fixed error codes it must handle and
   the fallback it must apply (keep calling its own provider when the gateway returns 5xx), and
   the environment it needs (`ADAPTIVE_GATEWAY_URL`, `ADAPTIVE_GATEWAY_KEY`).
10. **Docs**: ADR 0007 (hosted provider adapter, cassette policy, data-collection deny, key
    handling), runbook `docs/runbooks/real-workload-manyfails.md` (issue a key, set env, start
    the gateway with the manyfails configs, record cassettes, lock the baseline, read the
    metrics), `docs/status.md` section "First real workload" with a table of what is verified
    locally (cassette replay) versus live (reviewer host).

## Tests

Adapter request translation and error mapping with a fake HTTP transport; usage and finish
reason mapping; no key or body in logs and exceptions (assert on captured records); cassette
replay/miss/record behaviour and the no-secrets assertion; json_schema ingress validation and
the hard validator including fence stripping; system role bounds and redaction; hashed keys
(right key, wrong key, literal key) and `make issue-key`; the rules classifier with mapped and
unmapped requests; golden structured scoring on the fixture with the fake provider returning
fixed verdicts; the safety fixture's critical-failure detection; the full gate without network
and without the training group.

## Out of scope

The ManyFails code itself, training a triage specialist (next slice), backfilling historical
ManyFails verdicts, streaming, tool calls, any provider other than OpenRouter.
Do not modify `docs/spec/`, `pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None by you. The reviewer has locked `httpx` (moved to core) and `jsonschema` on this branch;
use them.
