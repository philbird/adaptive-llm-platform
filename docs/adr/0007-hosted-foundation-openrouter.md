# ADR 0007: OpenRouter foundation and offline fixture replay

Status: accepted for slice R1, 2026-09-26. Owner decisions are in
[decisions.md](../decisions.md); model selection is in the
[foundation benchmark](../evidence/triage-foundation-benchmark-2026-09-26.md).

ManyFails candidate triage uses `anthropic/claude-haiku-4.5` through OpenRouter's
`/api/v1/chat/completions`. The adapter implements the injected `Provider` boundary using
the already locked `httpx`. Messages preserve their roles. Supplied retrieval chunks become
one final user message with the same text and citation markers as the fake provider.
Generation is non-streaming at temperature zero, with an output budget and the remaining
routing deadline. Provider-reported input/output, cached and reasoning counts are recorded;
missing optional counts remain null. Pricing is integer USD micros from the versioned route.

Every outbound request sets `provider.data_collection: "deny"`. It omits
`provider.require_parameters`, which prevented routing in the benchmark. Region `us` is the
recorded routing assumption from the owner decision, not an independently verified provider
residency guarantee. Headers identify the Adaptive LLM Specialisation Platform project.

The inference key comes only from `OPENROUTER_API_KEY`; its Settings field is excluded from
repr. An OpenRouter foundation without a key fails startup with
`openrouter_api_key_required`. Explicit provider injection supports offline tests/evaluation.
Gateway bearer keys can instead be configured by SHA-256 digest. Authentication hashes the
presented key and compares every configured fixed-length digest with `hmac.compare_digest`.
Existing synthetic literal keys remain supported. `make issue-key` prints a random key once
and a labelled, hashed stanza; config labels are not bearer credentials.

The HMAC master secret comes from `ADAPTIVE_SECRET` (hex or strict base64 decoding to at
least 32 bytes, excluded from Settings repr), ahead of the optional identity-file
`local_secret`. Explicit Settings injection remains available to tests. ManyFails has no
file secret. Missing/invalid secrets fail startup with `adaptive_secret_required` or
`invalid_adaptive_secret`. The public synthetic default is accepted only for local literal
keys; any hashed key or a non-local environment rejects it with `insecure_adaptive_secret`.
Operators provision and retain a private secret independently of bearer/provider keys.

The adapter suppresses task-local httpx/httpcore transport logs, which can otherwise contain
upstream headers. Failures cross the provider boundary only as `provider_timeout`,
`provider_rate_limited`, `provider_unavailable`, or `provider_invalid_response`. No raw
body, header, transport exception text or credential enters an event or returned exception.
Cancellation remains cancellation. Structured JSON fences are removed before validation.
Each provider lazily pools one HTTP client per running event loop, with environment proxy
settings disabled. `aclose` closes the calling loop's pool; serving shutdown, evaluation
completion (including failures) and recording completion call it before their loop ends.
Null/unknown finish reasons become `stop` only with nonempty content and no tool calls.

Ingress accepts one first-position system message, bounded to 8,000 characters within the
32,000 total. The ordinary processing and persistence redaction passes apply to it.
Draft 2020-12 schemas are bounded to 8,192 canonical UTF-8 bytes and checked with the locked
`jsonschema`. External references are refused; output validation uses a registry without
network retrieval. Schema errors are hard failures that use the ordinary fallback chain.
Route records/events contain only schema name and canonical SHA-256, never the schema.
The response format is encrypted beside retained messages and follows their retention and
deletion rules. Corrections are checked against it before writing and after redaction;
invalid targets return 422 `invalid_correction_target`. If schema redaction would change
constraints, content persistence fails closed while serving continues.

Mapped applications record classifier version `rules-1` and reason `application_task_map`.
Unmapped requests retain the existing RAG-only label, confidence, reason and
`placeholder-rag-flag-1` version for compatibility with existing consumers.

Cassette files are a deliberate exception for the approved public golden fixture and
synthetic safety fixtures. They are fixture artifacts, never a production recording mode.
Names are SHA-256 hashes of canonical outbound requests, including model, messages,
response format/schema, maximum tokens, temperature and the fixed collection policy.
Files contain only canonical content, usage, finish reason and latency; citations are
reconstructed from markers. No request bodies, headers, keys or raw upstream responses are saved.
Replay is the default and never calls a delegate on a miss. Recording requires
`OPENROUTER_LIVE=1`, a key and an explicit recorder invocation. Existing entries are retained.

The reviewer recorded all 38 cassettes on the host; they are retained unchanged.
Replay-dependent tests skip with exactly `openrouter cassette missing: <hash>` if an entry
is absent until the reviewer records it on the host.
The default tests block the real async HTTP transport, even when live environment variables
are present. The evaluation CLI also always replays; serving uses the real adapter.

The first foundation lock uses 30 public golden cases and eight synthetic injection cases,
without manufacturing a training dataset. It requires operator tenant grants and current
processing/evaluation permission, and uses the existing isolated serving runner and signed
control-store publication. Its specification pins the application, fixture set and tenant;
digests cover both fixture bytes and referenced system prompts. Exact field and normalized
product matches produce fractional scores. Golden disagreements are measurements; malformed
output and safety failures block the lock. This fixture gate is limited to foundation
measurement and does not relax the five-suite specialist promotion gate.

Dataset-backed five-suite evaluations can also name `fixture_set`, selecting golden and
safety files (and their prompt digests) without entering the fixture-only gate. Held-out,
retrieval and performance cases then come from the dataset. Builds name the same
`fixture_set` for golden decontamination, defaulting to `synthetic`; system instructions
and different output labels cannot dilute the comparison of user input against golden inputs.

No ManyFails application changes, specialist training, tool use, streaming, backfill or
additional provider are part of this decision. The reviewer host recording and foundation
lock are recorded in [status.md](../status.md).
