# Slices 1a–1b: one interaction, from serving to persistence

This walkthrough uses only the synthetic keys, policies, prices and documents checked into
the repository. All serving components run in the application process, with one SQLite file
at `.local/local.sqlite3` by default. Database opening and migrations happen during application
lifespan startup; importing the module or constructing an app creates no files. There is no
external provider call or exporter. The
health stage label remains `slice-1a` for compatibility; persistence is now enabled.

Start the local gateway with `make dev`. `/healthz` reports
`{"status":"ok","stage":"slice-1a","inference_enabled":true}`. Setting
`Settings(inference_enabled=False)` omits the inference route and reports false.

Submit this synthetic request:

```sh
curl http://127.0.0.1:8000/v1/inference \
  -H 'Authorization: Bearer synthetic-key-a' \
  -H 'X-Subject: synthetic-caller-1' \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "synthetic-walkthrough-1",
    "application_id": "support-assistant",
    "messages": [{"role": "user", "content": "SYNTHETIC: unused items receipt return window for order ORD-12345; sk-synthetic123456789; 4111 1111 1111 1111"}],
    "rag": {"enabled": true, "index_id": "synthetic-kb"},
    "max_output_tokens": 128,
    "routing": {"mode": "foundation", "max_cost_micros": 20000, "deadline_ms": 5000}
  }'
```

1. **Authenticate and reserve the request.** `LocalAuthenticator` resolves the bearer key
   through `configs/identity/local.json` to tenant `synthetic-a`, environment `local` and the
   allowed application `support-assistant`. `X-Subject` is optional; when supplied its value
   is HMAC-SHA256 pseudonymised with the subject-purpose key and tenant scope. A missing subject is
   recorded as null. The raw header is never emitted. Missing/invalid keys return 401 and
   disallowed applications return 403. Body metadata cannot override this identity.
   The service atomically reserves `(tenant, application, request_id)` using an HMAC of the
   canonical validated body, including its defaults. JSON key order does not affect replay.
   `Settings.secret` defaults to the synthetic secret in the configured identity JSON and can
   be explicitly injected. Lifespan startup constructs a single `Keyring` and injects it into the
   authenticator, retriever and service. At startup it derives five keys with
   `HMAC-SHA256(secret, purpose_label)`, using labels `subject-pseudonym-v1`, `input-content-v1`,
   `output-content-v1`, `query-content-v1` and `replay-fingerprint-v1`. Each subsequent HMAC uses
   the appropriate derived key. A custom authenticator plus an explicit secret requires no local
   identity file and does not construct `LocalAuthenticator`.
2. **Decide policy and redact.** `LocalPolicyEngine` reads `configs/policy/local.json` at
   startup. Known tenants may process requests; unknown tenants are denied. Content logging,
   evaluation, human review and training default to false. The processing pass replaces the
   synthetic API key and card with `[REDACTED]`, recording one `api_keys` and one `card_numbers`
   match in `processing_redaction_counts`. It retains `ORD-12345` and other business identifiers.
   Every client metadata value goes through the same redaction patterns; counts aggregate across
   messages and metadata. Metadata keys retain their validated identifiers. Metadata remains
   untrusted for identity and policy and is not included in events or model context.
   Card candidates must pass a Luhn checksum; this is a local heuristic. A separately versioned
   persistence pass prepares hashes before each relevant event, as described below. Retrieval and generation continue
   to receive the processing-path output, including business identifiers.
3. **Start and retrieve.** New UUIDv7 interaction and trace ids correlate all records.
   `interaction.started.v1` is emitted after policy. A placeholder labels RAG requests
   `question_answering` and other requests `general`, with classifier version
   `placeholder-rag-flag-1`, confidence `0.5` and reason code `rag_flag_only`. This only describes
   the RAG flag; task classification is deferred to a later slice. `LocalRetriever` uses the last
   redacted user message as the query. Before lexical scoring it filters the injectable
   `tests/fixtures/documents.json` by tenant, environment, application ACL, index id and the
   policy's residency. A chunk's region must exactly match that residency. Neither tenant B's
   chunk nor a `eu-west` chunk reaches scoring for this `local` request. Positive lexical overlap supplies
   `refund-window` from `synthetic-refund-policy`, document version `synthetic-1`, index version
   `synthetic-index-1`. Each fixture row carries the immutable version of its index snapshot;
   mixed versions of the same tenant/environment/region/index are rejected at startup.
   The retriever records up to ten candidates and supplies at most three chunks within 2048
   fake whitespace tokens. Every candidate records its retrieved rank, supplied flag, context
   position (zero-based or null), token count and SHA-256 content hash. The query hash is
   HMAC-SHA256 of the persistence-redacted query with the derived query key. The gateway prepares
   this hash before emitting the retrieval record; the stored record retains the identical hash.
   Neither query nor chunk content appears in the event.
   Disabled RAG skips the retriever and retrieval span entirely, emits no retrieval event and
   leaves `Interaction.retrieval_run_id` null. The provider receives empty context. Direct calls
   to `LocalRetriever` with disabled RAG are rejected without creating a run; there is no disabled
   index sentinel. Enabled RAG against unknown or inaccessible indexes produces no evidence.
4. **Route.** `FoundationRouter` emits a single candidate from `configs/routing/local.json`:
   `fake-foundation-local-1`, region `local`, price list `synthetic-prices-1`. Policy residency
   must exactly equal the processing region. The route quote uses input tokens and the maximum
   output token budget; it must fit `max_cost_micros`. An ineligible candidate is recorded with
   its reason and a null `selected_model_deployment_id`, without calling the provider. Residency
   mismatch returns 403 `residency_unavailable`; exceeding the cost limit returns 422
   `cost_limit_exceeded`. Processing denial returns 403 `processing_forbidden` (normally before
   routing). These are constraint failures, not transient availability errors. No router 503
   case exists in this slice; capacity/health routing belongs to a later milestone. These prices
   are synthetic integer USD micros, rounded up after calculation.
5. **Generate and validate.** The injected `Provider` receives only canonical messages, supplied
   chunks, response format and output limit. `FakeProvider` returns deterministic content
   quoting the supplied synthetic context and a `[document_id/chunk_id]` marker with canonical
   citation ids. Usage is `locally_estimated`, tokenizer `fake-whitespace-v1` (each non-whitespace
   run is a token). At the output limit it truncates and reports `length`; citations refer only
   to markers retained in the output. Its simulated latency is zero; the gateway measures the
   actual provider call duration for generation evidence and the load test. The response cost
   uses actual estimated token counts with the same price list, so it may be lower than the
   route quote. The validator checks non-empty content, canonical and inline citation ids
   against supplied chunks, and an actual JSON object when requested. Truncated invalid JSON
   fails validation. Validation failure returns 502 `validation_failed`; there is no fallback.
6. **Prepare the response.** The gateway produces `InferenceResponse` with the same interaction/trace ids,
   content, citations, usage, estimated cost and finish reason. Operational events contain
   versions, ids, counts, decisions and keyed input/output hashes, never prompt, response or
   retrieved text.
7. **Persistence-path evidence.** `LocalPersistenceRedactor`, version
   `persistence-regex-local-1`, applies the processing credential/payment classes plus emails
   and phone numbers. The local rule set is shared by the configured tenants; names and postal
   codes are outside this slice. Counts cover processing-path messages, metadata values and
   generated output; the query reuses the redacted last message. Inputs are prepared before
   retrieval events, and output before generation events. Metadata values are checked but are
   not stored. Event and stored input/query/output hashes are HMACs of the same persistence-path
   text. Commit adds refs without recomputing those hashes.
   Governed chunk hashes remain the original source-version evidence; chunk content is not stored.
   The pass completes before any content-bearing write. Failure writes content-free metadata
   with `error_code="persistence_redaction_failed"`, null refs and no replay. Failed input
   redaction leaves input/query/output hashes null. If only output redaction fails, its hash is
   null and earlier successful input/query hashes remain unchanged in both events and storage.
   Serving still returns the generated response.
8. **Encrypt and persist.** One transaction writes the `Started`, `Interaction`, `RetrievalRun`,
   `RouteDecision` and `GenerationAttempt` contracts, with the same correlation ids as the events.
   Stages not reached are omitted. Failed attempts are recorded but never replayed. Message,
   query and output payload refs require `content_logging_allowed`; all three remain null for
   this walkthrough's default policy. The walkthrough database contains five metadata records,
   an encrypted replay response and its scoped replay entry. There are no plaintext messages,
   query, output, raw subject, credentials or retrieved chunks. Persistence counts are empty
   for this request because its secrets were already removed by the processing pass.
   SQLite rows include tenant, expiry and lifecycle state. Replay is stored for its operational
   purpose even when content logging is disabled, separately from logging refs.
   AES-256-GCM uses a fresh 96-bit random nonce for each blob, with a uniqueness constraint,
   `key_version="local-1"`, and AAD `tenant_id|interaction_id|field` (`messages`, `query`,
   `output`, or `replay`). Authentication failures never return plaintext. `Settings.payload_key`
   accepts 32 bytes; only local development derives it from the synthetic identity secret.
   Production must supply a KMS-managed key. A transaction failure rolls back metadata and
   payloads together, increments `application.state.persistence.failures`, and still serves
   the response. Encryption and the transaction run through `asyncio.to_thread`; the response
   waits for the write while other requests can use the event loop. Replay reads and privacy
   transactions also run off the event loop; the SQLite `RLock` protects shared transactions.
   No retry or emergency content buffer exists in this slice.

The event order with RAG enabled is:

| Order | Event | Payload |
| --- | --- | --- |
| 1 | `interaction.started.v1` | Application and policy version |
| 2 | `retrieval.completed.v1` | Exact source versions and retrieved/supplied evidence |
| 3 | `route.decided.v1` | Candidate eligibility, region, price version and quote |
| 4 | `generation.completed.v1` | Usage, measured provider time, cost and validation |
| 5 | `interaction.completed.v1` | Trusted identity, policy, task, record links and final status |

All five envelopes share one `trace_id`; all five payloads share one `interaction_id`.
With RAG disabled there are four events: `interaction.started.v1`, `route.decided.v1`,
`generation.completed.v1` (or `generation.failed.v1`), and `interaction.completed.v1`.
These four also share the trace/interaction ids, with a null retrieval run link on completion.
Events remain in memory in the same order. Completion additionally carries persistence redaction
counts and logging refs when allowed. Retrieval/generation event payloads and their persisted
records are identical except for refs filled at commit time.
`application.state.events.events_for_trace(trace_id)` reads them back in tests. No HTTP event
reader is exposed. The default capacity is 4096 accepted events. The collector ignores duplicate
accepted `event_id`s and drops new arrivals when full, incrementing `dropped_events`. Dropped
arrivals may be retried; there is no consumer or durable queue in this slice. Sink exceptions
increment `application.state.inference.emission_failures`. Neither dropping nor emission failure
changes the inference response. The happy-path integration test also requires zero
`emission_failures` and zero `dropped_events`, so silent contract/emission failures fail the test.

Repeat the identical walkthrough request to receive the original response with `replayed: true`, without
new events or generation. Change the body while keeping the same tenant/application/request id
to receive 409 `request_id_conflict`. A concurrent identical request receives 409
`request_in_progress` until the original finishes; a later retry replays a successful response.
Only successful responses that passed persistence redaction and committed are replayable.
Every generation failure, including cancellation and 500/502/503/504
errors from injected components, releases its reservation. Retrying the same request id then
executes again with new interaction/trace ids and a new event sequence for the stages reached.

`Settings.replay_capacity` defaults to 10,000 and must be positive. Memory holds only in-flight
reservations; at capacity, new requests receive 429 `replay_capacity_exceeded`. Completed
replays live solely in the store. Separately, the same setting caps completed replay rows per
tenant: `put_replay` deletes the oldest rows in reservation order and their blobs in SQL within
the interaction transaction. Replay hits do not refresh that order. No tenant bulk load or
per-request scan of completed entries occurs in memory. Evicted ids can execute again.
Cancellation during a started persistence write waits for the worker before releasing the
reservation; a successfully committed response remains replayable.
Replay survives a process restart with
the same data directory and keys. `Settings.replay_ttl_seconds` defaults to 86,400, capped by
the interaction retention deadline (one hour under the checked-in policy). Expired or deleted
entries cannot replay. In-flight reservations remain process-local; use one serving process for
this local slice.

**Replay content:** ADR 0003 and specification 9.1 require redaction before durable replay. When that
pass changes an output containing PII, its durable replay returns the redacted output with the
original ids, usage and cost; the first served output is unchanged. Exact response replay holds
when the second pass makes no changes, including the walkthrough above. A redaction failure
skips durable replay completely.

`Settings.retention_seconds=None` uses each policy's retention; an explicit positive value
overrides it locally. `make retention-sweep TENANT=synthetic-a` deletes expired payloads and
replays, clears refs and marks the graph metadata `expired`; metadata and tombstones remain.
No background scheduler runs the sweep. Payload reads and replay lookups enforce expiry even
before a sweep physically removes the blobs.

Delete one interaction using its returned id:

```sh
curl -X DELETE http://127.0.0.1:8000/v1/privacy/interactions/INTERACTION_ID \
  -H 'Authorization: Bearer synthetic-key-a'
curl -X POST http://127.0.0.1:8000/v1/privacy/subjects/deletion-requests \
  -H 'Authorization: Bearer synthetic-key-a' \
  -H 'Content-Type: application/json' \
  -d '{"subject":"synthetic-caller-1"}'
```

The first endpoint returns 204, or 404 for an absent/other-tenant interaction. The second takes
the raw subject in a strict JSON body (1–512 characters), HMACs it exactly as authentication
does, and returns a deletion count. Raw subjects never appear in the URL or events. The subject
operation records a subject-scope tombstone and deletes that tenant's existing interactions
for the pseudonym in one transaction. Each interaction deletion
atomically records a tombstone, removes all payloads including replay, nulls refs and marks
metadata `deleted`, then emits `privacy.deletion.requested.v1`. Later writes for the same
interaction id are refused. Subject deletion also emits one subject-scope deletion event, even
when no interactions existed, and later writes for that tenant/pseudonym raise `subject_deleted`.
Serving may still succeed, but no graph or replay can be persisted for the deleted subject.
An interaction-only deletion permits a new interaction id. `make dev` keeps access logging disabled.

For injected test failures, `FakeProvider(test_only_failure="error")` returns 502
`provider_failed`; `"deadline_exceeded"` returns 504 `provider_deadline_exceeded`.
The gateway also bounds the provider await using the time remaining in `deadline_ms`.
With RAG enabled these paths replace event 4 with `generation.failed.v1` and still emit event 5 with failed
status. Authentication, unsupported streaming (501) and policy denial fail before the event
sequence. A retrieval or routing failure emits only the stages actually reached and a failed
completion; it does not invent a generation attempt. Test controls are never accepted from HTTP.

OpenTelemetry API spans for policy, retrieval, routing, generation, validation and every event
emission sit within an inference span. Attributes contain only ids and versions. Automatic
exception recording is disabled to prevent provider error bodies entering traces. There are no
metric labels, SDK/exporter dependencies or metrics backend.

Run the review checks and additional privacy/load coverage:

```sh
make contracts
make check integration
uv run --locked pytest tests/security tests/load -s
```

The load test makes 200 distinct authenticated requests through the in-process ASGI application,
subtracts measured provider-call time from total request time, and requires nearest-rank p95
overhead below 50 ms. This is a local slice check, not a production throughput claim.
Durable telemetry, outbox/retry/dead letter, key rotation and backup/restore remain slice 1c.
Streaming, fallback, the feedback endpoint, real providers and exporters remain later work.
