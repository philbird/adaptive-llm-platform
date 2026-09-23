# Task: slice 1a — in-memory vertical slice

Implement milestone 1 slice 1a as described in `docs/status.md` and the specification
(`docs/spec/`, sections 7.1–7.8, 9.1, 13.1, 21.2 steps 3–6). Everything runs in process with
no database, no network and no real provider. Read `AGENTS.md` first.

## Deliverables

1. **Deterministic fake foundation provider** in `src/adaptive_llm/providers/`. Define a
   `Provider` Protocol (canonical request in, canonical result out: content, `Usage` with
   `source="locally_estimated"` and a named fake tokenizer, finish reason, latency). The fake
   must be deterministic for the same input, must cite the supplied context chunks, must honour
   `max_output_tokens` (finish reason `length`), and must be able to simulate `error` and
   `deadline_exceeded` via a test-only control. Add conformance tests any provider must pass.
2. **Static identity** in `src/adaptive_llm/gateway/`: an `Authenticator` Protocol and a local
   implementation mapping a bearer API key from `configs/identity/local.json` (create it, keys
   are synthetic) to tenant id, application ids allowed, environment and a pseudonymous subject
   id (HMAC of the caller-supplied subject header with a local secret). Requests whose
   `application_id` is not allowed for the key are rejected with 403. No key: 401.
3. **Policy and processing-path redaction** in `src/adaptive_llm/policy/`: a `PolicyEngine`
   Protocol and a local implementation returning a `PolicyDecision` from a per-tenant JSON
   config (`configs/policy/local.json`; defaults deny logging, evaluation, review and
   training). A processing-path redactor that removes only credentials, secrets and card
   numbers (simple regexes are fine) and records counts under
   `processing_redaction_counts`. Persistence-path redaction is slice 1b; do not implement.
4. **Tenant-scoped retrieval** in `src/adaptive_llm/rag/`: a `Retriever` Protocol and a local
   implementation over `tests/fixtures/documents.json` (load path injectable). Filter by
   tenant, environment, application ACL and index id **before** scoring. Score by simple lexical
   overlap. Produce a `RetrievalRun` with `ChunkEvidence` that distinguishes retrieved from
   supplied-to-model chunks, records `content_hash` (SHA-256 of chunk content is fine here) and
   exact `document_version`/`index_version`. Query hash uses HMAC-SHA256 with the local secret.
5. **Foundation-only router** in `src/adaptive_llm/routing/`: builds a `RouteDecision` with a
   single foundation candidate carrying `processing_region`, `price_list_version` and
   `estimated_cost_micros` from a price list in `configs/routing/local.json` (extend it). Reject
   the candidate if its region does not satisfy policy residency (write a negative test).
6. **Validator** in `src/adaptive_llm/validation/`: checks non-empty output, citation ids exist
   in supplied chunks, JSON validity when `response_format.type == "json_object"`. Failure in
   slice 1a returns a 502 with a content-free error code; fallback chains are milestone 4.
7. **Event collector** in `src/adaptive_llm/events/`: an `EventSink` Protocol and an in-memory
   implementation with a bounded queue, idempotent on `event_id`, a dropped-event counter and
   a way to read back events per trace id for tests. The gateway emits, in order,
   `interaction.started.v1`, `retrieval.completed.v1`, `route.decided.v1`,
   `generation.completed.v1` or `generation.failed.v1`, and `interaction.completed.v1`, all
   sharing one `trace_id` and `interaction_id`. Emitting must never fail the request.
8. **`POST /v1/inference`** wired through a `create_app(settings)` factory with dependency
   injection. Non-streaming only; `stream: true` returns 501. The response is
   `InferenceResponse`. Idempotent replay of `request_id` (same tenant + application, same body)
   within the process lifetime returns the original response with `replayed: true`; different
   body returns 409. Health reports `inference_enabled: true` when mounted.
9. **Observability**: OpenTelemetry API spans for policy, retrieval, routing, generation,
   validation and event emission, attributes limited to versions and ids that are not
   high-cardinality metric labels. No metrics backend yet.
10. **Tests**: provider conformance; identity 401/403; policy defaults; redactor removes a
    synthetic API key and card number and leaves an order number; retrieval tenant isolation
    (tenant B chunk never returned to tenant A) and ACL; router residency rejection; validator
    cases; event idempotency and bounded-queue drop; end-to-end integration test asserting all
    five events correlate; replay and conflict; a `@pytest.mark.load` test that runs 200 requests
    and asserts p95 of (total time − provider time) is under 50 ms.
11. **Walkthrough**: `docs/runbooks/interaction-walkthrough.md` tracing one request through
    each component with the event sequence, using synthetic values.

## Out of scope

Persistence, encryption, persistence-path redaction, streaming, fallback chains, feedback
endpoint, real providers, metrics exporters. Do not modify `docs/spec/`.

## Allowed dependency additions

None. `opentelemetry-api` is already present. If you believe one is required, stop and say so.
