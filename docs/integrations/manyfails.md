# ManyFails candidate triage integration contract

The server authenticates tenant `manyfails` and authorizes application `research-sweep` from
the bearer key. ManyFails supplies only the application selector in the body; metadata
cannot set tenant, subject, purposes or residency. Configure the ManyFails server with
`ADAPTIVE_GATEWAY_URL` and `ADAPTIVE_GATEWAY_KEY`. Keep the key on the server.

## Inference

Send `POST ${ADAPTIVE_GATEWAY_URL}/v1/inference` with
`Authorization: Bearer ${ADAPTIVE_GATEWAY_KEY}` and `Content-Type: application/json`.
This is the complete body shape. Substitute the production triage system prompt for
`SYSTEM_PROMPT`, the stable per-hit identifier, and the actual JSON-encoded news hit.
The checked-in prompt reference is `tests/fixtures/prompts/manyfails-triage.md`; the wire
message contains its text, not the filename or reference. The title/page are untrusted data.

```json
{
  "request_id": "manyfails-sweep-20260926-hit-001",
  "application_id": "research-sweep",
  "messages": [
    {"role": "system", "content": "SYSTEM_PROMPT"},
    {"role": "user", "content": "{\"title\":\"SYNTHETIC: Cedar Notes adds a theme\",\"description\":\"An optional theme release; the app remains available.\",\"url\":\"https://example.invalid/synthetic/theme\",\"page\":\"SYNTHETIC scraped excerpt\"}"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "triage",
      "schema": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "is_failure": {"type": "boolean"},
          "kind": {"type": "string", "enum": ["product_shutdown", "company_shutdown", "feature_pulled", "acquihire_killed", "scaled_back", "never_shipped", "pivoted_away", "incident", "watch", "not_a_failure"]},
          "product": {"type": ["string", "null"]},
          "company": {"type": ["string", "null"]},
          "event_date": {"type": ["string", "null"]},
          "size": {"type": "string", "enum": ["titan", "funded", "indie", "unknown"]},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "reason": {"type": "string"}
        },
        "required": ["is_failure", "kind", "product", "company", "event_date", "size", "confidence", "reason"]
      }
    }
  },
  "routing": {"mode": "auto", "max_cost_micros": 20000, "deadline_ms": 30000},
  "max_output_tokens": 400,
  "metadata": {"sweep_query_id": "synthetic-sweep-001", "hit_url_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
}
```

The schema above matches the supplied fixture. The task prompt specifies `event_date` as
`YYYY-MM-DD` or null and `reason` as one line. Bound `page` to 1,500 characters before
serialization. The system prompt may be at most 8,000 characters; all messages together
may be at most 32,000. Metadata is limited by this integration to `sweep_query_id` and the
SHA-256 of the exact hit URL (UTF-8, lowercase hex); each value has at most 200 characters.
No tools, streaming or RAG are requested. Temperature is zero in the provider adapter.

On 200, parse `JSON.parse(response.content)`; `content` is a string containing the verdict,
not a nested response object. Retain `interaction_id` alongside the editorial candidate for
later corrections. `usage`, `estimated_cost_micros`, `model_deployment_id`, `finish_reason`,
`route` and `replayed` supply operational context. Apply ManyFails' existing acceptance rule:

```typescript
const accepted = verdict.confidence >= 0.6
  && Boolean(verdict.product || verdict.company)
  && verdict.kind !== "watch" && verdict.kind !== "not_a_failure"
  && (verdict.kind === "incident" || verdict.is_failure);
```

Acceptance queues a candidate for Phil; it does not publish it. A non-`stop` finish reason
should be treated as an incomplete/filtered verdict even if it is parseable.

## Corrections

When Phil dismisses or promotes a candidate, send the complete human-corrected verdict as
a JSON string to `POST /v1/interactions/{interaction_id}/correction`, using the same bearer
credential. Example dismissal:

```json
{
  "correction": "{\"is_failure\":false,\"kind\":\"not_a_failure\",\"product\":null,\"company\":\"Cedar Labs\",\"event_date\":null,\"size\":\"unknown\",\"confidence\":1,\"reason\":\"SYNTHETIC: an ordinary release, not a shutdown.\"}",
  "training_authorised": true
}
```

For promotion, fill the actual failure/incident fields and preserve the complete verdict
shape. The gateway encrypts the request's response format alongside retained messages;
schema content never enters events. The correction endpoint validates both the submitted
and redacted target against that recorded schema before writing. Invalid JSON or schema
violations return 422 `invalid_correction_target` with no feedback or training target saved.
Validate in ManyFails too so the editor can explain field errors. Historical interactions
without a recorded format retain their previous correction behavior. A 200 feedback record
supplies `feedback_id` and `correction_ref`; if
`error_code` is `persistence_redaction_failed`, the target was not saved. Corrections are
append-only feedback, with no request-id deduplication: send once per deliberate human edit.
They are not model training or promotion approvals.

## Idempotency and errors

Choose a stable `request_id` per classification attempt, at most 100 characters from letters,
digits, underscore, dot and hyphen. Persist and retry the identical complete body with that
id. It is scoped to authenticated tenant and application. A successful stored response
replays with the same interaction id and `replayed: true` (default window 24 hours).
Changing the prompt, hit, schema, metadata or options requires a new id. Persistence
redaction can change replay content; persistence failures cannot promise durable replay.
Failed generations release the reservation, so retrying the same id may generate again.

Errors have the fixed envelope `{"error":{"code":"..."}}`:

| HTTP | Codes | ManyFails action |
| --- | --- | --- |
| 401 | `unauthenticated` | Fix/rotate gateway key |
| 403 | `application_forbidden`, `environment_forbidden`, `processing_forbidden`, `residency_unavailable` | Fix configuration; stop gateway retries |
| 422 | `invalid_request`, `cost_limit_exceeded` | Correct the body or configured budget |
| 422 | `invalid_correction_target` | Correct the full verdict to satisfy the original schema |
| 503 | `correction_schema_unavailable` | Retain the human edit locally; retry after stored schema recovery |
| 409 | `request_in_progress` | Retry the identical request after bounded backoff |
| 409 | `request_id_conflict` | Recover the original body or assign a new attempt id |
| 429 | `replay_capacity_exceeded` | Back off before retrying |
| 502 | `provider_invalid_response`, `provider_failed`, `validation_failed`, `pipeline_failed` | Use ManyFails' own provider |
| 503 | `provider_rate_limited`, `provider_unavailable`, `no_healthy_deployment` | Use ManyFails' own provider |
| 504 | `provider_timeout`, `provider_deadline_exceeded` | Use ManyFails' own provider |
| 501 | `streaming_not_supported` | Remove streaming; this is also a 5xx fallback condition |

For **any gateway 5xx**, keep calling ManyFails' existing provider for that hit; a transport
failure should use the same availability fallback. Avoid duplicate candidate insertion by
keeping ManyFails' existing hit deduplication. Do not invent a gateway interaction id for
the fallback result. Handle correction 404 `interaction_not_found` or 409
`interaction_inactive` as unavailable history; retain the human edit in ManyFails.
