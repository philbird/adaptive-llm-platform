# Adaptive LLM Specialisation Platform

## Engineering Specification for Codex

**Status:** Draft for implementation  
**Version:** 1.1 (1.0 plus review refinements of 2026-09-23; see `docs/adr/0003` and the changelog at the end)  
**Primary audience:** Software engineers, ML engineers, platform/security engineers, and Codex  
**Document purpose:** Define a production-oriented system that learns from observed LLM workloads while preserving quality, privacy, auditability, and safe fallback to a foundation model.

---

## 1. Executive summary

Build a platform that captures authorised LLM interactions, RAG retrieval evidence, token usage, quality signals, latency, and cost; converts those observations into governed training and evaluation datasets; trains workload-specific LoRA adapters and/or smaller distilled models; and safely routes eligible requests to those specialists.

The platform must not derive neural weights directly from raw token frequency. Token flow is telemetry and training data. Model weights are produced by a conventional, reproducible training pipeline using curated input/context/target examples, with held-out evaluation and explicit promotion gates.

The first production release should optimise a real workload through supervised fine-tuning or distillation and model routing. Activation analysis and structured pruning of open-weight models are a later, optional research phase. No research model may enter production through a path that bypasses the same evaluation, approval, deployment, monitoring, and rollback controls used for other specialists.

### 1.1 Core lifecycle

```text
Request
  │
  ├─► consent/policy checks ─► redaction ─► interaction telemetry
  │
  ▼
Task classification ─► RAG retrieval ─► route decision
                                    ├─► specialist model
                                    └─► foundation model
                                              │
                                              ▼
                                     response + evidence
                                              │
                 ┌────────────────────────────┘
                 ▼
      governed interaction/event store
                 │
                 ▼
   curation ─► dataset versions ─► train/evaluate
                                     │
                                     ▼
                          registry + promotion gate
                                     │
                                     └─► canary/shadow deployment
```

---

## 2. Objectives

1. Capture input and output token flows, model usage, timing, and cost without adding unacceptable serving latency.
2. Record which RAG documents and chunks were retrieved, their rankings/scores, and which context was supplied to the model.
3. Create reproducible, privacy-governed datasets representing the workload the system actually serves.
4. Train and version specialist capabilities using LoRA/adapters, smaller distilled models, or both.
5. Route requests to the least expensive model that is predicted to meet a defined quality threshold.
6. Fall back safely to a foundation model when a specialist is unavailable, out of scope, uncertain, or produces a response that fails validation.
7. Evaluate every candidate on quality, safety, latency, reliability, and cost before promotion.
8. Support shadow traffic, canaries, rollback, and auditable model lifecycle management.
9. Provide dashboards and alerts for serving, retrieval, routing, dataset, training, and model health.
10. Create a clean extension point for later activation analysis and structured pruning of open-weight models.

### 2.1 Target outcomes

Targets must be configurable per task/domain. Initial suggested targets are:

- Reduce total inference cost per successful outcome for eligible traffic by at least 30% versus
  foundation-only routing, where cost includes retrieval, all attempts, fallbacks and retries
  (see section 23, "Apparent savings hide fallback cost"). Report p50 and p95 cost per request
  alongside, but the gate is the total.
- Keep task success or rubric score within 2 percentage points of the approved foundation-model baseline.
- Keep safety-policy violations no worse than the baseline, with zero tolerance for critical regressions.
- Keep router false-specialist decisions below 2% on the held-out routing set.
- Add no more than 50 ms p95 overhead for logging, classification, and routing, excluding retrieval and model inference.
- Preserve 100% traceability from a deployed model version to its code, configuration, base model, dataset version, and evaluation report.

These are initial engineering goals, not universal product commitments. Confirm them against workload and business requirements before production rollout.

---

## 3. Non-goals

The initial system will not:

- Modify production model weights online or synchronously from individual interactions.
- Train on user data without an explicit lawful basis, policy decision, and tenant-aware consent/configuration.
- Treat token frequency, attention scores, or activation magnitude alone as evidence that a weight can be removed safely.
- Replace RAG with weight memorisation of frequently retrieved documents.
- Autonomously promote a model solely because it is cheaper or faster.
- Guarantee factual correctness from model confidence scores.
- Capture hidden chain-of-thought or require providers to expose it.
- Support arbitrary provider-specific features at the expense of a stable internal request/response contract.
- Implement unstructured weight pruning in the MVP.
- Build a general-purpose foundation model from scratch.

---

## 4. Scope and phased delivery

### Phase A — Observable baseline

- Add the gateway, canonical request/response contract, trace IDs, token/cost/latency capture, redaction, and event ingestion.
- Instrument RAG retrieval and generation.
- Establish storage tiers, retention, access controls, dashboards, and an offline replay/evaluation harness.

### Phase B — Dataset factory and evaluation

- Add eligibility filters, deduplication, decontamination, sampling, quality labels, provenance, dataset manifests, and immutable train/validation/test splits.
- Establish foundation-model baselines and task-specific evaluation suites.

### Phase C — Specialist training

- Implement LoRA/adaptor training first.
- Add knowledge distillation into a smaller model after the data and evaluation loop is stable.
- Register artifacts and evaluation reports in a model registry.

### Phase D — Safe routing

- Run specialists in shadow mode.
- Add constrained live canaries, uncertainty-aware routing, validators, automatic fallback, budget controls, and rollback.

### Phase E — Optional research track

- Add activation capture for approved open-weight models in an isolated environment.
- Study layer/head/channel importance and structured pruning.
- Fine-tune, evaluate, and deploy only through the standard model lifecycle.

---

## 5. System principles

1. **Asynchronous learning:** Production requests create immutable observations; they never trigger immediate weight updates.
2. **Policy before persistence:** Consent, tenant policy, and redaction are applied before durable content storage.
3. **Evidence over intuition:** Promotion depends on versioned evaluation results and statistical confidence.
4. **Foundation model as safety net:** Unsupported, low-confidence, anomalous, or failed specialist requests fall back.
5. **RAG remains the knowledge plane:** Specialists learn task behaviour and domain conventions; mutable facts remain in governed sources.
6. **Reproducibility:** Every dataset, training run, evaluation, and deployment is content-addressed or immutably versioned.
7. **Tenant isolation:** Data, indexes, models, adapters, metrics, and access policies must respect tenant boundaries.
8. **Least data:** Prefer hashes, IDs, counts, and redacted content over raw content when raw content is unnecessary.
9. **No silent degradation:** Routing and model changes require monitoring, comparison to baseline, and rollback criteria.
10. **Provider portability:** Provider details are handled by adapters around an internal canonical API.

---

## 6. High-level architecture

### 6.1 Online serving path

```text
Client
  │
  ▼
API Gateway / Auth / Rate Limits
  │
  ▼
Policy + Redaction ──────────────► append-only event pipeline
  │
  ▼
Task Classifier
  │
  ▼
RAG Orchestrator ─► retriever/reranker ─► governed knowledge sources
  │
  ▼
Model Router ───────────┬───────────────┐
  │                     │               │
  ▼                     ▼               ▼
LoRA endpoint     Small-model endpoint  Foundation endpoint
  │                     │               │
  └──────────────┬──────┴───────────────┘
                 ▼
       Output validators/guardrails
                 │
        pass ────┴──── failure/uncertainty
         │                    │
         ▼                    └─► foundation fallback ─► validate
      Response
```

### 6.2 Offline learning path

```text
Event stream/object store
          │
          ▼
Privacy enforcement + eligibility filters
          │
          ▼
Normalise / deduplicate / label / score
          │
          ▼
Versioned dataset registry
          │
          ├─► adapter fine-tuning
          ├─► small-model distillation
          └─► router training
                    │
                    ▼
          offline evaluation suite
                    │
                    ▼
         registry + approval + promotion
                    │
                    ▼
              shadow / canary
```

### 6.3 Storage separation

- **Operational database:** Requests, route decisions, statuses, manifests, references, and access-controlled metadata.
- **Event/analytics store:** High-volume telemetry and aggregate analysis.
- **Object store:** Encrypted redacted payloads, dataset shards, model artifacts, and evaluation reports.
- **Vector store:** Knowledge embeddings and retrieval indexes; do not commingle unrelated tenants.
- **Model registry:** Model lineage, metrics, approval state, deployment state, and artifact pointers.
- **Secrets manager:** Provider credentials, encryption keys, signing keys, and service credentials.

---

## 7. Components

### 7.1 Request gateway

Responsibilities:

- Authenticate callers and resolve tenant, user/pseudonymous subject, application, environment, and policy.
- Enforce quotas, payload limits, request deadlines, and idempotency.
- Create `trace_id`, `request_id`, and `interaction_id` identifiers.
- Normalise provider-specific requests into the canonical inference contract.
- Stream responses without waiting for analytics persistence.

The gateway must fail serving independently from the analytics pipeline. When telemetry is degraded, emit a counter and buffer non-sensitive metadata if allowed; do not retain unredacted content as an emergency fallback.

### 7.2 Policy, consent, and redaction service

Responsibilities:

- Decide whether content may be processed, logged, used for evaluation, or used for training.
- Detect and redact configured PII, credentials, secrets, payment data, and prohibited content.
- Apply tenant-specific retention and regional-residency rules.
- Run redaction in two passes with separate versions and counts: a processing-path pass before classification and retrieval that removes only credentials, secrets, payment data and prohibited content, and a persistence-path pass before any durable write that applies the tenant's full PII and identifier redaction. Retrieval and validation therefore keep business identifiers. See ADR 0003.
- Emit the policy version and both redaction summaries with every interaction.
- Support hard deletion by subject, tenant, document, interaction, and dataset lineage.

### 7.3 RAG orchestrator

Responsibilities:

- Parse/rewrite queries where configured.
- Retrieve, filter, and rerank candidate chunks.
- Assemble context within a token budget.
- Record document/chunk versions, retrieval scores, ranks, filters, embedding/reranker versions, context positions, and token counts.
- Return citations/evidence in a provider-neutral form.

The event must distinguish `retrieved` chunks from `supplied_to_model` chunks. It should optionally record evidence attribution produced by a citation checker, but attribution is not equivalent to causal proof.

### 7.4 Task classifier

Produces a task/domain label, capabilities required, risk tier, and confidence. Begin with deterministic rules plus embeddings or a lightweight classifier. Keep classification explainable through reason codes. Use the foundation model only when simpler methods are inadequate and cache safe results.

### 7.5 Model router

Selects an eligible candidate using:

- Tenant and policy constraints, including that the candidate's `processing_region` satisfies the policy's residency.
- Required modalities, context length, tool use, structured-output support, and language.
- Task/domain membership.
- Specialist scope and version.
- Router confidence and out-of-distribution score.
- Historical quality, latency, error rate, and cost.
- Current endpoint health, load, budgets, and circuit-breaker state.
- Experiment assignment.

The router must persist the candidate set, selected model, scores, policy constraints, reason codes, and fallback chain.

### 7.6 Model serving adapters

Expose a common interface for hosted foundation models, base models plus adapters, and independently served smaller models. Translate token accounting, finish reasons, tool calls, safety signals, and streaming semantics into canonical forms.

### 7.7 Output validation and fallback

Validators may include:

- Schema/JSON validation.
- Citation presence and entailment checks.
- Groundedness and answerability checks.
- Safety and content-policy checks.
- Tool-call allowlists and argument validation.
- Domain-specific deterministic tests.
- Repetition, truncation, language, and malformed-output checks.

Validation failure, endpoint failure, timeout, out-of-distribution detection, or low router confidence invokes the configured fallback. Prevent loops by tracking attempts and imposing a maximum chain length.

### 7.8 Event collector

- Accept versioned events asynchronously.
- Validate schemas and reject/quarantine invalid payloads.
- Support idempotent delivery using `event_id`.
- Partition by time and tenant or privacy domain.
- Preserve event ordering per interaction where feasible; consumers must tolerate reordering.
- Export OpenTelemetry-compatible traces and metrics.

### 7.9 Dataset builder

- Select only interactions explicitly eligible for the intended use.
- Exclude opt-outs, deleted sources, policy failures, low-quality outputs, unresolved negative feedback, and unsupported licences.
- Normalise messages and RAG context.
- Deduplicate exact and near-duplicate examples.
- Detect benchmark leakage and train/test contamination.
- Add quality and difficulty labels.
- Split by stable grouping keys such as customer, conversation, document family, or time—not by individual row where that would leak information.
- Generate a signed dataset manifest and data card.

### 7.10 Training orchestrator

- Launch reproducible training jobs from immutable configs.
- Track code revision, environment/container, seeds, base model, tokenizer, hyperparameters, dataset IDs, checkpoints, and resource usage.
- Support adapter tuning, supervised fine-tuning, distillation, and router training as separate job types.
- Store signed artifacts and training reports.
- Resume interrupted jobs safely.

### 7.11 Evaluation service

- Run deterministic tests, model-based rubrics where justified, human review samples, and online experiments.
- Compare candidate, foundation baseline, and incumbent specialist.
- Calculate confidence intervals and segmented metrics.
- Block promotion when required suites fail or data coverage is insufficient.

### 7.12 Model registry and deployment controller

- Maintain lifecycle states: `candidate`, `evaluating`, `approved`, `shadow`, `canary`, `production`, `deprecated`, `revoked`.
- Require immutable links to datasets and evaluation reports.
- Deploy signed artifacts only.
- Support gradual traffic shifts and one-action rollback.
- Record human or policy approval with actor, timestamp, and reason.

---

## 8. Canonical data model

Use UUIDv7 or another sortable unique identifier. Timestamps are UTC ISO 8601. Every schema includes `schema_version`, and additive changes remain backward compatible: client-facing request bodies reject unknown fields, but stored and emitted records must ignore unknown fields so producers can add fields before consumers upgrade. The event type suffix (`.v1`) and `schema_version` major must agree. Sensitive fields are classified and encrypted at rest. Hashes of user-derived text use keyed HMAC and record a `hash_scheme`; plain SHA-256 is reserved for governed knowledge content.

### 8.1 Interaction record

```json
{
  "schema_version": "1.0",
  "interaction_id": "uuid",
  "trace_id": "uuid",
  "request_id": "uuid",
  "parent_interaction_id": null,
  "tenant_id": "tenant_ref",
  "subject_id_pseudonymous": "hmac_ref_or_null",
  "application_id": "app_ref",
  "environment": "production",
  "started_at": "2026-09-23T10:00:00Z",
  "completed_at": "2026-09-23T10:00:01Z",
  "task": {
    "label": "customer_support.refund_policy",
    "language": "en",
    "risk_tier": "medium",
    "classifier_version": "task-router-12",
    "confidence": 0.94,
    "reason_codes": ["embedding_match", "intent_rule"]
  },
  "policy": {
    "policy_version": "2026-09-01",
    "content_logging_allowed": true,
    "evaluation_allowed": true,
    "training_allowed": false,
    "retention_class": "30d",
    "residency": "uk",
    "processing_redaction_version": "redactor-processing-2",
    "processing_redaction_counts": {"credential": 0},
    "persistence_redaction_version": "redactor-7",
    "persistence_redaction_counts": {"email": 1}
  },
  "input": {
    "messages_ref": "object://encrypted/redacted/payload",
    "content_hash": "hmac",
    "hash_scheme": "hmac-sha256",
    "token_count": 214,
    "tokenizer": "tokenizer_id",
    "modality": ["text"]
  },
  "retrieval_run_id": "uuid_or_null",
  "route_decision_id": "uuid",
  "generation_attempt_ids": ["uuid"],
  "final_attempt_id": "uuid",
  "feedback_ids": [],
  "status": "completed",
  "error_code": null
}
```

### 8.2 Retrieval run and chunk evidence

```json
{
  "schema_version": "1.0",
  "retrieval_run_id": "uuid",
  "interaction_id": "uuid",
  "query_hash": "sha256",
  "query_ref": "object://encrypted/redacted/query",
  "index_id": "support-kb",
  "index_version": "2026-09-20.3",
  "embedding_model": "embedding_model_id",
  "reranker_model": "reranker_id",
  "filters": {"locale": "en-GB", "valid_at": "2026-09-23"},
  "latency_ms": 42,
  "candidates": [
    {
      "document_id": "doc_ref",
      "document_version": "version_ref",
      "chunk_id": "chunk_ref",
      "rank_retrieved": 1,
      "retrieval_score": 0.88,
      "rerank_score": 0.95,
      "supplied_to_model": true,
      "context_position": 1,
      "token_count": 330,
      "content_hash": "sha256",
      "content_ref": "object://encrypted/redacted/chunk",
      "licence_class": "internal-approved"
    }
  ]
}
```

### 8.3 Route decision

```json
{
  "schema_version": "1.0",
  "route_decision_id": "uuid",
  "interaction_id": "uuid",
  "router_version": "router-18",
  "experiment_id": "exp_or_null",
  "candidates": [
    {
      "model_deployment_id": "specialist-refund-v4",
      "eligible": true,
      "predicted_quality": 0.93,
      "processing_region": "uk",
      "estimated_cost_usd": 0.0012,
      "price_list_version": "prices-2026-09-01",
      "estimated_latency_ms": 410,
      "ood_score": 0.04,
      "reason_codes": ["domain_match", "within_context_limit"]
    }
  ],
  "selected_model_deployment_id": "specialist-refund-v4",
  "fallback_deployment_ids": ["foundation-primary"],
  "decision_latency_ms": 6,
  "policy_constraints": ["uk_residency", "text_only"]
}
```

### 8.4 Generation attempt

```json
{
  "schema_version": "1.0",
  "attempt_id": "uuid",
  "interaction_id": "uuid",
  "attempt_number": 1,
  "model_provider": "provider_or_internal",
  "model_id": "base_model_id",
  "model_version": "immutable_version",
  "adapter_id": "adapter_version_or_null",
  "deployment_id": "specialist-refund-v4",
  "request_parameters": {
    "temperature": 0.1,
    "max_output_tokens": 800,
    "seed": 42
  },
  "input_tokens": 922,
  "cached_input_tokens": 0,
  "output_tokens": 186,
  "reasoning_tokens": null,
  "output_ref": "object://encrypted/redacted/output",
  "output_hash": "sha256",
  "first_token_latency_ms": 180,
  "total_latency_ms": 620,
  "estimated_cost_usd": 0.0012,
  "price_list_version": "prices-2026-09-01",
  "finish_reason": "stop",
  "validation": {
    "passed": true,
    "validator_version": "validator-9",
    "checks": [{"name": "citation_required", "passed": true}]
  },
  "fallback_reason": null,
  "error": null
}
```

When a provider does not expose a token class, store `null`, not zero. Mark token counts as `provider_reported` or `locally_estimated` in a companion field. Every cost carries the `price_list_version` that produced it, so savings can be recomputed when prices change.

`finish_reason` is one of `stop`, `length`, `error`, `cancelled` (client disconnected), `deadline_exceeded` (request deadline reached before completion) and `content_filter`. Cancelled and deadline-exceeded attempts are still recorded with their partial usage.

### 8.5 Feedback and labels

```json
{
  "schema_version": "1.0",
  "feedback_id": "uuid",
  "interaction_id": "uuid",
  "source": "user|reviewer|automated|business_outcome",
  "label_type": "thumb|rubric|correction|resolution|safety",
  "value": {"score": 4, "max_score": 5},
  "comment_ref": null,
  "rubric_version": "support-quality-3",
  "judge_version": null,
  "training_authorised": false,
  "created_at": "2026-09-23T10:05:00Z",
  "actor_id_pseudonymous": "hmac_ref_or_null"
}
```

Automated labels must never be represented as human labels. Preserve label source and model/rubric version; `automated` feedback must carry `judge_version`, and no other source may. Actor identifiers are pseudonymised the same way as subject identifiers.

### 8.6 Dataset manifest

```yaml
dataset_id: support-refunds-sft
version: 2026-09-23.1
purpose: adapter_training
created_at: 2026-09-23T12:00:00Z
source_window:
  start: 2026-07-01T00:00:00Z
  end: 2026-09-01T00:00:00Z
eligibility_policy_version: train-policy-5
transformation_code_revision: git_sha
redaction_version: redactor-7
licence_policy_version: licence-4
examples:
  train: 48000
  validation: 6000
  test: 6000
split_strategy: conversation_and_document_family
content_digest: sha256
deletions_applied_through: 2026-09-23T11:45:00Z
quality_summary:
  accepted_rate: 0.61
  duplicate_rate: 0.08
  languages: {en: 0.94, fr: 0.06}
approval:
  status: approved
  actor: approval_ref
```

### 8.7 Model manifest

Must include:

- Registry ID and immutable version.
- Base model and exact revision/licence.
- Adapter architecture or student architecture.
- Tokenizer and chat template versions.
- Dataset manifests and weights if multiple datasets are mixed.
- Training code revision, container digest, configuration, seed, and hardware class.
- Evaluation report IDs and pass/fail decisions.
- Known limitations, intended tasks, excluded tasks, languages, context limit, and safety notes.
- Artifact hashes/signatures and storage locations.
- Approval, deployment, deprecation, and revocation history.

---

## 9. APIs and events

### 9.1 Inference API

`POST /v1/inference`

Request:

```json
{
  "request_id": "client_supplied_idempotency_key",
  "messages": [{"role": "user", "content": "..."}],
  "application_id": "support-assistant",
  "rag": {"enabled": true, "index_id": "support-kb"},
  "response_format": {"type": "text"},
  "routing": {
    "mode": "auto",
    "max_cost_usd": 0.02,
    "deadline_ms": 5000
  },
  "metadata": {"locale": "en-GB"}
}
```

Response:

```json
{
  "interaction_id": "uuid",
  "model_deployment_id": "specialist-refund-v4",
  "content": "...",
  "citations": [{"document_id": "doc_ref", "chunk_id": "chunk_ref"}],
  "usage": {"input_tokens": 922, "output_tokens": 186},
  "route": {"fallback_used": false},
  "finish_reason": "stop"
}
```

`request_id` is the idempotency key. It is scoped to the authenticated tenant and `application_id`, never global. A replay of the same key with the same body within the idempotency window (default 24 hours) returns the original response with `replayed: true` and creates no new interaction; a replay with a different body is rejected with a conflict error. A replay of a request that is still in flight waits for or is rejected in favour of the original, never executed twice.

`application_id` and `rag.index_id` have no server-side defaults; `rag` defaults to disabled. Client metadata is a bounded map of short strings (`metadata.locale` above) and is never trusted for policy.

Support streaming with Server-Sent Events or the organisation's standard streaming protocol. Emit usage and final route metadata in the terminal event. Client disconnect during streaming records the attempt with `finish_reason: cancelled`.

### 9.2 Feedback API

- `POST /v1/interactions/{interaction_id}/feedback`
- `POST /v1/interactions/{interaction_id}/correction`
- `DELETE /v1/privacy/interactions/{interaction_id}`
- `POST /v1/privacy/subjects/{subject_id}/deletion-requests`

Feedback endpoints must authenticate the actor, protect against cross-tenant access, and record whether a correction is licensed/authorised for training.

### 9.3 Dataset and model control APIs

- `POST /v1/datasets/builds` — create a build from a versioned specification.
- `GET /v1/datasets/{dataset_id}/versions/{version}` — retrieve manifest and status.
- `POST /v1/training/jobs` — launch a declared job against approved datasets.
- `GET /v1/training/jobs/{job_id}` — status, lineage, and metrics.
- `POST /v1/evaluations` — evaluate a registered candidate.
- `POST /v1/models/{model_version}/promotion-requests` — request lifecycle transition.
- `POST /v1/deployments/{deployment_id}/rollback` — return to last approved version.

Control-plane write APIs require stronger authentication, role-based authorisation, audit logging, and idempotency.

### 9.4 Event envelope

Topics or event types:

- `interaction.started.v1`
- `retrieval.completed.v1`
- `route.decided.v1`
- `generation.completed.v1`
- `generation.failed.v1`
- `interaction.completed.v1`
- `feedback.recorded.v1`
- `privacy.deletion.requested.v1`
- `dataset.built.v1`
- `training.completed.v1`
- `evaluation.completed.v1`
- `deployment.changed.v1`

Envelope:

```json
{
  "event_id": "uuid",
  "event_type": "generation.completed.v1",
  "occurred_at": "2026-09-23T10:00:01Z",
  "producer": "model-gateway",
  "tenant_id": "tenant_ref",
  "trace_id": "uuid",
  "schema_version": "1.0",
  "data": {}
}
```

Use an outbox pattern or equivalent for events that must correspond to committed operational state. Consumers must be idempotent and maintain dead-letter handling.

---

## 10. Privacy, security, and governance

### 10.1 Data classification

Classify fields at minimum as public, internal, confidential, personal, highly sensitive, credential/secret, or prohibited. The schema registry should associate a classification with every field that may contain content.

### 10.2 Required controls

- Default content logging to off for a new tenant until configured.
- Separate permissions for operational access, raw-content access, dataset curation, training, model approval, and deployment.
- Encrypt in transit and at rest; use per-environment keys and consider per-tenant keys for high-sensitivity tenants.
- Store raw or redacted payloads separately from broadly queryable metadata.
- Tokenise or HMAC stable subject identifiers; do not use reversible pseudonyms in analytics tables.
- Never log provider API keys, auth headers, secrets, full payment data, or unredacted credentials.
- Run secret/PII detection before persistence and again during dataset build.
- Maintain immutable audit trails for data access, dataset creation, model approval, deployment, and deletion.
- Implement deletion propagation from source interactions/documents through datasets, checkpoints where feasible, and future model versions.
- Define a documented response when removal from an already trained model is required: revoke model, retrain without deleted examples, re-evaluate, and redeploy.
- Set explicit retention by data class. Expiry must delete payloads and derived indexes, not merely hide records.
- Prevent retrieval across tenant, environment, region, or access-control boundaries.
- Validate training-data licences and contractual restrictions.
- Perform threat modelling for prompt injection, retrieval poisoning, training-data poisoning, model extraction, membership inference, and malicious model artifacts.
- Scan model artifacts and containers; verify signatures at deployment.
- Restrict outbound network access from training and serving workloads.

### 10.3 Prompt injection and RAG security

Treat retrieved content as untrusted data. Preserve the distinction between system instructions and source text; apply document-level ACLs before retrieval; scan ingestion; bound tool permissions; require server-side tool allowlists; and test indirect prompt injection in the evaluation suite.

### 10.4 Consent and purpose limitation

Track separate permissions for operational processing, retained logging, offline evaluation, human review, and model training. A record eligible for observability is not automatically eligible for training. Dataset builds must re-evaluate current policy rather than trusting a stale boolean copied at ingestion.

---

## 11. Training pipeline

### 11.1 Example construction

Construct a canonical example as:

```text
system and policy instructions
+ conversation/input
+ supplied RAG context with source boundaries
+ permitted tool results
→ approved target output
```

Possible targets, in descending preference:

1. Verified human-authored correction or accepted response.
2. Response with a confirmed successful business outcome and no negative signals.
3. Foundation-model teacher response that passes deterministic and rubric evaluation.
4. Production response that passes high-confidence quality filters.

Do not train indiscriminately on every generated answer; that amplifies errors and style artefacts.

### 11.2 Curation stages

1. Re-evaluate consent, retention, licence, and deletion state.
2. Redact and scan for secrets/PII.
3. Remove corrupt, partial, failed, policy-violating, or ambiguous records.
4. Normalise message roles, tools, context delimiters, citations, and tokenizer-specific templates.
5. Deduplicate at interaction, semantic, document, and output-template levels.
6. Add task, language, difficulty, risk, retrieval-quality, and outcome labels.
7. Balance distributions or record deliberate sampling weights.
8. Freeze grouped train/validation/test splits.
9. Test for benchmark leakage and cross-split contamination.
10. Produce the manifest, data card, sample review, and approval record.

### 11.3 LoRA/adapters

Use adapter training as the first specialist approach because it is cheaper to iterate and easy to compare with the same base model. The initial training implementation should support:

- Configurable target modules, rank, alpha, dropout, learning rate, precision, and gradient accumulation.
- Optional multi-adapter serving and adapter hot-loading only where isolation and latency are acceptable.
- Checkpointing and deterministic validation.
- Merged and unmerged export formats.
- Task- and tenant-specific adapter policy.

Evaluate catastrophic forgetting, general instruction following, refusal behaviour, tool use, and context reliance—not only in-domain accuracy.

### 11.4 Smaller-model distillation

Train a smaller student after the dataset and eval framework is proven.

- Use the best approved output as the target; optionally include calibrated soft targets/logits only if available and contractually allowed.
- Mix domain traffic with a controlled general-capability and safety set.
- Retain RAG context so the student learns to use evidence rather than memorise mutable facts.
- Include abstention and escalation examples.
- Track teacher model/version, generation parameters, and judge versions.
- Filter synthetic targets before they enter the dataset.

### 11.5 Router training

Train the router from counterfactual evaluation where possible: run multiple candidate models offline on the same prompt and estimate which cheapest candidate clears the quality threshold. Avoid learning exclusively from historic selected routes because this creates selection bias.

Router output should include calibrated suitability probabilities and an out-of-distribution/abstain signal. A simple rules-plus-classifier baseline is preferred before reinforcement or bandit approaches.

### 11.6 Reproducibility and schedules

- Trigger dataset builds and training manually at first, then on a controlled schedule or data-volume threshold.
- Pin all dependencies, base model revisions, tokenizers, prompts, and judge rubrics.
- Log random seeds and nondeterministic operations.
- Make pipeline stages restartable and idempotent.
- Never overwrite a dataset or model version.

---

## 12. Evaluation framework

### 12.1 Evaluation layers

1. **Unit/contract tests:** schemas, provider adapters, redaction, routing constraints, fallback, event idempotency.
2. **Golden-set tests:** expert-reviewed task examples with deterministic assertions and rubrics.
3. **Held-out workload:** immutable examples separated by conversation, user/tenant where appropriate, document family, and time.
4. **Safety/adversarial suite:** jailbreaks, indirect prompt injection, sensitive-data leakage, tool abuse, and harmful content.
5. **Retrieval tests:** recall@k, nDCG/MRR, context precision, citation correctness, stale-document behaviour, and ACL isolation.
6. **Performance tests:** time to first token, total latency, throughput, memory, concurrency, cold start, and error rate.
7. **Cost tests:** per request, per successful outcome, and per million input/output tokens, including retrieval and infrastructure.
8. **Human review:** blinded pairwise or rubric scoring of statistically meaningful samples.
9. **Shadow/canary analysis:** real distribution, without exposing candidate output until approved.

### 12.2 Core metrics

Quality:

- Task success rate or domain rubric score.
- Exact match/F1 for tasks where appropriate.
- Groundedness, citation precision/recall, and unsupported-claim rate.
- Structured-output validity and tool-call success.
- Abstention appropriateness and escalation accuracy.
- Human preference rate versus baseline.

Safety and privacy:

- Policy violation rate by severity.
- Sensitive-data leakage rate.
- Prompt-injection success rate.
- Cross-tenant retrieval incidents.
- Memorisation/canary-string extraction tests where appropriate.

Operations:

- p50/p95/p99 time to first token and end-to-end latency.
- Availability, timeout rate, fallback rate, and retry rate.
- Input, retrieved-context, cached, and output tokens.
- Infrastructure and provider cost per request/successful outcome.

Routing:

- Coverage: share of traffic sent to each route.
- Quality-eligible coverage: share safely served by specialists.
- False-specialist rate and unnecessary-foundation rate.
- Calibration error and out-of-distribution detection rate.
- Incremental cost savings subject to quality constraints.

### 12.3 Promotion gate

A candidate may advance only when:

- All required suites completed on the declared versions.
- It is non-inferior to the baseline within the configured quality margin overall and on critical segments.
- No critical safety, privacy, tenant-isolation, or policy regression occurs.
- Latency/error/cost targets are met at expected concurrency.
- Human review meets the minimum sample and agreement threshold when required.
- The report includes confidence intervals, sample sizes, distribution coverage, and known limitations.
- Non-inferiority is tested as a paired comparison on the same held-out items: the lower bound of the 95% bootstrap confidence interval on the mean per-item (candidate − baseline) score must exceed −margin, overall and on every critical segment. The minimum sample size is derived from pilot variance so the interval half-width is below half the margin, and the pilot and derived n are recorded in the report. A gate with no recorded sample size is not passable.
- Data, ML, security, and product owners approve according to risk tier.

Do not collapse evaluation to a single composite score. Hard safety/quality gates take precedence over cost savings.

### 12.4 Judge-model controls

If LLM-as-judge is used:

- Pin model, prompt, rubric, and decoding parameters.
- Randomise answer order and mask model identity.
- Calibrate against human judgments.
- Use multiple judgments or adjudication for high-impact decisions.
- Never use the candidate itself as its sole judge.
- Record disagreements and avoid treating judge scores as objective truth.

---

## 13. Routing and fallback policy

### 13.1 Decision order

1. Apply hard policy, residency, modality, tool, and context constraints.
2. Reject unhealthy or over-capacity deployments.
3. Classify task and calculate out-of-distribution risk.
4. Estimate candidate quality, latency, and cost.
5. Select the cheapest candidate that exceeds quality and confidence thresholds.
6. If none qualifies, use the foundation model.
7. Validate output and fall back if necessary.

### 13.2 Example policy

```yaml
route_policy_id: support-v3
eligible_specialists:
  - specialist-refund-v4
quality_threshold: 0.90
router_confidence_threshold: 0.85
ood_threshold_max: 0.15
max_attempts: 2
hard_fallback_on:
  - validation_failure
  - endpoint_error
  - deadline_risk
  - policy_uncertainty
  - unsupported_tool
foundation_fallback: foundation-primary
```

### 13.3 Fallback behaviour

- Reuse the same redacted canonical input and approved RAG context when safe.
- Record each attempt separately under one interaction.
- Respect the overall request deadline; do not start fallback when success before deadline is implausible.
- Do not return the failed specialist output alongside the fallback response unless explicitly requested for internal diagnostics.
- Trigger circuit breakers when specialist validation failures, latency, or server errors exceed thresholds.

### 13.4 Online learning caution

Multi-armed bandits may later optimise among already approved candidates, but exploration must be bounded by quality and safety constraints. Do not explore unapproved models on users.

---

## 14. Observability

### 14.1 Tracing

Create a distributed trace spanning gateway, policy, classification, retrieval, reranking, routing, model attempts, validation, fallback, streaming, and event publication. Avoid attaching raw prompts/responses to general-purpose traces.

### 14.2 Metrics and dashboards

Required dashboards:

- Traffic, success, errors, saturation, and latency by tenant/application/model/task.
- Token and cost totals split by input, cached input, RAG context, and output.
- Route distribution, router confidence/OOD, fallback reasons, and circuit-breaker state.
- Retrieval latency, recall proxies, empty retrievals, stale documents, index versions, and citation results.
- Output validation failures and safety signals.
- Event lag, rejected schemas, dead-letter volume, and dropped telemetry.
- Dataset counts, rejection reasons, label mix, duplication, drift, consent/deletion state, and split integrity.
- Training loss, validation metrics, compute utilisation, failures, and artifact status.
- Evaluation comparisons, segment regressions, and promotion status.
- Production drift in inputs, tasks, languages, retrieved sources, output lengths, quality proxies, and business outcomes.

### 14.3 Alerts

Alert on:

- Critical safety/privacy or cross-tenant events immediately.
- Error/timeout/validation/fallback spikes.
- Quality proxy or business-outcome degradation.
- Latency/cost budget breaches.
- Retrieval empty-rate or index freshness regressions.
- Data pipeline lag or redaction failures.
- Distribution drift outside configured tolerances.
- Unsigned, unapproved, or lineage-incomplete deployment attempts.

All alerts should carry model/deployment, router, index, policy, and application versions.

---

## 15. Deployment and operations

### 15.1 Environments

Use separate development, staging, and production environments with separate credentials, storage, indexes, registries, and encryption keys. Production data must not be copied into lower environments without an approved, irreversible de-identification workflow.

### 15.2 Deployment progression

1. Offline evaluation.
2. Staging load and failure testing.
3. Production shadow at 0% user-visible responses.
4. Internal or allowlisted canary.
5. 1–5% eligible live traffic.
6. Incremental expansion subject to automated and human review gates.
7. Full eligible traffic, retaining the foundation fallback.

### 15.3 Rollback

- Keep the prior approved deployment warm where cost permits.
- Make route-policy rollback independent from model rollback.
- Support a global specialist kill switch and per-tenant/per-task disablement.
- Automatically stop expansion and optionally roll back on breached hard thresholds.
- Exercise rollback in staging and periodically in production-safe drills.

### 15.4 Reliability

- Define SLOs for inference availability, latency, and event durability.
- Use timeouts, bounded retries with jitter, bulkheads, queues, backpressure, and circuit breakers.
- Degrade gracefully to the foundation model or a documented error; never bypass policy controls.
- Capacity-plan both normal specialist use and fallback surges.

---

## 16. Optional activation analysis and structured pruning

This is a research extension, not an MVP dependency.

### 16.1 Preconditions

- Only open-weight models with compatible licences and local instrumentation.
- Approved, representative, privacy-safe calibration and evaluation datasets.
- Isolated compute and artifact storage.
- A stable unpruned baseline and end-to-end evaluation suite.

### 16.2 Instrumentation

Capture aggregated activation statistics for a configured sample rather than retaining full per-token tensors by default. Candidate statistics include:

- Per-layer and per-channel activation norms/sparsity.
- Attention-head output norms and ablation sensitivity.
- MLP neuron/channel activation frequency and magnitude.
- Gradient- or loss-based sensitivity on calibration samples.
- Latency and memory contribution by structure.

Instrumentation must be sampled, bounded, versioned, and disabled in normal production serving unless explicitly approved. Treat activation data as potentially sensitive.

### 16.3 Structured pruning workflow

1. Establish unpruned baseline.
2. Gather activation/sensitivity summaries over representative slices.
3. Rank removable structures such as attention heads, MLP channels, blocks, or layers.
4. Apply conservative structured pruning compatible with serving hardware.
5. Fine-tune or distil the pruned model.
6. Run the full evaluation and red-team suite.
7. Compare actual wall-clock latency and memory, not only parameter count or FLOPs.
8. Register as a new candidate with full lineage.

Magnitude-only removal and unstructured sparsity are out of scope unless hardware/runtime benchmarks demonstrate a real deployment benefit. Activation evidence does not replace ablation and end-to-end evaluation.

---

## 17. Suggested technology stack

The implementation should fit the organisation's existing platform. A sensible reference stack is:

- **Services:** Python with FastAPI and Pydantic for ML-facing APIs; optionally Go for a very high-throughput gateway.
- **Inference contract:** HTTP/JSON plus Server-Sent Events; gRPC internally if already standard.
- **Workflow orchestration:** Temporal, Dagster, Prefect, or the existing scheduler. Prefer a durable orchestrator with retries and lineage.
- **Event transport:** Kafka/Redpanda, managed equivalents, or a cloud-native queue/event bus.
- **Operational storage:** PostgreSQL.
- **Analytics:** ClickHouse, BigQuery, Snowflake, or the existing warehouse.
- **Object artifacts:** S3-compatible object storage with versioning, lifecycle policy, encryption, and immutability controls.
- **RAG:** PostgreSQL with pgvector for modest scale; a managed vector database or OpenSearch where scale/filters require it.
- **Training:** PyTorch, Hugging Face Transformers/Datasets, PEFT for LoRA, Accelerate or the organisation's distributed trainer.
- **Serving:** vLLM, Text Generation Inference, or an approved managed endpoint; adapter support must be load-tested.
- **Experiment/model tracking:** MLflow or an equivalent registry; optionally Weights & Biases for experiments if policy permits.
- **Evaluation:** pytest plus a versioned internal evaluation harness; integrate specialised libraries selectively rather than coupling core contracts to one framework.
- **Observability:** OpenTelemetry, Prometheus-compatible metrics, Grafana, and the existing log/trace backend.
- **Data quality:** Great Expectations, Pandera, or explicit schema/data tests in the pipeline.
- **Infrastructure:** Containers, Kubernetes or managed batch/serving, Terraform/OpenTofu, and Git-based CI/CD.
- **Security:** Cloud KMS, secrets manager, workload identity, artifact signing, SBOM and vulnerability scanning.

Do not introduce every listed product. Choose one per capability, favour existing organisational standards, and hide vendors behind interfaces where switching cost would otherwise be high.

---

## 18. Suggested repository structure

```text
adaptive-llm-platform/
├── README.md
├── pyproject.toml
├── Makefile
├── .env.example
├── docs/
│   ├── architecture.md
│   ├── data-governance.md
│   ├── threat-model.md
│   ├── runbooks/
│   └── adr/
├── contracts/
│   ├── openapi/
│   ├── events/
│   └── schemas/
├── services/
│   ├── gateway/
│   ├── policy_redaction/
│   ├── rag_orchestrator/
│   ├── task_classifier/
│   ├── model_router/
│   ├── inference_adapters/
│   ├── output_validator/
│   └── event_collector/
├── pipelines/
│   ├── dataset_builder/
│   ├── training/
│   │   ├── lora/
│   │   ├── distillation/
│   │   └── router/
│   ├── evaluation/
│   └── activation_research/
├── packages/
│   ├── canonical_models/
│   ├── policy_client/
│   ├── telemetry/
│   ├── model_clients/
│   └── test_fixtures/
├── configs/
│   ├── routing/
│   ├── datasets/
│   ├── training/
│   ├── evaluation/
│   └── retention/
├── migrations/
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── evaluation/
│   ├── security/
│   └── load/
├── infra/
│   ├── modules/
│   └── environments/
└── scripts/
```

A monorepo is recommended initially because contracts, fixtures, and end-to-end tests change together. Services may be extracted later based on independent scaling or ownership needs.

---

## 19. Milestones and deliverables

### Milestone 0 — Discovery and decisions

Deliverables:

- Confirmed workload, tenants, risk tiers, baseline models, RAG sources, volumes, latency/cost goals, regions, and retention.
- Architecture decision records for storage, event transport, serving, registry, and identity.
- Data inventory, threat model, and consent/training policy.
- Representative offline seed set and initial rubrics.

Exit criteria: security/data owners approve the design; unresolved decisions are documented with owners.

### Milestone 1 — Instrumented foundation-model baseline

Deliverables:

- Gateway and canonical provider adapter.
- RAG instrumentation.
- Policy/redaction path.
- Versioned events and storage.
- Cost, latency, token, retrieval, and error dashboards.
- Foundation-only inference with no behaviour regression.

Deliver milestone 1 as three reviewable slices: **1a** an in-memory vertical slice (fake provider, tenant resolution, policy and processing-path redaction, retrieval, generation, validation, correlated events, one end-to-end test); **1b** persistence (migrations, persistence-path redaction, encrypted payload refs bound to tenant and interaction, idempotent replay, retention and deletion tombstones); **1c** resilience (durable outbox, retry, dead letter, quarantine, telemetry-outage drill, runbooks and drills).

Exit criteria: ≥99.9% valid event correlation in staging load tests; content policy/retention tests pass; telemetry degradation does not break serving; slice 1a alone must meet the 50 ms p95 overhead target on a local load test.

### Milestone 2 — Dataset and evaluation platform

Deliverables:

- Dataset eligibility and build pipeline.
- Immutable manifests, data cards, splits, deduplication, deletion propagation, and lineage.
- Golden set, held-out set, retrieval suite, safety suite, and reporting.

Exit criteria: a dataset build can be reproduced from its manifest; leakage and cross-tenant tests pass; baseline report is approved.

### Milestone 3 — First adapter specialist

Deliverables:

- LoRA training configuration and pipeline.
- Registered artifact and model card.
- Offline evaluation against the foundation baseline.
- Staging serving endpoint.

Exit criteria: candidate passes promotion gates and load/security tests.

### Milestone 4 — Router, shadow, and canary

Deliverables:

- Deterministic constraints plus trained/calibrated router.
- Shadow comparison pipeline.
- Output validators, fallback, circuit breakers, and kill switch.
- Canary and rollback automation.

Exit criteria: specialist meets live canary quality, safety, cost, and latency thresholds; rollback drill succeeds.

### Milestone 5 — Smaller distilled model

Deliverables:

- Teacher-target generation and filtering.
- Student training and evaluation.
- Deployment benchmark on intended hardware.

Exit criteria: student provides material real-world cost/latency improvement while passing the same promotion gates.

### Milestone 6 — Activation/pruning research, optional

Deliverables:

- Instrumentation library and calibration study.
- Structured-pruning experiments.
- Fine-tuned pruned candidate, reproducible benchmark, and research report.

Exit criteria: demonstrated hardware-level benefit with no unacceptable quality/safety regression. Production deployment remains a separate approval.

---

## 20. Acceptance criteria

### 20.1 Serving and telemetry

- Every completed inference has correlated request, retrieval, route, attempt, validation, and response records, subject to explicit no-logging policy.
- Token counts preserve provider-reported versus estimated status and distinguish input/output/cached tokens where available.
- RAG records distinguish retrieved chunks from supplied chunks and identify exact source versions.
- Event ingestion is versioned, idempotent, retryable, and has dead-letter handling.
- Analytics unavailability does not cause successful inference requests to fail.
- Raw content never appears in ordinary application logs or general trace attributes.

### 20.2 Privacy and security

- Consent/purpose policy is evaluated before storage and again before dataset inclusion.
- Cross-tenant retrieval and dataset isolation tests pass.
- PII/secret redaction and prohibited-content tests pass at the required thresholds.
- Deletion requests remove or tombstone eligible source records, indexes, and future dataset inclusion, with auditable completion.
- Model and dataset access follows least privilege; all control-plane changes are audited.
- Only signed, approved artifacts can be deployed.

### 20.3 Dataset and training

- Every training example has source provenance, policy eligibility, transformation version, and split assignment.
- Splits prevent conversation/document-family leakage and pass contamination checks.
- Dataset and model builds are reproducible from manifests within documented nondeterminism tolerances.
- Adapter and distillation pipelines register checkpoints, final artifacts, code/config lineage, and failure status.
- Training does not begin on unapproved datasets.

### 20.4 Evaluation and routing

- A foundation baseline and at least one held-out, golden, safety, retrieval, and performance suite exist.
- Promotion is automatically blocked on hard-gate failure.
- Router decisions include reason codes and enforce hard capability/policy constraints.
- Low-confidence and out-of-distribution inputs select the foundation route.
- Specialist failures and validator failures invoke a bounded, observable fallback.
- Shadow and canary reports compare quality, safety, latency, error rate, and cost by critical segment.
- The global kill switch and rollback meet the defined recovery-time target in a drill.

### 20.5 Business outcome

- On the agreed eligible workload, the production canary meets the configured non-inferiority quality margin and safety gates.
- It demonstrates the agreed cost or latency improvement with statistically meaningful volume.
- Owners can trace any production response to the model, adapter, router, RAG index, policy, and deployment versions used.

---

## 21. Explicit implementation instructions for Codex

Codex should implement this specification incrementally and keep the repository runnable at each milestone.

### 21.1 Before coding

1. Inspect the existing repository, conventions, CI, deployment platform, data infrastructure, identity model, and test tooling. Extend existing patterns rather than introducing parallel frameworks.
2. Read repository-local instruction files and architecture decisions.
3. Produce a short gap analysis mapping existing components to this specification.
4. List unresolved decisions that materially affect privacy, tenancy, retention, model/provider choice, volume, latency, or infrastructure. Use safe defaults only where decisions are reversible.
5. Create architecture decision records for consequential choices.
6. Build a vertical slice before broadening: one request → one RAG run → one model attempt → one correlated event set → one evaluation record.

### 21.2 Build order

1. Define canonical Pydantic/domain models and JSON Schemas first.
2. Add contract tests and example payloads.
3. Implement the foundation-model adapter and a deterministic fake provider for tests.
4. Implement request IDs, trace context, policy hooks, and redaction before durable content logging.
5. Instrument RAG with exact document/chunk versions.
6. Add event production, idempotent collection, operational metadata, and encrypted payload references.
7. Add dashboards and failure alerts for the baseline.
8. Implement the dataset builder with eligibility, deduplication, grouped splits, manifests, and deletion tests.
9. Implement the evaluation harness and lock the initial baseline before training.
10. Add LoRA training and registry integration.
11. Add routing in shadow mode, then validation/fallback, then live canary.
12. Add distillation only after adapter promotion proves the lifecycle.
13. Keep activation/pruning code behind an experimental package and feature flag.

### 21.3 Engineering constraints

- Prefer typed interfaces and dependency injection at provider, storage, retriever, router, validator, registry, and policy boundaries.
- Keep request serving independent from long-running training/evaluation jobs.
- Do not place raw prompts or outputs in logs, exception strings, metric labels, traces, fixtures, or snapshots.
- Never use high-cardinality IDs as metric labels.
- Do not silently discard event failures; meter, retry, and quarantine them.
- Use database migrations; do not mutate schemas manually.
- Store times in UTC and money in fixed-precision decimal or integer micros.
- Use content hashes and immutable versions for source documents, datasets, prompts, configs, models, and reports.
- Make external writes idempotent.
- Bound retries, payload sizes, queue depth, context length, generation length, and fallback attempts.
- Add feature flags for telemetry content, specialist routing, each deployment, shadowing, canaries, and pruning research.
- Keep secrets out of source control and provide a safe `.env.example` containing names only.
- Pin dependencies and generate an SBOM in CI.

### 21.4 Required tests

Codex must add:

- Unit tests for token/cost accounting, policy decisions, redaction, router constraints, validation, and fallback.
- Schema compatibility and API contract tests.
- Provider-adapter conformance tests using fakes.
- Event duplication, reordering, retry, and dead-letter tests.
- RAG provenance and tenant-ACL tests.
- Dataset eligibility, deletion, deduplication, grouped-split, and leakage tests.
- Reproducibility tests for dataset manifests and training configuration.
- Model promotion state-machine tests.
- Integration tests for the end-to-end vertical slice.
- Load tests for the gateway/router and failure tests for unavailable dependencies.
- Security tests for prompt injection, malicious tool calls, secret leakage, and cross-tenant access.
- A small CPU-safe smoke test for training/evaluation; mark accelerator tests separately.

### 21.5 Local developer experience

Provide:

- One documented command to start local dependencies and services.
- One command for lint/type-check/unit tests.
- One command for the end-to-end integration test.
- Seeded fake interactions and synthetic RAG documents containing no real user data.
- A deterministic fake model provider so core development requires no paid API.
- Example dashboards/configuration and a walkthrough that traces one interaction through the system.

### 21.6 Pull request discipline

Keep changes milestone-sized. Each pull request should include:

- Scope and design summary.
- Contract/schema changes and compatibility impact.
- Privacy/security considerations.
- Tests run and results.
- Observability added.
- Migration and rollback steps.
- Screenshots or sample reports when dashboards/evaluation output change.
- Follow-up work explicitly out of scope.

Do not combine the production telemetry path, full training system, routing rollout, and pruning research into one change.

### 21.7 Definition of done for each feature

A feature is done only when code, tests, schema/docs, metrics, alerts where relevant, privacy review notes, migration, and rollback/runbook updates are included. A model feature additionally requires registry lineage and an evaluation report.

---

## 22. Initial configuration and decisions to confirm

Before production implementation, owners must confirm:

- Tenants/applications and isolation model.
- Data residency, retention, deletion, consent, and training permissions.
- Expected requests per second, prompt/context/output sizes, and availability/latency SLOs.
- Foundation provider(s), approved open-weight base models, licences, and hosting constraints.
- RAG sources, ACL model, index freshness, embedding/reranking choices, and citation requirements.
- First specialist task, minimum dataset size/quality, supported language(s), and risk tier.
- Quality rubric, hard safety gates, non-inferiority margin, and cost/latency target.
- Human-review workflow and promotion approvers.
- Existing event, warehouse, orchestration, registry, serving, secrets, observability, and CI/CD systems.
- Recovery objectives, canary plan, and rollback authority.

If these inputs are missing, Codex should build provider-neutral interfaces, fakes, and local infrastructure, but must not invent legal/privacy policy or deploy real user-data training.

---

## 23. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Training on incorrect model outputs | Require quality filters, verified outcomes, human corrections, and held-out evaluation |
| Privacy or consent violation | Purpose-specific policy checks at ingestion and dataset build; deletion lineage; least-data storage |
| Retrieval/training poisoning | Source ACLs, ingestion scanning, provenance, anomaly detection, trusted dataset approval |
| Evaluation leakage | Grouped/time-based splits, benchmark decontamination, immutable test sets |
| Router sends novel tasks to specialist | Calibrated confidence, OOD detection, hard constraints, foundation fallback |
| Apparent savings hide fallback cost | Measure total cost per successful outcome, including retries and retrieval |
| Judge-model bias | Blind/randomised comparisons, human calibration, multiple evaluators, pinned versions |
| Model drift or workload shift | Distribution monitoring, recurring evaluation, route disablement, retraining gates |
| Adapter cross-tenant leakage | Strict adapter/data tenancy policy: dataset manifests carry the tenant set and the registry refuses a deployment scope wider than training consent scope; isolated serving where required; access and membership-inference tests |
| Pruning reduces parameters but not latency | Benchmark on target hardware/runtime before promotion |
| Provider telemetry discrepancies | Preserve reported/estimated provenance and reconcile sampled counts |
| Foundation fallback overload | Capacity planning, circuit breakers, admission controls, and degradation plan |

---

## 24. Final success definition

The platform is successful when it can prove—not merely assume—that an approved subset of real traffic is served by a cheaper or faster specialist at an agreed quality and safety level, while every decision and artifact is traceable, user data remains governed, the foundation model remains a dependable fallback, and future experimentation such as structured pruning can occur without weakening production controls.

---

## Changelog

- **1.1 (2026-09-23):** cost target defined as total cost per successful outcome (was median; conflicted with section 23). Two-pass redaction (7.2, ADR 0003). Backward compatibility clarified as strict ingress, tolerant records (8). Keyed hashes for user-derived text (8, 8.1). `processing_region` and `price_list_version` on route candidates and attempts (8.3, 8.4). Finish reasons enumerated including `cancelled` and `deadline_exceeded` (8.4). Feedback actor pseudonymised, `judge_version` required for automated labels, `training_authorised` recorded (8.5). Idempotency semantics for `request_id` and no demo defaults in the request (9.1). Residency as a hard router constraint (7.5). Non-inferiority test and sample-size rule made explicit (12.3). Milestone 1 split into slices 1a/1b/1c (19). Adapter cross-tenant mitigation made concrete (23).
