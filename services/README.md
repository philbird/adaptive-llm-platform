# Online component boundaries

These directories reserve the specification's service boundaries. Implementation will initially
live in the shared `src/adaptive_llm` package; they are not independently deployed services.

| Component | Planned responsibility |
| --- | --- |
| gateway | Auth, trusted identity, request limits, deadlines, idempotency and SSE |
| policy_redaction | Purpose-specific policy, redaction and retention decisions |
| rag_orchestrator | ACL-filtered retrieval, source versions and context assembly |
| task_classifier | Explainable task, capability, risk and OOD signals |
| model_router | Hard constraints, foundation default and later approved specialists |
| inference_adapters | Canonical provider interface and deterministic fake |
| output_validator | Structured output, citation, safety and tool validation |
| event_collector | Validated asynchronous delivery, deduplication, retry and dead letter |

