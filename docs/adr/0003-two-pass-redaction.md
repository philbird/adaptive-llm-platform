# ADR 0003: two-pass redaction (processing path and persistence path)

Status: accepted for milestone 1 design, 2026-09-23.

Context: the specification (sections 1.1, 5.2 and 6.1) places policy and redaction before task
classification and retrieval, and requires policy before durable content storage. Read literally,
a single redaction pass runs before retrieval, so the retriever sees redacted queries. Identifiers
that redaction removes, such as order numbers, account references, invoice numbers and email
addresses, are often exactly the terms that retrieve the right document or that a validator needs
to check a citation. One pass therefore trades retrieval quality for a privacy guarantee that the
processing path does not need, because processing content is transient and already permitted by
`processing_allowed`.

Decision: redaction runs twice with separate configurations and versions.

1. **Processing-path pass** runs in the gateway after policy resolution and before classification,
   retrieval, routing and generation. It removes only credentials, secrets, payment card data and
   content the policy prohibits from processing at all. It never removes business identifiers.
   Its output is the canonical input for the whole online path, including fallback attempts.
2. **Persistence-path pass** runs on the processing-path output immediately before any durable
   write: encrypted payload refs, retrieval query refs, chunk refs, output refs and any content
   that reaches events, evaluation or datasets. It applies the full configured PII and identifier
   redaction for the tenant. Redaction failure fails closed for persistence, never for serving.
   The dataset builder repeats this pass at build time as the specification already requires.

Both passes record their own version and counts. `PolicyDecision.redaction_version` and
`redaction_counts` become two fields each, one per pass, when the contract is next revised.
Hashes such as `input_hash` and `query_hash` are computed on the persistence-path output so a
stored hash never fingerprints an unredacted identifier. Retrieval ACL filtering is unaffected
and still runs before retrieval.

Consequences: retrieval and validation operate on business identifiers, so recall and citation
checks are not degraded by privacy controls. The online path handles unredacted personal data
in memory for the request lifetime, which is already the case for the model call itself and is
bounded by `processing_allowed` and residency constraints. Logs, traces, metrics and exceptions
remain content-free regardless of pass. Two configurations must be tested separately, and the
secret-leak test suite must assert that the processing-path pass removes every credential class
the persistence-path pass removes.
