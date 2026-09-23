# Data governance design notes

Scaffold privacy review: the health-only service accepts no content and writes no data.
Fixtures are synthetic. Contracts classify text fields as confidential; a full field-level
classification registry is still required before persistence is implemented.

The serving increment must apply authenticated tenant policy before durable content storage,
redact both input and output in two passes as set out in ADR 0003 (secrets only before
processing, full PII before persistence), HMAC subject identifiers with a managed secret, and
keep payloads separate from queryable metadata. Hashes of user-derived text (inputs, queries,
outputs) use keyed HMAC with a recorded `hash_scheme`, because a plain hash of a short
message can be reversed by dictionary attack; plain SHA-256 is reserved for governed knowledge
content. Never include raw content or provider error bodies in
exceptions, logs, metric labels or traces. Redaction failure must fail closed for persistence.

Logging, evaluation, review and training permissions are independent. A future dataset build
must re-evaluate current consent, retention, licensing and deletion state. No cached boolean
is sufficient authorisation. Retention deletes payloads and derived indexes, not just views.

Deletion must propagate through events, payloads, datasets and registry lineage. Retraining
or revocation is required when removal from a trained model is necessary; deleting a source
row does not remove learned information. Signed audit metadata should remain content-free.
Key management, regional residency, retention periods and legal basis await owner confirmation.

