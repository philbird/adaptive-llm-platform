# ADR 0002: governance and staged releases

Status: accepted for scaffolding, 2026-09-23.

Decision: content logging, offline evaluation, human review, training and research default off.
Processing and each retained purpose require independent policy decisions. Trusted tenancy
comes from authentication, never a request body. Foundation-only routing is the first serving
milestone. Training, shadow/canary and research remain separate changes.

Ingress contracts (client request bodies) reject unknown fields. Records (stored and emitted
payloads) ignore unknown fields so producers may add fields before consumers upgrade, which
the specification's backward-compatibility rule (section 8) requires. Every contract rejects
an unknown `schema_version`; the event type suffix carries the same major version and the
envelope validator checks they agree. New fields require schema regeneration and review;
additive changes must include defaults. Removing or retyping a field is a new major version.
Use integer USD micros for internal/API cost fields, UUIDv7 identifiers, UTC timestamps and
null for unavailable usage categories.

Consequences: no content collection or external provider is enabled by the scaffold. Production
ownership, policy and infrastructure decisions remain explicit blockers for production rollout.

