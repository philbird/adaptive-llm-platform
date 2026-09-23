# ADR 0002: governance and staged releases

Status: accepted for scaffolding, 2026-09-23.

Decision: content logging, offline evaluation, human review, training and research default off.
Processing and each retained purpose require independent policy decisions. Trusted tenancy
comes from authentication, never a request body. Foundation-only routing is the first serving
milestone. Training, shadow/canary and research remain separate changes.

Draft contracts reject unknown fields and schema versions. New fields require schema regeneration
and compatibility review; additive changes must include defaults and consumer-first rollout.
This scaffold's consumers are not yet a compatibility guarantee for external producers.
Use integer USD micros for internal/API cost fields, UUIDv7 identifiers, UTC timestamps and
null for unavailable usage categories.

Consequences: no content collection or external provider is enabled by the scaffold. Production
ownership, policy and infrastructure decisions remain explicit blockers for production rollout.

