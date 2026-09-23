# ADR 0001: local-first Python monorepo

Status: accepted for reversible scaffolding, 2026-09-23.

Context: no existing target repository or confirmed platform. Core development must not
require paid APIs, real user data or accelerators.

Decision: Python 3.12+, FastAPI, Pydantic, uv lockfile, pytest, Ruff and strict mypy. Keep
logical components in one package until extraction is justified. Start with synthetic data
and a deterministic fake provider. Plan SQLite for local metadata only; production storage,
transport and identity are unresolved and must be hidden behind typed interfaces.

Consequences: easy local setup and shared contracts; local development does not validate
production durability, isolation, throughput or legal/privacy compliance. Broad dependency
ranges declare compatibility, while committed uv.lock pins the complete resolved environment.

