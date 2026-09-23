# Adaptive LLM Specialisation Platform

A provider-neutral project for governed LLM telemetry, dataset curation, specialist training,
evaluation and safe routing, based on the supplied engineering specification v1.0.

**Current status: scaffolding only. Implementation is paused at the user's request.**
No inference, content collection, provider calls, training or deployment is enabled.
No production policy or consent decisions have been assumed.

## Local development

Prerequisites: Python 3.12–3.14, [uv](https://docs.astral.sh/uv/) and Make.

```sh
make dev         # installs locked dependencies; starts health-only API on 127.0.0.1:8000
make check       # formatting, lint, strict typing, unit and contract tests
make ci          # every gate: check, integration, sbom, security, load, drills, smoke
make integration # application startup/health integration test
make contracts   # regenerate initial JSON Schemas and current OpenAPI
```

Visit `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/healthz`.
There are no external dependencies or credentials needed at this stage.
The application is deliberately bound to loopback. Stop it with Ctrl-C.

## What exists

- Installable Python package, dependency lock, Make commands and a local verification gate
  (`make ci`) that runs lint, typing, all test suites, drills and a CycloneDX SBOM audit.
- Pydantic contracts for inference, policy decisions, RAG evidence, routing, attempts,
  interactions, usage, feedback, deletion, dataset/training/evaluation/deployment events and
  the event envelope; generated JSON Schemas. Ingress is strict, records tolerate additions.
- Sortable UUIDv7 identifiers, UTC timestamps, bounded requests and integer USD micros.
- Health-only FastAPI application with a tested startup/shutdown path.
- Synthetic knowledge fixtures, initial local configuration examples and component boundaries.
- Gap analysis, architecture decisions, privacy notes, threat model and milestone checklist.

These are draft contracts, not a complete implementation of the specification. In particular,
dataset/model manifests and control-plane contracts will be added with their milestones.
OpenAPI describes only implemented routes; the inference request JSON Schema is a future contract.

## Structure

```text
src/adaptive_llm/       initial contracts and health-only application
contracts/             generated JSON Schemas, event schema and current OpenAPI
services/              planned online component boundaries
packages/              planned shared interfaces (currently consolidated in Python package)
pipelines/             milestone plans; no executable training/research code
configs/               local synthetic examples and future configurable thresholds
migrations/            database migration policy; no database created yet
tests/                 unit, contract, application integration tests and synthetic fixtures
docs/                  architecture, governance, threat model, decisions and runbooks
infra/                 future environment/deployment boundary
scripts/               contract export
```

The specification is checked in at `docs/spec/`.
Start with [the implementation status and gap analysis](docs/status.md), then
[architecture](docs/architecture.md) and [open decisions](docs/decisions.md).
The next increment is one foundation-only request through policy, RAG, generation,
correlated events and an offline evaluation record. Training and live routing remain
separate later increments.

