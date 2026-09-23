# Instructions for coding agents working in this repository

You are implementing the specification in `docs/spec/` one reviewable slice at a time. An
orchestrating reviewer reads your diff, runs the checks and sends corrections; you do not
commit or push. Work only on the task in `docs/tasks/` you were given.

## Non-negotiable rules

- `make check integration` must pass before you finish. Run it yourself and fix failures.
- Run `make contracts` after any change to `src/adaptive_llm/contracts.py` and keep the
  generated files in `contracts/` in sync.
- Do not add dependencies unless the task brief allows it; justify any addition in your final
  message. Never change `uv.lock` except through `uv lock` after an allowed dependency change.
- Never put prompts, responses, retrieved text or provider error bodies in logs, exception
  messages, metric labels, traces, fixtures or test snapshots. Fixtures are synthetic only.
- Trusted tenant, subject and application identity come from authentication, never from the
  request body. Policy runs before any durable write. Redaction failure fails closed for
  persistence, never for serving.
- Money is integer USD micros. Identifiers are UUIDv7 via `contracts.uid()`. Times are UTC.
- Typed interfaces (Protocols or ABCs) with injected implementations at provider, policy,
  retriever, router, validator, event and storage boundaries. Local implementations are
  replaceable without touching contracts.
- Keep the health endpoint truthful: `inference_enabled` reflects whether `/v1/inference`
  is mounted.
- Do not widen scope. If the brief is ambiguous, choose the narrower reading and state the
  assumption in your final message. Do not touch `docs/spec/`.

## Conventions

- Python 3.12+, FastAPI, Pydantic v2, Ruff (line length 100), strict mypy. Type everything.
- Package layout: subpackages under `src/adaptive_llm/` (for example `providers/`, `policy/`,
  `rag/`, `routing/`, `validation/`, `events/`, `gateway/`). Do not create files under the
  top-level placeholder directories `services/`, `packages/` or `pipelines/`.
- Tests under `tests/unit`, `tests/contract`, `tests/integration`, `tests/security`,
  `tests/load` (mark load tests with `@pytest.mark.load`). Every new module has tests.
- Config is JSON under `configs/`; local synthetic data under `tests/fixtures`.

## Finish with a report

Your final message must list: files changed, assumptions made, anything deliberately left
out and why, the exact commands you ran and their results.
