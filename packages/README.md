# Shared package boundaries

`canonical_models` maps to the initial `src/adaptive_llm/contracts.py`.
`policy_client`, `telemetry` and `model_clients` will hold typed interfaces with injected
implementations. Avoid creating multiple distributions until ownership or reuse justifies it.
Synthetic fixtures live in `tests/fixtures`.

