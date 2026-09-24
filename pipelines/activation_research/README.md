# Optional milestone 6: local activation and structured pruning research

Implemented in `src/adaptive_llm/research/`; this directory contains no executable pipeline.
Both the Settings and routing-config `pruning_research_enabled` flags default to false and
must be enabled. Jobs require a research-capable operator, an allowlisted local tiny base,
approved calibration/evaluation data and a passed exact unpruned baseline evaluation.

Bounded hooks persist only encrypted, authenticated model-sized activation/sensitivity aggregates.
Plans physically remove heads, MLP channels or layers, then reuse full student training and
register a signed `pruned-full-v1` candidate. Magnitude-only ranking is refused. Candidates use
the ordinary evaluation, measured benchmark and explicit promotion path. Parameter reductions
alone never qualify: latency or independently measured process RSS must improve, and every
quality/safety gate still applies. No research job changes a serving route.

See [activation research](../../docs/runbooks/activation-research.md) for configuration, endpoints,
artifact formats, grouped-query constraints, reproduction and the synthetic-only limitations.
