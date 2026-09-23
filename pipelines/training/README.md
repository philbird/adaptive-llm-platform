# Training milestones (not implemented)

Milestone 3: LoRA first, against approved immutable datasets and a locked evaluated baseline.
Track base/tokenizer revisions, code/container/config hashes, seeds, hardware, checkpoints,
signatures and full registry lineage. Include CPU-safe smoke and separately marked GPU tests.

Milestone 4: counterfactual router evaluation, then shadow, canary and bounded fallback.
Milestone 5: filtered teacher targets and smaller-model distillation through the same gates.
Never update weights synchronously from requests or promote solely on cost/latency savings.

