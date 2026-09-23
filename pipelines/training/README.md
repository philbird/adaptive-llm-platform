# Training lifecycle

Slice 3a lives in `src/adaptive_llm/training/` and `src/adaptive_llm/registry/`. It implements
operator dataset approval, current-policy checks, deterministic fake CPU training, authenticated
checkpoints/artifacts, resume, complete lineage, candidate evaluation, gated promotions and
transactional rollback. The CPU smoke test exercises train → evaluate → approve → shadow.
See the [training runbook](../../docs/runbooks/training-and-promotion.md).

Slice 3b adds real LoRA through the `Trainer` boundary, using PyTorch/PEFT in an optional
dependency group with accelerator tests. Slice 3a has no new dependencies and creates no
tensors. Milestone 4 adds actual specialist/shadow/canary routing; milestone 5 adds distillation.
Registry production state alone does not enable specialist serving.
