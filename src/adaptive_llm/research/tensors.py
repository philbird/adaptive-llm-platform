"""Tiny Llama aggregate instrumentation and physically compact structured exports."""

import hashlib
import json
import math
from typing import Any

from adaptive_llm.contracts import PruningPlan
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.training.lora import (
    BaseFiles,
    Libraries,
    TrainingExample,
    encoded,
    example_messages,
)


def geometry(base: BaseFiles) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(base.files["config.json"])
    if (
        config.get("model_type") != "llama"
        or not 1 <= config.get("num_hidden_layers", 0) <= 2
        or not 1 <= config.get("hidden_size", 0) <= 32
        or not 1 <= config.get("intermediate_size", 0) <= 64
        or config.get("vocab_size") != 259
        or config.get("num_attention_heads") != 4
        or config.get("num_key_value_heads") not in {2, 4}
    ):
        raise GatewayError(422, "research_tiny_base_required")
    return config


def study_memory_bytes(model: Any, batch_size: int, width: int) -> int:
    """Conservative admission estimate including the eager attention quadratic term."""
    config = model.config
    tokens = batch_size * width
    # Saved hidden/MLP intermediates and their backward workspace; logits, CE and gradients.
    activations = (
        2
        * config.num_hidden_layers
        * tokens
        * (8 * config.hidden_size + 3 * config.intermediate_size)
    )
    logits = 4 * tokens * config.vocab_size
    attention = 3 * config.num_hidden_layers * batch_size * config.num_attention_heads * width**2
    # At least fp32 (statistics/loss), plus integer tokens/masks and a 1.5 safety allowance.
    elements = sum(p.numel() for p in model.parameters()) * 24
    buffers = 4 * (activations + logits + attention) + 32 * tokens
    return int(math.ceil((elements + buffers) * 1.5))


def instrument(
    model: Any,
    tokenizer: Any,
    rows: list[bytes],
    limit: int,
    libs: Libraries,
    *,
    memory_limit_bytes: int = 2_000_000_000,
) -> tuple[bytes, dict[str, list[int]], dict[str, list[int]]]:
    """One bounded batch, one backward pass; only geometry-sized reductions leave this call."""
    torch = libs.torch
    sequences, targets = [], []
    for raw in rows:
        row = TrainingExample.model_validate_json(raw)
        messages = example_messages(row)
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": row.target}], tokenize=True
        )
        if full[: len(prefix)] != prefix:
            raise GatewayError(422, "training_configuration_invalid")
        target = full[len(prefix) :][: limit - 1]
        prompt = prefix[-(limit - len(target)) :]
        if not prompt or not target:
            raise GatewayError(422, "training_configuration_invalid")
        sequences.append(prompt + target)
        targets.append([-100] * len(prompt) + target)
    width = max(map(len, sequences))
    if study_memory_bytes(model, len(sequences), width) > memory_limit_bytes:
        raise GatewayError(422, "training_memory_limit")
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in sequences])
    valid = mask.bool()
    aggregate: dict[str, Any] = {}
    handles = []

    def observe(value: Any, name: str, groups: int, contribution: Any | None = None) -> None:
        activation = value.detach() if contribution is None else contribution
        shaped = activation.reshape(*value.shape[:2], groups, -1)
        active = shaped.detach()[valid].float()
        norms = active.norm(dim=-1)
        stats = torch.stack(
            (
                norms.mean(0),
                norms.var(0, unbiased=False),
                (active.abs() < 1e-8).float().mean((0, 2)),
                torch.zeros(groups),
            ),
            dim=-1,
        )
        aggregate[name] = stats

        def gradient(grad: Any) -> None:
            # First-order change under zero ablation: |activation dot d(loss)/d(activation)|.
            product = (shaped.detach() * grad.reshape_as(shaped)).sum(-1).abs()
            stats[:, 3] = product[valid].mean(0)

        value.register_hook(gradient)

    def output_hook(name: str, groups: int) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            # Removing a decoder layer replaces its output with its residual input.
            observe(value, name, groups, value.detach() - args[0].detach())

        return hook

    def input_hook(name: str, groups: int) -> Any:
        def hook(module: Any, args: Any) -> None:
            observe(args[0], name, groups)

        return hook

    model.eval()
    model.requires_grad_(True)
    model.config.use_cache = False
    try:
        for index, layer in enumerate(model.model.layers):
            handles.extend(
                [
                    layer.register_forward_hook(output_hook(f"layers.{index}", 1)),
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        input_hook(f"attention_heads.{index}", model.config.num_attention_heads)
                    ),
                    layer.mlp.down_proj.register_forward_pre_hook(
                        input_hook(f"mlp_channels.{index}", model.config.intermediate_size)
                    ),
                ]
            )
        ids = torch.tensor([x + [tokenizer.pad_token_id] * (width - len(x)) for x in sequences])
        labels = torch.tensor([x + [-100] * (width - len(x)) for x in targets])
        model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
        tensors = {
            kind: torch.stack(
                [aggregate[f"{kind}.{i}"] for i in range(model.config.num_hidden_layers)]
            )
            for kind in ("layers", "attention_heads", "mlp_channels")
        }
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise GatewayError(409, "research_nonfinite_statistics")
        rankings = {
            f"{kind}.{i}": sorted(range(t.shape[1]), key=lambda j: (float(t[i, j, 3]), j))
            for kind, t in tensors.items()
            if kind != "layers"
            for i in range(t.shape[0])
        }
        rankings["layers"] = sorted(
            range(tensors["layers"].shape[0]),
            key=lambda i: (float(tensors["layers"][i, 0, 3]), i),
        )
        return libs.tensors.save(tensors), {k: list(v.shape) for k, v in tensors.items()}, rankings
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def prune(
    model: Any, base: BaseFiles, plan: PruningPlan, rankings: dict[str, list[int]], libs: Libraries
) -> tuple[BaseFiles, dict[str, list[int]]]:
    if plan.ranking_rule != "ablation_sensitivity":
        raise GatewayError(422, "magnitude_ranking_unsupported")
    config = geometry(base)
    layers, heads, channels = (
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["intermediate_size"],
    )
    removed: dict[str, list[int]] = {}

    def keep(name: str, count: int, kind: str) -> list[int]:
        n = (
            min(count - 1, math.floor(count * plan.maximum_fraction))
            if kind in plan.structures
            else 0
        )
        removed[name] = sorted(rankings[name][:n])
        return [i for i in range(count) if i not in removed[name]]

    retained_layers = keep("layers", layers, "layers")
    state = model.state_dict()
    result = {k: v for k, v in state.items() if not k.startswith("model.layers.")}
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    groups = heads // config["num_key_value_heads"]
    new_heads, new_channels = heads, channels
    for new_index, old_index in enumerate(retained_layers):
        retained_heads = keep(f"attention_heads.{old_index}", heads, "attention_heads")
        retained_channels = keep(f"mlp_channels.{old_index}", channels, "mlp_channels")
        new_heads, new_channels = len(retained_heads), len(retained_channels)
        q = [h * head_dim + d for h in retained_heads for d in range(head_dim)]
        kv = [h // groups * head_dim + d for h in retained_heads for d in range(head_dim)]
        prefix = f"model.layers.{old_index}."
        for key, value in state.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            if new_heads != heads:
                if name in {"self_attn.q_proj.weight", "self_attn.q_proj.bias"}:
                    value = value[q]
                elif name in {
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                }:
                    # Expand shared KV heads for retained queries when GQA groups become uneven.
                    value = value[kv]
                elif name == "self_attn.o_proj.weight":
                    value = value[:, q]
            if name in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
                value = value[retained_channels]
            elif name == "mlp.down_proj.weight":
                value = value[:, retained_channels]
            result[f"model.layers.{new_index}.{name}"] = value.contiguous().clone()
    if not any(removed.values()):
        raise GatewayError(422, "no_structures_removed")
    config.update(
        num_hidden_layers=len(retained_layers),
        num_attention_heads=new_heads,
        num_key_value_heads=new_heads if new_heads != heads else config["num_key_value_heads"],
        head_dim=head_dim,
        intermediate_size=new_channels,
    )
    files = {
        k: v
        for k, v in base.files.items()
        if not k.endswith(".safetensors") and not k.endswith(".safetensors.index.json")
    }
    files["config.json"] = encoded(config)
    files["model.safetensors"] = libs.tensors.save(result)
    digest = hashlib.sha256(
        encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    ).hexdigest()
    return BaseFiles(
        files, digest, base.context_limit, sum(v.numel() for v in result.values())
    ), removed
