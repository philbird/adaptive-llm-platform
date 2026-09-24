"""Residual ablation and batch memory admission on the synthetic tiny base."""

import json

import pytest

from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.research.tensors import instrument, study_memory_bytes
from adaptive_llm.training.lora import base_files, cpu, generate_tiny_base, libraries, load_base


@pytest.fixture
def tiny(tmp_path):
    try:
        libs = libraries()
    except GatewayError:
        pytest.skip("optional training group absent")
    generate_tiny_base(tmp_path / "base-models/tiny/seed-17")
    base = base_files(tmp_path, "tiny", "seed-17", "tiny-byte-v1", "tiny-chat-v1", "CC0-1.0")
    with cpu(libs, 23):
        model, tokenizer = load_base(base, libs, "fp32")
        yield model, tokenizer, libs


def rows(count=1):
    return [
        json.dumps(
            {
                "input": {"messages": [{"role": "user", "content": "SYNTHETIC " * 20}]},
                "target": "SYNTHETIC answer",
            }
        ).encode()
    ] * count


@pytest.mark.parametrize("zero_layer", [0, 1])
def test_zero_residual_contribution_ranks_first_for_layer_removal(tiny, zero_layer):
    model, tokenizer, libs = tiny

    def zero_contribution(module, args, output):
        return args[0] + (output - args[0]) * 0

    handle = model.model.layers[zero_layer].register_forward_hook(zero_contribution)
    try:
        raw, _, rankings = instrument(model, tokenizer, rows(), 128, libs)
    finally:
        handle.remove()
    layers = libs.tensors.load(raw)["layers"][:, 0]
    assert layers[zero_layer].tolist() == [0, 0, 1, 0]
    assert layers[1 - zero_layer, 0] > 0
    assert layers[1 - zero_layer, 3] > 0
    assert rankings["layers"][0] == zero_layer


def test_batch_memory_refusal_precedes_any_hook_registration(tiny, monkeypatch):
    model, tokenizer, libs = tiny
    memory_limit = 2_000_000
    assert sum(p.numel() for p in model.parameters()) * 24 < memory_limit
    assert study_memory_bytes(model, 4, 128) > memory_limit
    assert study_memory_bytes(model, 512, 512) > 2_000_000_000

    def forbidden(*args, **kwargs):
        pytest.fail("memory refusal must precede hook registration and forward")

    monkeypatch.setattr(libs.torch.nn.Module, "register_forward_hook", forbidden)
    monkeypatch.setattr(libs.torch.nn.Module, "register_forward_pre_hook", forbidden)
    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(GatewayError, match="training_memory_limit"):
        instrument(model, tokenizer, rows(4), 128, libs, memory_limit_bytes=memory_limit)
