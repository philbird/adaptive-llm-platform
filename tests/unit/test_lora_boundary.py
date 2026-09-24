import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import TrainingJobSpecification
from adaptive_llm.evaluation.data import fixture_cases
from adaptive_llm.evaluation.judge import DeterministicJudge, JudgeInput
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.training.lora import TrainingExample, base_files, example_messages, libraries


def test_canonical_chat_preserves_system_tools_sources_and_target():
    row = TrainingExample.model_validate(
        {
            "tenant_id": "synthetic-a",
            "split": "train",
            "example_hash": "synthetic-hash",
            "input": {
                "messages": [
                    {"role": "system", "content": "SYNTHETIC policy"},
                    {"role": "user", "content": "SYNTHETIC question"},
                    {"role": "tool", "content": "SYNTHETIC permitted result"},
                ],
                "sources": ["<<source synthetic/chunk v1>>SYNTHETIC context<</source>>"],
            },
            "target": "SYNTHETIC approved answer",
            "sources": [],
            "labels": {},
        }
    )
    messages = example_messages(row)
    assert [m["role"] for m in messages] == ["system", "user", "user", "tool"]
    assert messages[-2]["content"] == row.input.sources[0]
    assert row.target not in json.dumps(messages)


@pytest.mark.parametrize(
    "field,value",
    [("steps", 0), ("batch_size", 0), ("max_sequence_length", 1), ("checkpoint_every", 0)],
)
def test_training_dimensions_are_bounded(field, value):
    with pytest.raises(ValidationError):
        TrainingJobSpecification(
            dataset_id="synthetic", dataset_version="synthetic", **{field: value}
        )


@pytest.mark.parametrize("model,revision", [("..", "seed"), ("tiny", "../seed"), ("/tmp", "seed")])
def test_base_paths_cannot_escape(tmp_path, model, revision):
    with pytest.raises(GatewayError, match="^invalid_base_model$"):
        base_files(tmp_path, model, revision, "synthetic", "synthetic", "synthetic")


def test_missing_optional_group_is_a_fixed_error(monkeypatch):
    original = importlib.import_module

    def absent(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("synthetic private import body")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", absent)
    with pytest.raises(GatewayError, match="^training_dependencies_unavailable$"):
        libraries()


def test_platform_import_and_fake_training_without_optional_group(tmp_path):
    script = """
import sys
from importlib.abc import MetaPathFinder
class Absent(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'safetensors'}:
            raise ImportError('optional training group intentionally unavailable')
sys.meta_path.insert(0, Absent())
import pytest
raise SystemExit(pytest.main(['tests/integration/test_training.py', '-m', 'smoke', '-q']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "UV_CACHE_DIR": "/private/tmp/adaptive-llm-uv-cache"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
    assert "failed" not in result.stdout.lower()
    assert "error" not in result.stdout.lower()


def test_forgetting_and_refusal_items_use_deterministic_judge():
    cases = fixture_cases(Path("tests/fixtures/golden/synthetic.jsonl"), "synthetic-a")
    selected = [c for c in cases if c.category in {"catastrophic_forgetting", "refusal"}]
    assert len(selected) == 2
    judge = DeterministicJudge()
    for case in selected:
        assert judge.score(JudgeInput(case.target, case.expected_facts, case.prohibited, True)) == 5
        assert (
            judge.score(JudgeInput("SYNTHETIC wrong", case.expected_facts, case.prohibited, True))
            < 5
        )
    refusal = next(c for c in selected if c.category == "refusal")
    assert (
        judge.score(
            JudgeInput(
                refusal.target + " SYNTHETIC_FORBIDDEN_SECRET",
                refusal.expected_facts,
                refusal.prohibited,
                True,
            )
        )
        == 0
    )


@pytest.mark.accelerator
@pytest.mark.skip(reason="GPU training is outside slice 3b; CPU-only implementation")
def test_gpu_training():
    pass
