"""Disabled research imports nothing and admits no routes, tables or artifact tree."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from adaptive_llm.app import create_app
from adaptive_llm.contracts import PruningPlan
from adaptive_llm.gateway.identity import KeyIdentity


@pytest.mark.parametrize("setting,config", [(False, False), (True, False), (False, True)])
def test_both_flags_required_and_no_research_tables(settings, tmp_path, setting, config):
    routing = json.loads(settings.routing_path.read_text())
    routing["pruning_research_enabled"] = config
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(routing))
    app = create_app(replace(settings, pruning_research_enabled=setting, routing_path=path))
    assert not any("research" in p for p in app.openapi()["paths"])
    with TestClient(app) as client:
        assert client.post("/v1/research/jobs", json={}).status_code == 404
        assert not hasattr(app.state, "research")
        for db in (app.state.metadata.database, app.state.registry.database):
            assert not db.connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE '%research%'"
            ).fetchall()
        assert not (settings.data_dir / "research").exists()


def test_disabled_research_is_not_imported():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from adaptive_llm.app import Settings, create_app
assert not Settings().pruning_research_enabled
assert not any('research' in p for p in create_app().openapi()['paths'])
assert not any(n == 'adaptive_llm.research' or n.startswith('adaptive_llm.research.')
               for n in sys.modules)
assert not any(n in sys.modules for n in ('torch', 'transformers', 'peft', 'safetensors'))
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_research_capability_is_opt_in_and_plan_bounded():
    key = KeyIdentity(tenant_id="synthetic", application_ids=[], environment="local")
    assert key.capabilities == frozenset()
    assert PruningPlan().maximum_fraction == 0.25
    with pytest.raises(ValidationError):
        PruningPlan(maximum_fraction=0.9)


def test_research_request_bounds():
    from adaptive_llm.research.models import ActivationSpecification

    args = dict(
        calibration_dataset_id="synthetic",
        calibration_dataset_version="synthetic-v1",
        evaluation_dataset_id="synthetic",
        evaluation_dataset_version="synthetic-v2",
        baseline_evaluation_id="synthetic-report",
    )
    assert ActivationSpecification(**args).sample_size == 64
    for update in (
        {"sample_size": 513},
        {"sample_size": 0},
        {"study_id": ".."},
        {"calibration_split": "test"},
        {"max_sequence_length": 513},
    ):
        with pytest.raises(ValidationError):
            ActivationSpecification(**args, **update)
