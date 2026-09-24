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


def test_study_signatures_public_verification_and_legacy_policy(tmp_path):
    from adaptive_llm.contracts import DatasetLineage, SignedRecord
    from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
    from adaptive_llm.research.models import ActivationSpecification, StudySummary
    from adaptive_llm.research.store import LocalStudyStore
    from adaptive_llm.signing import FIELDS, local_keys, sign_record
    from adaptive_llm.storage.crypto import PayloadCipher
    from adaptive_llm.training.lora import encoded

    signer, verifier = local_keys(tmp_path / "signing")
    keyring = Keyring(b"synthetic-secret", signer=signer, verifier=verifier)
    store = LocalStudyStore(tmp_path, PayloadCipher({"synthetic": b"x" * 32}, "synthetic"), keyring)
    actor = Identity(
        "synthetic",
        frozenset(),
        "local",
        "synthetic-operator",
        "operator",
        frozenset({"synthetic"}),
        frozenset({"research"}),
    )
    lineage = DatasetLineage(
        dataset_id="synthetic", version="synthetic-v1", content_digest="synthetic"
    )
    summary = StudySummary(
        specification=ActivationSpecification(
            calibration_dataset_id=lineage.dataset_id,
            calibration_dataset_version=lineage.version,
            evaluation_dataset_id=lineage.dataset_id,
            evaluation_dataset_version=lineage.version,
            baseline_evaluation_id="synthetic-report",
        ),
        tenant_ids=["synthetic"],
        base_digest="synthetic",
        adapter_digest=None,
        calibration_dataset=lineage,
        evaluation_dataset=lineage,
        sample_size=1,
        wall_seconds=0,
        shapes={},
        rankings={},
        parameter_count_before=1,
    )
    store.save(summary, b"SYNTHETIC AGGREGATES", actor)
    path = store.path(summary.specification.study_id) / "summary.json"
    envelope = json.loads(path.read_text())
    assert envelope["signature_version"] == "ed25519-v1" and "mac" not in envelope
    keyring.signer = None
    (tmp_path / "signing/private.pem").unlink()
    assert store.get(summary.specification.study_id, actor) == (summary, b"SYNTHETIC AGGREGATES")
    unsigned = {k: v for k, v in envelope.items() if k not in FIELDS}
    legacy = {**unsigned, "mac": store._mac(unsigned)}
    path.write_text(json.dumps(legacy))
    with pytest.raises(GatewayError, match="study_integrity_failed"):
        store.get(summary.specification.study_id, actor)
    keyring.legacy_mac_records = True
    assert store.get(summary.specification.study_id, actor)[0] == summary
    for change in (
        {"signature": ""},
        {"signature_key_id": signer.key_id},
        {"signature": "invalid"},
    ):
        path.write_text(json.dumps({**legacy, **change}))
        with pytest.raises(GatewayError, match="study_integrity_failed"):
            store.get(summary.specification.study_id, actor)
    wrong_purpose = sign_record(
        SignedRecord(), signer, "training-checkpoint", encoded(unsigned).decode()
    )
    path.write_text(json.dumps({**unsigned, **wrong_purpose.model_dump(include=FIELDS)}))
    with pytest.raises(GatewayError, match="study_integrity_failed"):
        store.get(summary.specification.study_id, actor)
    with pytest.raises(GatewayError, match="signing_key_required"):
        store.save(summary, b"SYNTHETIC AGGREGATES", actor)
