"""Pre-versioning records remain readable; new MACs cover all immutable fields."""

import json

import pytest
from test_distillation import build, identity, use_dataset
from test_training import approve, train

from adaptive_llm.contracts import DatasetManifest, ModelManifest
from adaptive_llm.datasets.artifacts import approval_mac, read_shards
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.registry.artifacts import signed_metadata, verify_artifact


@pytest.fixture(autouse=True)
def legacy_mac_writes(monkeypatch):
    # Exercise historical v1/v2 encodings independently of the new signing configuration.
    monkeypatch.setattr("adaptive_llm.app.local_keys", lambda _: (None, None))


@pytest.mark.parametrize("distilled", [False, True])
def test_approval_mac_versions_cover_legacy_and_full_new_records(evaluation_seed, distilled):
    seed = evaluation_seed
    approve(seed)
    if distilled:
        use_dataset(seed, build(seed))
    manifest = seed.app.state.metadata.get_manifest(seed.manifest.dataset_id, seed.manifest.version)
    service = seed.app.state.training
    assert manifest.approval.approval_mac_version == "2"
    unsigned = manifest.model_dump(mode="json")
    unsigned["approval"]["mac"] = None
    assert manifest.approval.mac == service.keyring.manifest_mac(
        json.dumps(unsigned, separators=(",", ":"), ensure_ascii=False)
    )
    assert read_shards(manifest, seed.directory, service.cipher, service.keyring)["train"]
    changed = manifest.model_copy(
        update={
            "specification": manifest.specification.model_copy(
                update={"soft_targets": not manifest.specification.soft_targets}
            )
        }
    )
    assert approval_mac(changed, service.keyring) != manifest.approval.mac

    # Encode the historical record independently, with no version field and its original null
    # omissions. This is deliberately not signed with the current approval_mac implementation.
    unsigned["approval"].pop("approval_mac_version")
    if not distilled:
        unsigned.pop("distillation")
        for field in (
            "source_dataset_id",
            "source_dataset_version",
            "teacher_deployment_id",
            "teacher_minimum_score",
            "teacher_max_output_tokens",
            "general_safety_fraction",
            "soft_targets",
        ):
            unsigned["specification"].pop(field)
    unsigned["approval"]["mac"] = service.keyring.manifest_mac(
        json.dumps(unsigned, separators=(",", ":"), ensure_ascii=False)
    )
    legacy = DatasetManifest.model_validate(unsigned)
    assert legacy.approval.approval_mac_version == "1"
    assert approval_mac(legacy, service.keyring) == legacy.approval.mac
    assert read_shards(legacy, seed.directory, service.cipher, service.keyring)["train"]
    downgraded = manifest.model_copy(
        update={"approval": manifest.approval.model_copy(update={"approval_mac_version": "1"})}
    )
    with pytest.raises(GatewayError, match="invalid_dataset_artifact"):
        read_shards(downgraded, seed.directory, service.cipher, service.keyring)


@pytest.mark.parametrize("capabilities", [None, "1"])
@pytest.mark.parametrize("distilled", [False, True])
def test_manifest_mac_versions_preserve_legacy_and_authenticate_new_metadata(
    evaluation_seed,
    capabilities,
    distilled,
):
    seed = evaluation_seed
    approve(seed)
    job = train(seed)
    keyring = seed.app.state.keyring
    model = seed.app.state.registry.get(job.model_version, identity(seed))
    path = seed.directory / model.storage_location
    assert model.manifest_mac_version == "2"
    immutable = model.model_dump(
        mode="json",
        exclude={
            "artifact_mac",
            "state",
            "evaluation_reports",
            "lifecycle_history",
        },
    )
    assert signed_metadata(model) == json.dumps(immutable, sort_keys=True)
    assert verify_artifact(model, path, keyring)
    for change in (
        {"student_parameter_count": 100},
        {"student_architecture": "synthetic-v1"},
        {"known_limitations": ["synthetic alteration"]},
        {"manifest_mac_version": "1"},
    ):
        with pytest.raises(GatewayError, match="artifact_integrity_failed"):
            verify_artifact(model.model_copy(update=change), path, keyring)

    # A v1 distillation model contains lineage/architecture but predates the student count.
    if distilled:
        lineage = build(seed).distillation
        model = model.model_copy(
            update={"distillation": lineage, "student_architecture": "tiny-llama-1x16-v1"}
        )
    raw = model.model_copy(update={"capability_signature_version": capabilities}).model_dump(
        mode="json"
    )
    raw.pop("manifest_mac_version")
    raw.pop("student_parameter_count")
    if not distilled:
        raw.pop("student_architecture")
        raw.pop("distillation")
    if capabilities is None:
        for field in (
            "capability_signature_version",
            "processing_region",
            "modalities",
            "tools_supported",
            "input_micros_per_1000_tokens",
            "output_micros_per_1000_tokens",
        ):
            raw.pop(field)
    unsigned = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "artifact_mac",
            "state",
            "evaluation_reports",
            "lifecycle_history",
        }
    }
    raw["artifact_mac"] = keyring.artifact_mac(json.dumps(unsigned, sort_keys=True))
    legacy = ModelManifest.model_validate(raw)
    assert legacy.manifest_mac_version == "1"
    assert verify_artifact(legacy, path, keyring)
    # A previously unsigned audit field cannot be added to an old MAC.
    with pytest.raises(GatewayError, match="artifact_integrity_failed"):
        verify_artifact(legacy.model_copy(update={"student_parameter_count": 100}), path, keyring)
