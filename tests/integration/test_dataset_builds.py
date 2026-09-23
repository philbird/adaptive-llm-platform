import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event as ThreadEvent
from typing import TYPE_CHECKING, Any

import pytest

from adaptive_llm.contracts import (
    DatasetBuilt,
    DatasetManifest,
    DatasetSpecification,
    Interaction,
    now,
)
from adaptive_llm.datasets.construction import BuiltExample, PayloadSnapshot
from adaptive_llm.datasets.curation import SPLITS
from adaptive_llm.datasets.eligibility import Example
from adaptive_llm.storage import EncryptedPayload, PayloadReader

OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}
USER = {"Authorization": "Bearer synthetic-key-a"}

if TYPE_CHECKING:
    from conftest import DatasetSeed


def read_rows(seed: "DatasetSeed", manifest: DatasetManifest) -> list[dict[str, Any]]:
    directory = seed.directory / "datasets" / manifest.dataset_id / manifest.version
    rows = []
    hashes = []
    for tenant in sorted(manifest.tenant_ids):
        for split in SPLITS:
            envelope = json.loads((directory / f"{tenant}.{split}.jsonl.enc").read_text())
            blob = EncryptedPayload(
                reference="synthetic",
                tenant_id=tenant,
                interaction_id=f"{manifest.dataset_id}/{manifest.version}",
                field="dataset",
                nonce=base64.b64decode(envelope["nonce"]),
                ciphertext=base64.b64decode(envelope["ciphertext"]),
                key_version=envelope["key_version"],
                expires_at=now() + timedelta(hours=1),
            )
            plaintext = seed.app.state.persistence.cipher.decrypt(
                blob, tenant, blob.interaction_id, "dataset", aad_field=split
            )
            digest = hashlib.sha256(plaintext).hexdigest()
            assert digest == envelope["plaintext_hash"]
            hashes.append(digest)
            rows.extend(json.loads(line) for line in plaintext.splitlines())
    assert hashlib.sha256("".join(hashes).encode()).hexdigest() == manifest.content_digest
    return rows


def build(seed: "DatasetSeed") -> DatasetManifest:
    result = seed.client.post(
        "/v1/datasets/builds", headers=OPERATOR, json=seed.specification.model_dump(mode="json")
    )
    assert result.status_code == 200, result.json()
    return DatasetManifest.model_validate(result.json())


def test_factory_reproducible_targets_provenance_encryption_and_events(
    dataset_seed: "DatasetSeed",
) -> None:
    seed = dataset_seed
    first = build(seed)
    second = build(seed)
    assert first.version != second.version
    # Immutable version and build-clock watermark necessarily change on rebuild.
    dynamic = {"version", "created_at", "deletions_applied_through"}
    assert first.model_dump(exclude=dynamic) == second.model_dump(exclude=dynamic)
    assert second.deletions_applied_through >= first.deletions_applied_through
    rows = read_rows(seed, first)
    assert rows == read_rows(seed, second)
    assert {row["target_source"] for row in rows} == {
        "correction",
        "positive_resolution",
        "production_output",
    }
    assert {row["tenant_id"] for row in rows} == {"synthetic-a"}
    assert first.quality_summary.considered == 60
    assert first.quality_summary.exclusions["training_forbidden"] == 20
    assert first.quality_summary.exclusions["subject_deleted"] == 2
    assert first.quality_summary.exclusions["unresolved_negative_feedback"] > 0
    assert first.approval.status == "pending"
    assert first.examples["train"] >= 5
    assignments = {}
    for row in rows:
        for key in [row["subject_id_pseudonymous"], *row["document_families"]]:
            if key is not None:
                assert assignments.setdefault(key, row["split"]) == row["split"]
        assert row["input"]["tool_results"] == []
        if row["sources"]:
            assert (
                "<<source synthetic-refund-policy/refund-window synthetic-1>>"
                in row["input"]["sources"][0]
            )
            assert row["sources"][0]["index_version"] == "synthetic-index-1"
    base = seed.directory / "datasets" / first.dataset_id
    encoded = (base / first.version / "manifest.json").read_text()
    assert (
        base / first.version / "manifest.mac"
    ).read_text() == seed.app.state.keyring.manifest_mac(encoded)
    assert not (base / first.version / "manifest.sig").exists()
    assert (
        "signing key the builder does not hold"
        in (base / first.version / "data-card.md").read_text()
    )
    assert "SYNTHETIC ANSWER" not in "".join(p.read_text() for p in base.rglob("*") if p.is_file())
    assert seed.client.get(
        f"/v1/datasets/{first.dataset_id}/versions/{first.version}", headers=OPERATOR
    ).json() == first.model_dump(mode="json")
    assert seed.app.state.metadata.get_manifest(first.dataset_id, first.version) == first
    while seed.app.state.dispatcher.dispatch_once():
        pass
    events = [e for e in seed.app.state.events.events if e.event_type == "dataset.built.v1"]
    assert len(events) == 4
    assert len({event.event_id for event in events}) == 4
    for manifest in [first, second]:
        batch = [
            event
            for event in events
            if isinstance(event.data, DatasetBuilt) and event.data.version == manifest.version
        ]
        assert {event.tenant_id for event in batch} == set(manifest.tenant_ids)
        assert len({event.trace_id for event in batch}) == 1
        digest = hashlib.sha256(manifest.model_dump_json(indent=2).encode()).hexdigest()
        assert all(
            isinstance(event.data, DatasetBuilt) and event.data.manifest_digest == digest
            for event in batch
        )


def test_delete_subject_rebuild_watermark_and_current_policy(dataset_seed: "DatasetSeed") -> None:
    seed = dataset_seed
    first = build(seed)
    target = seed.ids[2]
    assert target in {row["interaction_id"] for row in read_rows(seed, first)}
    response = seed.client.post(
        "/v1/privacy/subjects/deletion-requests", headers=USER, json={"subject": seed.subjects[2]}
    )
    assert response.json()["deleted"] == 2
    second = build(seed)
    assert target not in {row["interaction_id"] for row in read_rows(seed, second)}
    assert second.deletions_applied_through > first.deletions_applied_through
    assert second.content_digest != first.content_digest
    seed.policy.training["synthetic-a"] = False
    result = seed.client.post(
        "/v1/datasets/builds", headers=OPERATOR, json=seed.specification.model_dump(mode="json")
    )
    assert result.status_code == 422
    assert len(list((seed.directory / "datasets" / first.dataset_id).iterdir())) == 2


def test_build_auth_policy_version_and_atomic_outbox_failure(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed
    payload = seed.specification.model_dump(mode="json")
    assert seed.client.post("/v1/datasets/builds", json=payload).status_code == 401
    assert seed.client.post("/v1/datasets/builds", json=payload, headers=USER).status_code == 403
    denied = dict(payload, tenant_ids=["synthetic-unapproved"])
    assert seed.client.post("/v1/datasets/builds", json=denied, headers=OPERATOR).status_code == 403
    mismatch = dict(payload, eligibility_policy_version="wrong")
    assert (
        seed.client.post("/v1/datasets/builds", json=mismatch, headers=OPERATOR).status_code == 409
    )

    def fail(*args: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(seed.app.state.outbox, "enqueue", fail)
    assert (
        seed.client.post("/v1/datasets/builds", json=payload, headers=OPERATOR).status_code == 503
    )
    assert (
        seed.app.state.database.connection.execute(
            "SELECT count(*) FROM dataset_manifests"
        ).fetchone()[0]
        == 0
    )
    assert not list((seed.directory / "datasets").rglob("manifest.json"))


def test_cli_same_builder(
    dataset_seed: "DatasetSeed",
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import asyncio

    from adaptive_llm.datasets.__main__ import run

    seed = dataset_seed
    spec = seed.specification.model_copy(update={"eligibility_policy_version": "local-policy-1"})
    path = tmp_path / "spec.json"
    path.write_text(spec.model_dump_json())
    # Default policy explicitly denies training, even though the source rows permitted it.
    with pytest.raises(Exception, match="insufficient_training_examples"):
        asyncio.run(run(path, seed.directory, "local", "synthetic-operator-key"))
    assert capsys.readouterr().out == ""
    from adaptive_llm.app import ROOT
    from adaptive_llm.datasets.__main__ import main

    path.write_text(seed.specification.model_dump_json())
    main(
        [
            "--spec",
            str(path),
            "--data-dir",
            str(seed.directory),
            "--policy",
            str(ROOT / "configs/policy/dataset-demo.json"),
        ]
    )
    manifest = DatasetManifest.model_validate_json(capsys.readouterr().out)
    assert manifest.approval.status == "pending"
    assert manifest.examples["train"] >= 5
    assert manifest.content_digest == build(seed).content_digest
    with pytest.raises(SystemExit) as result:
        main(
            [
                "--spec",
                str(path),
                "--data-dir",
                str(seed.directory),
                "--operator-key",
                "synthetic-key-a",
            ]
        )
    assert result.value.code == 1
    assert capsys.readouterr().err == "dataset_build_failed\n"


def block_constructor(
    seed: "DatasetSeed", monkeypatch: pytest.MonkeyPatch
) -> tuple[ThreadEvent, ThreadEvent]:
    entered, release = ThreadEvent(), ThreadEvent()
    constructor = seed.app.state.datasets.constructor
    original = constructor.build

    def blocked(
        example: Example,
        spec: DatasetSpecification,
        at: datetime,
        *,
        payloads: PayloadReader | None = None,
    ) -> BuiltExample:
        assert isinstance(payloads, PayloadSnapshot)
        entered.set()
        assert release.wait(5), "synthetic constructor release timeout"
        return original(example, spec, at, payloads=payloads)

    monkeypatch.setattr(constructor, "build", blocked)
    return entered, release


def test_inference_completes_while_dataset_constructor_is_blocked(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed
    entered, release = block_constructor(seed, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        building = executor.submit(build, seed)
        try:
            assert entered.wait(2), "synthetic build did not enter construction"
            serving = executor.submit(
                seed.client.post,
                "/v1/inference",
                headers=USER,
                json={
                    "request_id": "synthetic-during-build",
                    "application_id": "support-assistant",
                    "messages": [{"role": "user", "content": "SYNTHETIC concurrent serving"}],
                },
            )
            response = serving.result(timeout=1)
            assert response.status_code == 200
            stored = seed.app.state.metadata.get(
                "synthetic-a", Interaction, response.json()["interaction_id"]
            )
            assert stored is not None and stored.status == "completed"
            assert not building.done()
        finally:
            release.set()
        assert building.result(timeout=5).examples["train"] >= 5


@pytest.mark.parametrize("scope", ["interaction", "subject"])
def test_deletion_during_construction_is_excluded_before_publication(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    seed = dataset_seed
    entered, release = block_constructor(seed, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        building = executor.submit(build, seed)
        try:
            assert entered.wait(2)
            if scope == "subject":
                deleting = executor.submit(
                    seed.client.post,
                    "/v1/privacy/subjects/deletion-requests",
                    headers=USER,
                    json={"subject": seed.subjects[2]},
                )
                assert deleting.result(timeout=1).json()["deleted"] == 2
            else:
                deleting = executor.submit(
                    seed.client.delete, f"/v1/privacy/interactions/{seed.ids[2]}", headers=USER
                )
                assert deleting.result(timeout=1).status_code == 204
        finally:
            release.set()
        manifest = building.result(timeout=5)
    assert manifest.quality_summary.exclusions["deleted_during_build"] == (
        2 if scope == "subject" else 1
    )
    assert "missing_payload" not in manifest.quality_summary.exclusions
    rows = read_rows(seed, manifest)
    assert seed.ids[2] not in {row["interaction_id"] for row in rows}
    if scope == "subject":
        assert seed.ids[3] not in {row["interaction_id"] for row in rows}


def test_publish_rechecks_deletions_after_each_restage(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed
    builder = seed.app.state.datasets
    original = builder._stage
    stages = 0

    def stage(*args: Any, **kwargs: Any) -> DatasetManifest:
        nonlocal stages
        assert not seed.app.state.database.connection.in_transaction
        result = original(*args, **kwargs)
        stages += 1
        if stages <= 2:
            seed.app.state.persistence.delete("synthetic-a", seed.ids[stages + 1], None)
        return result

    monkeypatch.setattr(builder, "_stage", stage)
    manifest = build(seed)
    assert stages == 3
    assert manifest.quality_summary.exclusions["deleted_during_build"] == 2
    assert not {seed.ids[2], seed.ids[3]} & {
        row["interaction_id"] for row in read_rows(seed, manifest)
    }


def test_delete_all_during_compute_fails_minimum_and_cleans_staging(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed
    builder = seed.app.state.datasets
    original = builder._stage

    def stage(*args: Any, **kwargs: Any) -> DatasetManifest:
        result = original(*args, **kwargs)
        for iid in seed.ids[:40]:
            seed.app.state.persistence.delete("synthetic-a", iid, None)
        return result

    monkeypatch.setattr(builder, "_stage", stage)
    response = seed.client.post(
        "/v1/datasets/builds", headers=OPERATOR, json=seed.specification.model_dump(mode="json")
    )
    assert response.status_code == 422
    parent = seed.directory / "datasets" / seed.specification.dataset_id
    assert list(parent.iterdir()) == []
    db = seed.app.state.database.connection
    assert db.execute("SELECT count(*) FROM dataset_manifests").fetchone()[0] == 0
    assert (
        db.execute("SELECT count(*) FROM outbox WHERE event_type='dataset.built.v1'").fetchone()[0]
        == 0
    )


def test_publish_rename_failure_rolls_back_manifest_and_all_tenant_events(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed

    def fail_rename(*args: object) -> None:
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    response = seed.client.post(
        "/v1/datasets/builds", headers=OPERATOR, json=seed.specification.model_dump(mode="json")
    )
    assert response.status_code == 503
    assert list((seed.directory / "datasets" / seed.specification.dataset_id).iterdir()) == []
    db = seed.app.state.database.connection
    assert db.execute("SELECT count(*) FROM dataset_manifests").fetchone()[0] == 0
    assert (
        db.execute("SELECT count(*) FROM outbox WHERE event_type='dataset.built.v1'").fetchone()[0]
        == 0
    )


def test_revision_is_computed_once_at_startup_and_reused_by_builds(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def revision() -> str:
        nonlocal calls
        calls += 1
        return "a" * 40

    monkeypatch.setattr("adaptive_llm.app.code_revision", revision)
    monkeypatch.setattr("adaptive_llm.datasets.builder.code_revision", revision)
    seed: DatasetSeed = request.getfixturevalue("dataset_seed")
    assert calls == 1
    assert seed.app.state.datasets.revision == "a" * 40
    assert build(seed).transformation_code_revision == "a" * 40
    assert build(seed).transformation_code_revision == "a" * 40
    assert calls == 1
