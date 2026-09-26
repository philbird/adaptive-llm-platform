import json

import pytest
from fastapi.testclient import TestClient
from test_training import OPERATOR, approve, evaluate, promote, train

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSpecification,
    PairedComparison,
)
from adaptive_llm.datasets.artifacts import approval_mac, verify_manifest
from adaptive_llm.distillation.benchmark import SQLiteBenchmarkStore
from adaptive_llm.gateway.identity import GatewayError, Keyring
from adaptive_llm.registry.artifacts import verify_artifact
from adaptive_llm.signing import FIELDS, Ed25519Signer, Ed25519Verifier, rotate


def test_artifacts_reports_approval_rotation_and_public_only_promotion(evaluation_seed):
    seed = evaluation_seed
    approve(seed)
    baseline = evaluate(seed)
    job = train(seed)
    assert job.state == "succeeded"
    keyring = seed.app.state.keyring
    actor = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    model = seed.app.state.registry.get(job.model_version, actor)
    manifest = seed.app.state.metadata.get_manifest(seed.manifest.dataset_id, seed.manifest.version)
    for record in (manifest, manifest.approval, baseline, model):
        assert record.signature and record.signature_key_id == "local-1"
        assert record.signature_version == "ed25519-v1"
    checkpoint = (
        seed.directory
        / "models"
        / model.registry_id
        / ".checkpoints"
        / model.version
        / "checkpoint-1"
        / "checkpoint.mac"
    )
    assert json.loads(checkpoint.read_text())["signature_version"] == "ed25519-v1"
    private, public = seed.directory / "signing/b.pem", seed.directory / "signing/public.json"
    rotate(private, public, "b")
    keyring.signer = Ed25519Signer(private, "b")
    keyring.verifier = Ed25519Verifier.load(public)
    report = evaluate(seed, model.version)
    assert report.signature_key_id == "b"

    measurement = BenchmarkMeasurement(
        requests=8,
        successes=8,
        p50_latency_ms=5,
        p95_latency_ms=10,
        requests_per_second=100,
        peak_rss_bytes=100000,
        total_cost_micros=80,
        cost_per_success_micros=10,
        input_micros_per_1000_tokens=10,
        output_micros_per_1000_tokens=10,
    )
    benchmark = BenchmarkReport(
        specification=BenchmarkSpecification(
            candidate_version=model.version,
            evaluation_id=report.specification.evaluation_id,
            requests=8,
        ),
        tenant_ids=manifest.tenant_ids,
        candidate_artifact_digest=model.artifact_digest,
        teacher_version="synthetic-teacher",
        teacher_artifact_digest="synthetic-hash",
        dataset_content_digest=manifest.content_digest,
        request_mix_digest="synthetic-mix",
        student=measurement,
        teacher=measurement,
        quality_comparison=PairedComparison(sample_size=8),
        latency_reduction_fraction=0,
        cost_reduction_fraction=0,
        passed=False,
        known_limitations=["Synthetic measurement"],
    )
    store = SQLiteBenchmarkStore(seed.app.state.evaluation_database, keyring)
    benchmark = store.publish(benchmark, actor)
    assert benchmark.signature_key_id == "b"
    keyring.signer = None
    private.unlink()
    (seed.directory / "signing/private.pem").unlink()
    public_only = Keyring(b"independent-synthetic-hmac-secret", verifier=keyring.verifier)
    verify_manifest(manifest, seed.directory, public_only)
    verify_artifact(model, seed.directory / model.storage_location, public_only)
    store.keyring = public_only
    assert store.get(benchmark.specification.benchmark_id, actor) == benchmark
    legacy_benchmark = benchmark.model_copy(update=dict.fromkeys(FIELDS))
    legacy_benchmark = legacy_benchmark.model_copy(update={"mac": store._mac(legacy_benchmark)})
    store.database.connection.execute(
        "UPDATE benchmark_reports SET report=? WHERE benchmark_id=?",
        (legacy_benchmark.model_dump_json(), benchmark.specification.benchmark_id),
    )
    with pytest.raises(GatewayError, match="benchmark_integrity_failed"):
        store.get(benchmark.specification.benchmark_id, actor)
    legacy_approval = manifest.model_copy(
        update={"approval": manifest.approval.model_copy(update=dict.fromkeys(FIELDS))}
    )
    legacy_approval = legacy_approval.model_copy(
        update={
            "approval": legacy_approval.approval.model_copy(
                update={"mac": approval_mac(legacy_approval, public_only)}
            )
        }
    )
    with pytest.raises(GatewayError, match="invalid_dataset_artifact"):
        verify_manifest(legacy_approval, seed.directory, public_only)
    public_only.legacy_mac_records = True
    assert store.get(benchmark.specification.benchmark_id, actor) == legacy_benchmark
    verify_manifest(legacy_approval, seed.directory, public_only)
    public_only.legacy_mac_records = False
    store.database.connection.execute(
        "UPDATE benchmark_reports SET report=? WHERE benchmark_id=?",
        (benchmark.model_dump_json(), benchmark.specification.benchmark_id),
    )
    promote(seed, model.version, "approved", report.specification.evaluation_id)
    promote(seed, model.version, "shadow")
    deployment = create_app(
        Settings(
            data_dir=seed.directory,
            signing_public_keys_path=public,
            policy=seed.app.state.training.policy,
            outbox_dispatch_enabled=False,
        )
    )
    with TestClient(deployment) as client:
        assert deployment.state.keyring.signer is None
        loaded = deployment.state.registry.get(model.version, actor)
        verify_artifact(loaded, seed.directory / loaded.storage_location, deployment.state.keyring)
        assert (
            client.post(
                f"/v1/models/{model.version}/promotion-requests",
                headers=OPERATOR,
                json={
                    "model_version": model.version,
                    "target_state": "shadow",
                    "reason": "SYNTHETIC_PRIVATE_NOTE",
                },
            ).status_code
            == 200
        )
        assert not (seed.directory / "signing/private.pem").exists()
    for change in (
        {"signature": "invalid"},
        {"signature_key_id": "unknown"},
        {"signature": None, "signature_key_id": None, "signature_version": None},
        {"context_limit": model.context_limit + 1},
    ):
        with pytest.raises(GatewayError, match="artifact_integrity_failed"):
            verify_artifact(
                model.model_copy(update=change),
                seed.directory / model.storage_location,
                public_only,
            )
    damaged = benchmark.model_copy(update={"passed": True})
    seed.app.state.evaluation_database.connection.execute(
        "UPDATE benchmark_reports SET report=?", (damaged.model_dump_json(),)
    )
    with pytest.raises(GatewayError, match="benchmark_integrity_failed"):
        store.get(benchmark.specification.benchmark_id, actor)
    # Every promotion rechecks the signature; a valid earlier approval cannot hide tampering.
    damaged_model = model.model_copy(update={"signature": "invalid"})
    seed.app.state.evaluation_database.connection.execute(
        "UPDATE model_versions SET data=? WHERE version=?",
        (damaged_model.model_dump_json(), model.version),
    )
    response = seed.client.post(
        f"/v1/models/{model.version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": model.version,
            "target_state": "approved",
            "evaluation_id": report.specification.evaluation_id,
            "reason": "SYNTHETIC tampered promotion",
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "artifact_integrity_failed"


@pytest.mark.parametrize("allow_legacy", [False, True])
@pytest.mark.parametrize("evaluation_seed", ["private_secret"], indirect=True)
def test_nonlocal_public_only_app_requires_explicit_legacy_opt_in(evaluation_seed, allow_legacy):
    from adaptive_llm.contracts import DatasetManifest
    from adaptive_llm.datasets.artifacts import approval_mac
    from adaptive_llm.evaluation.storage import SQLiteEvaluationStore
    from adaptive_llm.registry.artifacts import signed_metadata
    from adaptive_llm.signing import FIELDS

    seed = evaluation_seed
    approve(seed)
    report = evaluate(seed)
    job = train(seed)
    assert job.state == "succeeded"
    actor = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    writer = seed.app.state.keyring
    model = seed.app.state.registry.get(job.model_version, actor)
    unsigned = dict.fromkeys(FIELDS)
    model = model.model_copy(update=unsigned)
    model = model.model_copy(update={"artifact_mac": writer.artifact_mac(signed_metadata(model))})
    directory = seed.directory / "datasets" / seed.manifest.dataset_id / seed.manifest.version
    dataset = DatasetManifest.model_validate_json((directory / "manifest.json").read_text())
    dataset = dataset.model_copy(update=unsigned)
    encoded = dataset.model_dump_json()
    (directory / "manifest.json").write_text(encoded)
    (directory / "manifest.mac").write_text(writer.manifest_mac(encoded))
    approval = seed.app.state.metadata.get_manifest(dataset.dataset_id, dataset.version).approval
    dataset = dataset.model_copy(update={"approval": approval.model_copy(update=unsigned)})
    dataset = dataset.model_copy(
        update={
            "approval": dataset.approval.model_copy(update={"mac": approval_mac(dataset, writer)})
        }
    )
    report = report.model_copy(update=unsigned)
    encoded = report.model_dump_json()
    report_dir = seed.directory / "evaluations" / report.specification.evaluation_id
    (report_dir / "report.json").write_text(encoded)
    mac = writer.report_mac(encoded)
    (report_dir / "report.mac").write_text(mac)
    checkpoint = {"mac": writer.artifact_mac("synthetic-checkpoint")}
    # The deployment receives public keys and the shared HMAC secret, but no private key.
    public = seed.directory / "signing/public.json"
    (seed.directory / "signing/private.pem").unlink()
    options = {"legacy_mac_records": True} if allow_legacy else {}
    app = create_app(
        Settings(
            data_dir=seed.directory,
            environment="production",
            secret=seed.app.state.settings.secret,
            payload_key=b"x" * 32,
            signing_public_keys_path=public,
            outbox_dispatch_enabled=False,
            **options,
        )
    )
    with TestClient(app):
        reader = app.state.keyring
        assert reader.signer is None and reader.verifier is not None
        assert reader.legacy_mac_records is allow_legacy
        store = SQLiteEvaluationStore(
            app.state.evaluation_database, app.state.evaluation_outbox, reader, seed.directory, 100
        )
        with store.database.transaction():
            store.database.connection.execute(
                "INSERT INTO evaluation_reports VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    report.specification.evaluation_id,
                    json.dumps(dataset.tenant_ids),
                    encoded,
                    mac,
                    None,
                    actor.subject_id_pseudonymous,
                    report.completed_at.isoformat(),
                ),
            )
        checks = [
            (
                lambda: verify_artifact(model, seed.directory / model.storage_location, reader),
                GatewayError,
                "artifact_integrity_failed",
            ),
            (
                lambda: verify_manifest(dataset, seed.directory, reader),
                GatewayError,
                "invalid_dataset_artifact",
            ),
            (
                lambda: store.get(report.specification.evaluation_id, actor),
                GatewayError,
                "report_integrity_failed",
            ),
            (
                lambda: reader.verify_checkpoint(checkpoint, "synthetic-checkpoint"),
                ValueError,
                "checkpoint_integrity_failed",
            ),
        ]
        for verify, error, code in checks:
            if allow_legacy:
                verify()
            else:
                with pytest.raises(error, match=f"^{code}$"):
                    verify()
