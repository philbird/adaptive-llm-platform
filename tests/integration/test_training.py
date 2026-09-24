import json
from time import perf_counter
from typing import TYPE_CHECKING

import pytest
from conftest import wait_training

from adaptive_llm.contracts import (
    EvaluationReport,
    Event,
    ModelManifest,
    TrainingJob,
    TrainingJobSpecification,
    uid,
)
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.registry.artifacts import verify_artifact
from adaptive_llm.training.fake import FakeTrainer

if TYPE_CHECKING:
    from conftest import EvaluationSeed

OPERATOR = {"Authorization": "Bearer synthetic-operator-key", "X-Subject": "synthetic-operator"}


def approve(seed: "EvaluationSeed") -> None:
    seed.app.state.training.policy.training["synthetic-b"] = True
    response = seed.client.post(
        f"/v1/datasets/{seed.manifest.dataset_id}/versions/{seed.manifest.version}/approval",
        headers=OPERATOR,
        json={"reason": "SYNTHETIC_PRIVATE_NOTE"},
    )
    assert response.status_code == 200, response.json()
    assert response.json()["approval"]["status"] == "approved"
    assert response.json()["approval"]["reason"] == "SYNTHETIC_PRIVATE_NOTE"
    stored = seed.app.state.metadata.get_manifest(seed.manifest.dataset_id, seed.manifest.version)
    assert stored.approval.reason == "SYNTHETIC_PRIVATE_NOTE"


def spec_for(seed: "EvaluationSeed") -> TrainingJobSpecification:
    return TrainingJobSpecification(
        dataset_id=seed.manifest.dataset_id, dataset_version=seed.manifest.version
    )


def train(seed: "EvaluationSeed", spec: TrainingJobSpecification | None = None) -> TrainingJob:
    response = seed.client.post(
        "/v1/training/jobs", headers=OPERATOR, json=(spec or spec_for(seed)).model_dump(mode="json")
    )
    assert response.status_code == 200, response.json()
    return wait_training(seed.client, response.json()["specification"]["job_id"], OPERATOR)


def evaluate(
    seed: "EvaluationSeed", version: str | None = None, **updates: object
) -> EvaluationReport:
    spec = seed.request.model_copy(
        update={
            "evaluation_id": uid(),
            **updates,
            **({"candidate_deployment_id": version} if version else {}),
        }
    )
    response = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert response.status_code == 200, response.json()
    return EvaluationReport.model_validate(response.json())


def promote(
    seed: "EvaluationSeed", version: str, state: str, evaluation: str | None = None
) -> ModelManifest:
    response = seed.client.post(
        f"/v1/models/{version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": version,
            "target_state": state,
            "reason": "SYNTHETIC_PRIVATE_NOTE",
            "evaluation_id": evaluation,
        },
    )
    assert response.status_code == 200, response.json()
    return ModelManifest.model_validate(response.json())


@pytest.mark.smoke
def test_cpu_train_evaluate_approve_shadow(evaluation_seed: "EvaluationSeed") -> None:
    seed = evaluation_seed
    start = perf_counter()
    approve(seed)
    baseline = evaluate(seed)
    job = train(seed)
    assert job.state == "succeeded" and job.checkpoint_refs == ["checkpoint-1", "checkpoint-2"]
    assert job.specification.code_revision != "server"
    assert seed.client.get(
        f"/v1/training/jobs/{job.specification.job_id}", headers=OPERATOR
    ).json() == job.model_dump(mode="json")
    report = evaluate(seed, job.model_version)
    assert report.passed and report.baseline_report_id == baseline.specification.evaluation_id
    assert report.candidate_manifest_version == job.model_version
    assert report.candidate_artifact_digest == job.artifact_digest
    assert job.artifact_digest in report.candidate_model_version
    registry = seed.app.state.registry
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    assert registry.get(job.model_version, identity).state == "evaluating"
    promote(seed, job.model_version, "approved", report.specification.evaluation_id)
    model = promote(seed, job.model_version, "shadow")
    # Retried operator actions compare the readable reason and produce no duplicate audit row.
    assert promote(seed, job.model_version, "shadow") == model
    assert [t.to_state for t in model.lifecycle_history] == [
        "candidate",
        "evaluating",
        "approved",
        "shadow",
    ]
    events = [
        Event.model_validate_json(r[0])
        for r in seed.app.state.evaluation_database.connection.execute(
            "SELECT envelope FROM outbox WHERE event_type='deployment.changed.v1'"
        )
    ]
    assert len(events) == 8
    for state in ["candidate", "evaluating", "approved", "shadow"]:
        selected = [e for e in events if e.data.new_state == state]
        assert {e.tenant_id for e in selected} == {"synthetic-a", "synthetic-b"}
        assert len({e.trace_id for e in selected}) == 1
    assert len({e.event_id for e in events}) == 8
    assert all(
        e.data.reason == "SYNTHETIC_PRIVATE_NOTE"
        for e in events
        if e.data.new_state in {"approved", "shadow"}
    )
    control = seed.app.state.evaluation_database.connection
    history = [
        json.loads(r[0])
        for r in control.execute(
            "SELECT data FROM lifecycle_transitions WHERE model_version=?", (job.model_version,)
        )
    ]
    assert [t["reason"] for t in history if t["to_state"] in {"approved", "shadow"}] == [
        "SYNTHETIC_PRIVATE_NOTE",
        "SYNTHETIC_PRIVATE_NOTE",
    ]
    stored_model = ModelManifest.model_validate_json(
        control.execute(
            "SELECT data FROM model_versions WHERE version=?", (job.model_version,)
        ).fetchone()[0]
    )
    assert stored_model.lifecycle_history[-1].reason == "SYNTHETIC_PRIVATE_NOTE"
    for file in (seed.directory / "models").rglob("*"):
        if file.is_file():
            for forbidden in [
                b"SYNTHETIC unique",
                b"SYNTHETIC ANSWER",
                b"Example Shop",
                b"SYNTHETIC_PRIVATE_NOTE",
            ]:
                assert forbidden not in file.read_bytes()
    elapsed = perf_counter() - start
    assert elapsed < 10
    print(f"training smoke: train/evaluate/approve/shadow elapsed_s={elapsed:.3f}")


def test_approval_current_policy_and_immutable_directory(evaluation_seed: "EvaluationSeed") -> None:
    seed = evaluation_seed
    spec = spec_for(seed)
    path = seed.directory / "datasets" / seed.manifest.dataset_id / seed.manifest.version
    original = {p.name: p.read_bytes() for p in path.iterdir()}
    body = spec.model_dump(mode="json")
    assert seed.client.post("/v1/training/jobs", headers=OPERATOR, json=body).status_code == 409
    approve(seed)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == original
    seed.app.state.training.policy.training["synthetic-b"] = False
    response = seed.client.post("/v1/training/jobs", headers=OPERATOR, json=body)
    assert (
        response.status_code == 403 and response.json()["error"]["code"] == "training_policy_denied"
    )
    assert not (seed.directory / "models").exists()
    assert (
        seed.app.state.evaluation_database.connection.execute(
            "SELECT count(*) FROM training_jobs"
        ).fetchone()[0]
        == 0
    )
    seed.app.state.training.policy.training["synthetic-b"] = True
    job = train(seed, spec)
    assert train(seed, spec) == job
    conflict = spec.model_copy(update={"seed": 99})
    assert (
        seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=conflict.model_dump(mode="json")
        ).status_code
        == 409
    )
    events = seed.app.state.evaluation_database.connection.execute(
        "SELECT count(*) FROM outbox WHERE event_type='training.completed.v1'"
    ).fetchone()[0]
    assert events == 2


def test_resume_and_reproducible_artifacts(evaluation_seed: "EvaluationSeed") -> None:
    seed = evaluation_seed
    approve(seed)
    spec = spec_for(seed)
    service = seed.app.state.training
    trainer = FakeTrainer(seed.directory, service.cipher, service.keyring, fail_after_checkpoint=1)
    service.trainer = trainer
    result = seed.client.post(
        "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert result.status_code == 200
    assert result.json()["state"] == "queued"
    failed = wait_training(seed.client, spec.job_id, OPERATOR).model_dump(mode="json")
    assert failed["state"] == "failed" and failed["checkpoint_refs"] == ["checkpoint-1"]
    assert failed["failure_code"] == "training_interrupted"
    # A new trainer instance represents a restarted process, using durable checkpoint metadata.
    replacement = FakeTrainer(seed.directory, service.cipher, service.keyring)
    service.trainer = replacement
    resumed = train(seed, spec)
    assert resumed.model_version == failed["model_version"]
    assert replacement.executed_steps == [2]
    fresh = train(seed, spec.model_copy(update={"job_id": uid()}))
    assert resumed.artifact_digest == fresh.artifact_digest
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    models = [seed.app.state.registry.get(j.model_version, identity) for j in [resumed, fresh]]
    manifests = []
    for model in models:
        verify_artifact(model, seed.directory / model.storage_location, service.keyring)
        manifests.append(
            model.model_dump(
                exclude={
                    "version",
                    "created_at",
                    "training_job_id",
                    "storage_location",
                    "artifact_mac",
                    "lifecycle_history",
                }
            )
        )
    assert manifests[0] == manifests[1]
    for name in models[0].artifact_hashes:
        assert (seed.directory / models[0].storage_location / name).read_bytes() == (
            seed.directory / models[1].storage_location / name
        ).read_bytes()


def test_promotion_exact_model_failure_and_replaced_baseline(
    evaluation_seed: "EvaluationSeed",
) -> None:
    seed = evaluation_seed
    approve(seed)
    baseline = evaluate(seed)
    first, second = train(seed), train(seed)
    failed = evaluate(seed, first.model_version, minimum_sample_size=99)
    passed = evaluate(seed, second.model_version)
    assert not failed.passed and passed.passed
    for report in [baseline, failed, passed]:
        response = seed.client.post(
            f"/v1/models/{first.model_version}/promotion-requests",
            headers=OPERATOR,
            json={
                "model_version": first.model_version,
                "target_state": "approved",
                "reason": "synthetic",
                "evaluation_id": report.specification.evaluation_id,
            },
        )
        assert response.status_code == 409
    evaluate(seed, replace=True, operator_note="synthetic replacement")
    response = seed.client.post(
        f"/v1/models/{second.model_version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": second.model_version,
            "target_state": "approved",
            "reason": "synthetic",
            "evaluation_id": passed.specification.evaluation_id,
        },
    )
    assert response.status_code == 409


def test_rollback_after_baseline_replacement_and_transactional_events(
    evaluation_seed: "EvaluationSeed", monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = evaluation_seed
    approve(seed)
    baseline = evaluate(seed)
    versions = []
    for _ in range(2):
        job = train(seed)
        report = evaluate(seed, job.model_version)
        promote(seed, job.model_version, "approved", report.specification.evaluation_id)
        for state in ["shadow", "canary", "production"]:
            promote(seed, job.model_version, state)
        versions.append(job.model_version)
        if len(versions) == 1:
            assert (
                seed.client.post(
                    "/v1/deployments/synthetic-specialist/rollback",
                    headers=OPERATOR,
                    json={"reason": "synthetic"},
                ).status_code
                == 409
            )
            replacement = evaluate(
                seed, replace=True, operator_note="SYNTHETIC baseline replacement"
            )
            assert replacement.passed
            assert replacement.specification.evaluation_id != baseline.specification.evaluation_id
    db = seed.app.state.evaluation_database.connection
    count = db.execute("SELECT count(*) FROM outbox").fetchone()[0]
    last_sequence = db.execute("SELECT max(sequence) FROM outbox").fetchone()[0]
    original = seed.app.state.evaluation_outbox.enqueue
    calls = 0

    def fail_if_evaluation_rechecked(*args):
        raise AssertionError("rollback_must_not_recheck_evaluation")

    monkeypatch.setattr(seed.app.state.registry, "_passed", fail_if_evaluation_rechecked)

    def fail_second(events, limit):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic_failure")
        return original(events, limit)

    monkeypatch.setattr(seed.app.state.evaluation_outbox, "enqueue", fail_second)
    assert (
        seed.client.post(
            "/v1/deployments/synthetic-specialist/rollback",
            headers=OPERATOR,
            json={"reason": "synthetic"},
        ).status_code
        == 503
    )
    assert db.execute("SELECT current_version FROM deployments").fetchone()[0] == versions[1]
    assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == count
    monkeypatch.setattr(seed.app.state.evaluation_outbox, "enqueue", original)
    result = seed.client.post(
        "/v1/deployments/synthetic-specialist/rollback",
        headers=OPERATOR,
        json={"reason": "synthetic"},
    )
    assert result.status_code == 200, result.json()
    assert [m["state"] for m in result.json()] == ["deprecated", "production"]
    assert db.execute("SELECT current_version FROM deployments").fetchone()[0] == versions[0]
    assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == count + 4
    events = [
        Event.model_validate_json(r[0])
        for r in db.execute("SELECT envelope FROM outbox WHERE sequence>?", (last_sequence,))
    ]
    assert all(e.event_type == "deployment.changed.v1" for e in events)
    for tenant in seed.manifest.tenant_ids:
        transitions = [e.data for e in events if e.tenant_id == tenant]
        assert [(t.model_version, t.new_state) for t in transitions] == [
            (versions[1], "deprecated"),
            (versions[0], "production"),
        ]
        assert all(t.reason == "synthetic" for t in transitions)


@pytest.mark.parametrize(
    "file",
    [
        "adapter_weights.bin",
        "adapter_config.json",
        "training_report.json",
        "unexpected.json",
    ],
)
def test_specialist_refuses_tampered_artifact(evaluation_seed: "EvaluationSeed", file: str) -> None:
    seed = evaluation_seed
    approve(seed)
    job = train(seed)
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    model = seed.app.state.registry.get(job.model_version, identity)
    path = seed.directory / model.storage_location
    SpecialistProvider(model, path, seed.app.state.keyring, seed.directory)
    (path / file).write_bytes(b"SYNTHETIC tamper")
    with pytest.raises(GatewayError, match="artifact_integrity_failed"):
        SpecialistProvider(model, path, seed.app.state.keyring, seed.directory)
    response = seed.client.post(
        "/v1/evaluations",
        headers=OPERATOR,
        json=seed.request.model_copy(
            update={"candidate_deployment_id": job.model_version}
        ).model_dump(mode="json"),
    )
    assert response.status_code == 409


def test_training_cli_uses_same_jobs_and_limits_listing(
    evaluation_seed: "EvaluationSeed", capsys: pytest.CaptureFixture[str]
) -> None:
    from adaptive_llm.training.__main__ import main

    seed = evaluation_seed
    approve(seed)
    spec = spec_for(seed)
    path = seed.directory / "train.json"
    path.write_text(spec.model_dump_json())
    policy = seed.directory / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "policy_version": "synthetic-dataset-policy-1",
                "tenants": {
                    t: {
                        "processing_allowed": True,
                        "training_allowed": True,
                        "retention_seconds": 3600,
                    }
                    for t in seed.manifest.tenant_ids
                },
            }
        )
    )
    main(["train", "--spec", str(path), "--policy", str(policy), "--data-dir", str(seed.directory)])
    job = TrainingJob.model_validate_json(capsys.readouterr().out)
    assert job == train(seed, spec)
    main(["models", "--data-dir", str(seed.directory)])
    listing = json.loads(capsys.readouterr().out)
    assert set(listing[0]) == {"registry_id", "version", "state", "evaluation_ids"}
    main(
        [
            "promote",
            "--model",
            job.model_version,
            "--to",
            "evaluating",
            "--note",
            "synthetic",
            "--data-dir",
            str(seed.directory),
        ]
    )
    assert json.loads(capsys.readouterr().out)["state"] == "evaluating"
    with pytest.raises(SystemExit) as error:
        main(["models", "--operator-key", "synthetic-key-a", "--data-dir", str(seed.directory)])
    assert error.value.code == 1 and capsys.readouterr().err == "training_control_failed\n"


def test_training_worker_keeps_serving_available_and_excludes_duplicate_runner(
    evaluation_seed: "EvaluationSeed",
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event as Signal

    seed = evaluation_seed
    approve(seed)
    entered, release = Signal(), Signal()
    original = seed.app.state.training.trainer

    class BlockingTrainer:
        architecture = original.architecture

        def train(self, *args):
            entered.set()
            assert release.wait(5)
            return original.train(*args)

    seed.app.state.training.trainer = BlockingTrainer()
    spec = spec_for(seed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            seed.client.post,
            "/v1/training/jobs",
            headers=OPERATOR,
            json=spec.model_dump(mode="json"),
        )
        try:
            assert entered.wait(3)
            assert (
                seed.client.get(f"/v1/training/jobs/{spec.job_id}", headers=OPERATOR).json()[
                    "state"
                ]
                == "running"
            )
            duplicate = seed.client.post(
                "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
            )
            assert duplicate.status_code == 200
            assert duplicate.json()["state"] == "running"
            serving = executor.submit(
                seed.client.post,
                "/v1/inference",
                headers={"Authorization": "Bearer synthetic-key-a"},
                json={
                    "request_id": uid(),
                    "application_id": "support-assistant",
                    "messages": [{"role": "user", "content": "SYNTHETIC during training"}],
                },
            )
            assert serving.result(timeout=1).status_code == 200
            assert future.result(timeout=1).json()["state"] == "queued"
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200


def test_training_cli_failed_job_exits_nonzero(evaluation_seed, monkeypatch, capsys):
    from dataclasses import replace

    from adaptive_llm.app import create_app
    from adaptive_llm.training.__main__ import main

    seed = evaluation_seed
    approve(seed)
    seed.app.state.training.stop()

    class FailingTrainer:
        architecture = "deterministic-fake-adapter-v1"

        def train(self, *args):
            raise RuntimeError("SYNTHETIC_PRIVATE_ERROR")

    monkeypatch.setattr(
        "adaptive_llm.training.__main__.create_app",
        lambda settings: create_app(replace(seed.app.state.settings, trainer=FailingTrainer())),
    )
    path = seed.directory / "failed-training.json"
    path.write_text(spec_for(seed).model_dump_json())
    with pytest.raises(SystemExit) as error:
        main(["train", "--spec", str(path), "--data-dir", str(seed.directory)])
    output = capsys.readouterr()
    assert error.value.code == 1 and output.err == "training_control_failed\n"
    assert TrainingJob.model_validate_json(output.out).failure_code == "training_failed"


@pytest.mark.parametrize("field", ["dataset_version", "dataset_id", "candidate_artifact_digest"])
def test_promotion_rechecks_exact_dataset_and_artifact(
    evaluation_seed: "EvaluationSeed", monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    seed = evaluation_seed
    approve(seed)
    evaluate(seed)
    job = train(seed)
    report = evaluate(seed, job.model_version)
    registry = seed.app.state.registry
    get = registry.evaluations.get
    if field.startswith("dataset_"):
        wrong = report.model_copy(
            update={"specification": report.specification.model_copy(update={field: uid()})}
        )
    else:
        wrong = report.model_copy(update={field: "synthetic-wrong-digest"})

    def wrong_report(evaluation_id, identity):
        return (
            wrong
            if evaluation_id == report.specification.evaluation_id
            else get(evaluation_id, identity)
        )

    monkeypatch.setattr(registry.evaluations, "get", wrong_report)
    response = seed.client.post(
        f"/v1/models/{job.model_version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": job.model_version,
            "target_state": "approved",
            "reason": "synthetic",
            "evaluation_id": report.specification.evaluation_id,
        },
    )
    assert response.status_code == 409


def test_corrupt_checkpoint_refuses_resume_and_publication_failure_recovers(
    evaluation_seed: "EvaluationSeed", monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = evaluation_seed
    approve(seed)
    service = seed.app.state.training
    spec = spec_for(seed)
    service.trainer.fail_after_checkpoint = 1
    job = train(seed, spec)
    assert job.state == "failed"
    checkpoint = (
        seed.directory
        / "models"
        / spec.registry_id
        / f".{job.model_version}.training/checkpoint-1/adapter_weights.bin"
    )
    original_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(b"SYNTHETIC corrupt checkpoint")
    assert train(seed, spec).failure_code == "checkpoint_integrity_failed"
    checkpoint.write_bytes(original_bytes)
    enqueue = seed.app.state.evaluation_outbox.enqueue

    def failed_completion(events, limit):
        if any(
            e.event_type == "training.completed.v1" and e.data.status == "succeeded" for e in events
        ):
            raise RuntimeError("synthetic_failure")
        return enqueue(events, limit)

    monkeypatch.setattr(seed.app.state.evaluation_outbox, "enqueue", failed_completion)
    assert train(seed, spec).state == "failed"
    db = seed.app.state.evaluation_database.connection
    assert db.execute("SELECT count(*) FROM model_versions").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM lifecycle_transitions").fetchone()[0] == 0
    assert not (seed.directory / "models" / spec.registry_id / job.model_version).exists()
    archive = seed.directory / "models" / spec.registry_id / ".checkpoints" / job.model_version
    assert {p.name for p in archive.iterdir()} == {"checkpoint-1", "checkpoint-2"}
    # Simulate a crash partway through restoring an archived checkpoint inventory.
    (archive / "checkpoint-1").rename(checkpoint.parent)
    assert (archive / "checkpoint-2").is_dir() and checkpoint.is_file()
    monkeypatch.setattr(seed.app.state.evaluation_outbox, "enqueue", enqueue)
    resumed = train(seed, spec)
    assert resumed.state == "succeeded"
    assert {p.name for p in archive.iterdir()} == set(resumed.checkpoint_refs)
    assert not list((seed.directory / resumed.artifact_ref).glob("checkpoint-*"))


def test_registration_uses_and_checks_trainer_architecture(
    evaluation_seed: "EvaluationSeed",
) -> None:
    seed = evaluation_seed
    approve(seed)
    # Only the declaration changes here: real PEFT training remains slice 3b.
    seed.app.state.training.trainer.architecture = "lora-peft-v1"
    job = train(seed)
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    registry = seed.app.state.registry
    model = registry.get(job.model_version, identity)
    assert model.adapter_architecture == "lora-peft-v1"
    verify_artifact(model, seed.directory / model.storage_location, seed.app.state.keyring)
    with pytest.raises(GatewayError, match="^trainer_architecture_mismatch$"):
        registry.complete(job, model, trainer_architecture="deterministic-fake-adapter-v1")
    assert registry.models(identity) == [model]


@pytest.mark.parametrize("lineage_count", [0, 2])
def test_registration_and_approval_refuse_unsupported_lineage(
    evaluation_seed: "EvaluationSeed", lineage_count: int
) -> None:
    seed = evaluation_seed
    approve(seed)
    job = train(seed)
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    registry = seed.app.state.registry
    model = registry.get(job.model_version, identity)
    mixed = model.model_copy(
        update={
            "datasets": [
                model.datasets[0].model_copy(update={"version": uid()})
                for _ in range(lineage_count)
            ]
        }
    )
    with pytest.raises(GatewayError, match="^single_dataset_required$"):
        registry.complete(
            job, mixed, trainer_architecture=seed.app.state.training.trainer.architecture
        )
    with pytest.raises(GatewayError, match="^single_dataset_required$"):
        registry._passed(mixed, None, identity)
    assert registry.models(identity) == [model]


@pytest.mark.parametrize(
    "failure", ["missing_approval", "wrong_approval_edge", "revoked", "artifact"]
)
def test_rollback_still_requires_previous_approval_history_and_valid_artifact(
    evaluation_seed: "EvaluationSeed", failure: str
) -> None:
    seed = evaluation_seed
    approve(seed)
    evaluate(seed)
    versions = []
    for _ in range(2):
        job = train(seed)
        report = evaluate(seed, job.model_version)
        promote(seed, job.model_version, "approved", report.specification.evaluation_id)
        for state in ["shadow", "canary", "production"]:
            promote(seed, job.model_version, state)
        versions.append(job.model_version)
    registry = seed.app.state.registry
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    previous = registry.get(versions[0], identity)
    db = seed.app.state.evaluation_database.connection
    if failure == "artifact":
        (seed.directory / previous.storage_location / "adapter_weights.bin").write_bytes(
            b"synthetic-tamper"
        )
    elif failure == "revoked":
        promote(seed, previous.version, "revoked")
    else:
        history = [t for t in previous.lifecycle_history if t.to_state != "approved"]
        if failure == "wrong_approval_edge":
            history = [
                t.model_copy(update={"from_state": "candidate"}) if t.to_state == "approved" else t
                for t in previous.lifecycle_history
            ]
        changed = previous.model_copy(update={"lifecycle_history": history})
        db.execute(
            "UPDATE model_versions SET data=? WHERE version=?",
            (changed.model_dump_json(), previous.version),
        )
    count = db.execute("SELECT count(*) FROM outbox").fetchone()[0]
    response = seed.client.post(
        "/v1/deployments/synthetic-specialist/rollback",
        headers=OPERATOR,
        json={"reason": "SYNTHETIC emergency rollback"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == (
        "artifact_integrity_failed" if failure == "artifact" else "rollback_version_unavailable"
    )
    assert db.execute("SELECT current_version FROM deployments").fetchone()[0] == versions[1]
    assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == count
    assert registry.get(versions[1], identity).state == "production"
