from dataclasses import replace
from threading import Event

import pytest
from conftest import EvaluationSeed, wait_training
from fastapi.testclient import TestClient
from test_training import OPERATOR, approve, spec_for, train

from adaptive_llm.app import create_app
from adaptive_llm.contracts import TrainingJob, now
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.training.fake import FakeTrainer


def test_queue_is_immediate_serial_and_cancellation_is_durable(evaluation_seed: EvaluationSeed):
    seed = evaluation_seed
    approve(seed)
    entered, release = Event(), Event()
    delegate = seed.app.state.training.trainer

    class BlockingTrainer:
        architecture = delegate.architecture

        def train(self, *args):
            entered.set()
            assert release.wait(5)
            return delegate.train(*args)

    seed.app.state.training.trainer = BlockingTrainer()
    first, second = spec_for(seed), spec_for(seed)
    try:
        response = seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=first.model_dump(mode="json")
        )
        assert response.json()["state"] == "queued"
        assert entered.wait(3)
        response = seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=second.model_dump(mode="json")
        )
        assert response.json()["state"] == "queued"
        for spec in (second, first):
            cancelled = seed.client.post(
                f"/v1/training/jobs/{spec.job_id}/cancel", headers=OPERATOR
            )
            assert cancelled.status_code == 200
            assert cancelled.json()["cancel_requested"]
        assert (
            seed.client.get(f"/v1/training/jobs/{second.job_id}", headers=OPERATOR).json()["state"]
            == "cancelled"
        )
    finally:
        release.set()
    assert wait_training(seed.client, first.job_id, OPERATOR).state == "cancelled"
    for spec in (first, second):
        for _ in range(2):
            assert (
                seed.client.post(
                    f"/v1/training/jobs/{spec.job_id}/cancel", headers=OPERATOR
                ).json()["state"]
                == "cancelled"
            )
    db = seed.app.state.evaluation_database.connection
    assert db.execute("SELECT count(*) FROM model_versions").fetchone()[0] == 0
    assert (
        db.execute(
            "SELECT count(*) FROM outbox WHERE event_type='training.completed.v1'"
        ).fetchone()[0]
        == 4
    )


def test_restart_resumes_running_and_restarts_without_checkpoint(evaluation_seed: EvaluationSeed):
    seed = evaluation_seed
    approve(seed)
    service = seed.app.state.training
    service.trainer = FakeTrainer(
        seed.directory, service.cipher, service.keyring, fail_after_checkpoint=1
    )
    spec = spec_for(seed)
    failed = train(seed, spec)
    assert failed.failure_code == "training_interrupted"
    service.stop()
    registry = seed.app.state.registry
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    registry.save_job(
        failed.model_copy(update={"state": "running", "completed_at": None}),
        seed.manifest.tenant_ids,
    )
    second = service.submit(spec_for(seed), identity)
    registry.save_job(
        second.model_copy(update={"state": "running", "started_at": now()}),
        seed.manifest.tenant_ids,
    )
    replacement = FakeTrainer(seed.directory, service.cipher, service.keyring)
    app = create_app(replace(seed.app.state.settings, trainer=replacement))
    with TestClient(app) as client:
        resumed = wait_training(client, spec.job_id, OPERATOR)
        restarted = wait_training(client, second.specification.job_id, OPERATOR)
        assert resumed.state == restarted.state == "succeeded"
        assert resumed.model_version == failed.model_version
        assert replacement.executed_steps == [2, 1, 2]


def test_queued_policy_is_rechecked_and_queue_continues(evaluation_seed: EvaluationSeed):
    seed = evaluation_seed
    approve(seed)
    service = seed.app.state.training
    service.stop()
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    job = service.submit(spec_for(seed), identity)
    service.policy.training["synthetic-b"] = False
    service.stopping.clear()
    service.work_once()
    result: TrainingJob = seed.app.state.registry.job(job.specification.job_id, identity)
    assert result.state == "failed" and result.failure_code == "training_eligibility_failed"
    assert not (seed.directory / "models").exists()
    service.policy.training["synthetic-b"] = True
    assert train(seed).state == "succeeded"


def test_worker_and_retries_preserve_queued_trainer_architecture(evaluation_seed: EvaluationSeed):
    seed = evaluation_seed
    approve(seed)
    service = seed.app.state.training
    service.stop()
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    job = service.submit(spec_for(seed), identity)
    # A 3a job has no authenticated submitter until an operator resubmits it.
    db = seed.app.state.evaluation_database.connection
    db.execute("DELETE FROM training_submitters WHERE job_id=?", (job.specification.job_id,))
    assert seed.app.state.registry.pending_jobs() == []
    assert service.submit(job.specification, identity) == job
    service.trainer.architecture = "synthetic-other-trainer"
    service.stopping.clear()
    service.work_once()
    assert seed.app.state.registry.job(job.specification.job_id, identity).state == "queued"
    with pytest.raises(GatewayError, match="training_trainer_mismatch"):
        service.submit(job.specification, identity)
    service.trainer.architecture = job.trainer_architecture
    service.work_once()
    assert wait_training(seed.client, job.specification.job_id, OPERATOR).state == "succeeded"
