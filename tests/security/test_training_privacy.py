import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from conftest import wait_training

from adaptive_llm.contracts import TrainingJobSpecification
from adaptive_llm.gateway.identity import GatewayError

if TYPE_CHECKING:
    from conftest import EvaluationSeed

OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}
USER = {"Authorization": "Bearer synthetic-key-a"}


def prepare(seed: "EvaluationSeed") -> TrainingJobSpecification:
    seed.app.state.training.policy.training["synthetic-b"] = True
    result = seed.client.post(
        f"/v1/datasets/{seed.manifest.dataset_id}/versions/{seed.manifest.version}/approval",
        headers=OPERATOR,
        json={"reason": "SYNTHETIC_PRIVATE_APPROVAL"},
    )
    assert result.status_code == 200
    assert result.json()["approval"]["reason"] == "SYNTHETIC_PRIVATE_APPROVAL"
    return TrainingJobSpecification(
        dataset_id=seed.manifest.dataset_id, dataset_version=seed.manifest.version
    )


def test_operator_auth_and_trusted_actor(evaluation_seed: "EvaluationSeed") -> None:
    seed = evaluation_seed
    spec = prepare(seed)
    for headers, code in [({}, 401), (USER, 403)]:
        assert (
            seed.client.post(
                "/v1/training/jobs", headers=headers, json=spec.model_dump(mode="json")
            ).status_code
            == code
        )
        assert (
            seed.client.get(f"/v1/training/jobs/{spec.job_id}", headers=headers).status_code == code
        )
        assert (
            seed.client.post(f"/v1/training/jobs/{spec.job_id}/cancel", headers=headers).status_code
            == code
        )
        assert (
            seed.client.post(
                f"/v1/datasets/{seed.manifest.dataset_id}/versions/{seed.manifest.version}/approval",
                headers=headers,
                json={"reason": "synthetic"},
            ).status_code
            == code
        )
    job = seed.client.post(
        "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
    ).json()
    job = wait_training(seed.client, spec.job_id, OPERATOR).model_dump(mode="json")
    model = job["model_version"]
    body = {"model_version": model, "target_state": "evaluating", "reason": "synthetic"}
    for headers, code in [({}, 401), (USER, 403)]:
        assert (
            seed.client.post(
                f"/v1/models/{model}/promotion-requests", headers=headers, json=body
            ).status_code
            == code
        )
        assert (
            seed.client.post(
                "/v1/deployments/synthetic-specialist/rollback",
                headers=headers,
                json={"reason": "synthetic"},
            ).status_code
            == code
        )
    assert (
        seed.client.post(
            f"/v1/models/{model}/promotion-requests",
            headers=OPERATOR,
            json={**body, "actor": "forged-actor"},
        ).status_code
        == 403
    )
    identity = seed.app.state.authenticator.authenticate(OPERATOR["Authorization"], None)
    limited = replace(identity, dataset_tenants=frozenset({"synthetic-a"}))
    with pytest.raises(GatewayError, match="training_job_not_found"):
        seed.app.state.registry.job(spec.job_id, limited)
    with pytest.raises(GatewayError, match="training_job_not_found"):
        seed.app.state.registry.cancel_job(spec.job_id, limited)
    with pytest.raises(GatewayError, match="model_not_found"):
        seed.app.state.registry.get(model, limited)
    assert seed.app.state.registry.models(limited) == []
    with pytest.raises(GatewayError, match="dataset_not_found"):
        seed.app.state.training.run(spec, limited)
    assert (
        seed.client.post(
            f"/v1/models/{model}/promotion-requests", headers=OPERATOR, json={**body, "reason": " "}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("artifact", ["manifest", "shard", "approval"])
def test_training_rejects_tampered_dataset(
    evaluation_seed: "EvaluationSeed", artifact: str
) -> None:
    seed = evaluation_seed
    spec = prepare(seed)
    directory = seed.directory / "datasets" / seed.manifest.dataset_id / seed.manifest.version
    if artifact == "manifest":
        path = directory / "manifest.json"
        data = json.loads(path.read_text())
        data["signature"] = "synthetic-forgery"
        path.write_text(json.dumps(data))
    elif artifact == "shard":
        (directory / "synthetic-a.train.jsonl.enc").write_text("synthetic-forgery")
    else:
        db = seed.app.state.database.connection
        row = db.execute("SELECT data FROM dataset_manifests").fetchone()
        data = json.loads(row[0])
        data["approval"]["signature"] = "synthetic-forgery"
        db.execute("UPDATE dataset_manifests SET data=?", (json.dumps(data),))
    result = seed.client.post(
        "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert result.status_code == 409
    assert result.json() == {"error": {"code": "invalid_dataset_artifact"}}
    assert not (seed.directory / "models").exists()


def test_training_errors_and_examples_do_not_retain_content(
    evaluation_seed: "EvaluationSeed", caplog: pytest.LogCaptureFixture
) -> None:
    seed = evaluation_seed
    spec = prepare(seed)

    class FailingTrainer:
        architecture = "synthetic-failing-trainer-v1"

        def train(self, *args):
            raise GatewayError(500, "SYNTHETIC_PRIVATE_TRAINER_BODY")

    seed.app.state.training.trainer = FailingTrainer()
    result = seed.client.post(
        "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert result.status_code == 200
    assert wait_training(seed.client, spec.job_id, OPERATOR).failure_code == "training_failed"
    job = seed.client.get(f"/v1/training/jobs/{spec.job_id}", headers=OPERATOR)
    assert job.json()["failure_code"] == "training_failed"
    example = "SYNTHETIC unique case 0 unused receipt"
    error_body = "SYNTHETIC_PRIVATE_TRAINER_BODY"
    assert example not in caplog.text and error_body not in caplog.text
    for file in seed.directory.rglob("*"):
        if file.is_file():
            content = file.read_bytes()
            assert example.encode() not in content
            assert error_body.encode() not in content
