import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    Chunk,
    DatasetManifest,
    DatasetSpecification,
    EvaluationInput,
    InferenceRequest,
    Message,
    PolicyDecision,
    SourceWindow,
    TimeSplit,
    TrainingJob,
    now,
    uid,
)
from adaptive_llm.gateway.identity import Identity, Keyring, LocalAuthenticator
from adaptive_llm.providers import ProviderRequest


@pytest.fixture(autouse=True)
def metrics_listener(monkeypatch):
    # API tests use an in-process client. Dedicated observability tests exercise real sockets.
    server, thread = Mock(), Mock()
    start = Mock(return_value=(server, thread))
    monkeypatch.setattr("adaptive_llm.app.start_http_server", start)
    return start


class DatasetPolicy:
    """Synthetic mutable policy for revocation tests; stored decisions are deliberately stale."""

    def __init__(self) -> None:
        self.training = {"synthetic-a": True, "synthetic-b": False}
        self.logging = True

    def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
        return PolicyDecision(
            policy_version="synthetic-dataset-policy-1",
            retention_seconds=3600,
            training_allowed=self.training[identity.tenant_id],
            content_logging_allowed=self.logging,
        )


@dataclass
class DatasetSeed:
    app: FastAPI
    client: TestClient
    specification: DatasetSpecification
    policy: DatasetPolicy
    ids: list[str]
    subjects: list[str]
    directory: Path


@pytest.fixture
def dataset_seed(tmp_path: Path) -> Iterator[DatasetSeed]:
    policy = DatasetPolicy()
    app = create_app(Settings(data_dir=tmp_path, policy=policy, outbox_dispatch_enabled=False))
    ids: list[str] = []
    subjects: list[str] = []
    rng = random.Random(23)
    start = now() - timedelta(seconds=1)
    with TestClient(app) as client:
        for index in range(60):
            tenant = "a" if index < 40 else "b"
            subject = f"synthetic-dataset-subject-{index // 2}"
            subjects.append(subject)
            headers = {"Authorization": f"Bearer synthetic-key-{tenant}", "X-Subject": subject}
            vocabulary = " ".join(
                "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=9)) for _ in range(24)
            )
            response = client.post(
                "/v1/inference",
                headers=headers,
                json={
                    "request_id": f"synthetic-dataset-{index}",
                    "application_id": "support-assistant",
                    "messages": [
                        {"role": "user", "content": f"SYNTHETIC {vocabulary} unused receipt"}
                    ],
                    "rag": {"enabled": index >= 30, "index_id": "synthetic-kb"},
                },
            )
            assert response.status_code == 200
            iid = response.json()["interaction_id"]
            ids.append(iid)
            if index % 6 in {0, 1}:
                assert (
                    client.post(
                        f"/v1/interactions/{iid}/feedback",
                        headers=headers,
                        json={"label_type": "thumb", "value": {"score": 0, "max_score": 1}},
                    ).status_code
                    == 200
                )
            if index % 6 == 1:
                assert (
                    client.post(
                        f"/v1/interactions/{iid}/correction",
                        headers=headers,
                        json={
                            "correction": f"SYNTHETIC approved correction {vocabulary}",
                            "training_authorised": True,
                        },
                    ).status_code
                    == 200
                )
            if index % 6 == 2:
                assert (
                    client.post(
                        f"/v1/interactions/{iid}/feedback",
                        headers=headers,
                        json={"label_type": "resolution", "value": {"score": 1, "max_score": 1}},
                    ).status_code
                    == 200
                )
        assert (
            client.post(
                "/v1/privacy/subjects/deletion-requests",
                headers={"Authorization": "Bearer synthetic-key-a"},
                json={"subject": subjects[38]},
            ).json()["deleted"]
            == 2
        )
        spec = DatasetSpecification(
            dataset_id="synthetic-training",
            tenant_ids=["synthetic-a", "synthetic-b"],
            source_window=SourceWindow(start=start, end=now() + timedelta(seconds=1)),
            eligibility_policy_version="synthetic-dataset-policy-1",
            minimum_examples=5,
            seed=7,
        )
        yield DatasetSeed(app, client, spec, policy, ids, subjects, tmp_path)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "drill: measured local resilience and recovery drills")
    config.addinivalue_line("markers", "smoke: CPU-only training and evaluation under ten seconds")


def wait_training(client: TestClient, job_id: str, headers: dict[str, str]) -> TrainingJob:
    deadline = monotonic() + 60
    while monotonic() < deadline:
        response = client.get(f"/v1/training/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        job = TrainingJob.model_validate(response.json())
        if job.state not in {"queued", "running"}:
            return job
        sleep(0.01)
    raise AssertionError("training_poll_timeout")


@pytest.fixture(autouse=True)
def isolated_default_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each app gets a fresh database unless a test explicitly shares data_dir (restart tests).
    monkeypatch.setattr("adaptive_llm.app._default_data_dir", lambda: tmp_path / uid())


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def keyring(settings: Settings) -> Keyring:
    assert settings.secret is not None
    return Keyring(settings.secret)


@pytest.fixture
def identity(settings: Settings, keyring: Keyring) -> Identity:
    return LocalAuthenticator(settings.identity_path, keyring).authenticate(
        "Bearer synthetic-key-a", "synthetic-subject"
    )


@pytest.fixture
def inference_request() -> InferenceRequest:
    return InferenceRequest(
        request_id="synthetic-request-1",
        application_id="support-assistant",
        messages=[Message(role="user", content="SYNTHETIC: unused items receipt return window")],
        rag={"enabled": True, "index_id": "synthetic-kb"},
    )


@pytest.fixture
def chunks(settings: Settings) -> tuple[Chunk, ...]:
    return tuple(
        Chunk.model_validate(row) for row in json.loads(settings.documents_path.read_text())
    )


@pytest.fixture
def provider_request(
    inference_request: InferenceRequest, chunks: tuple[Chunk, ...]
) -> ProviderRequest:
    return ProviderRequest(
        messages=tuple(inference_request.messages),
        context=(chunks[0],),
        response_format=inference_request.response_format,
        max_output_tokens=inference_request.max_output_tokens,
    )


@pytest.fixture
def documents_path(settings: Settings, tmp_path: Path) -> Path:
    path = tmp_path / "documents.json"
    path.write_text(settings.documents_path.read_text())
    return path


@pytest.fixture
def shadow_seed(evaluation_seed: "EvaluationSeed") -> tuple["EvaluationSeed", str]:
    seed = evaluation_seed
    headers = {"Authorization": "Bearer synthetic-operator-key"}
    seed.app.state.training.policy.training["synthetic-b"] = True
    assert (
        seed.client.post(
            f"/v1/datasets/{seed.manifest.dataset_id}/versions/{seed.manifest.version}/approval",
            headers=headers,
            json={"reason": "SYNTHETIC shadow test"},
        ).status_code
        == 200
    )
    from adaptive_llm.contracts import TrainingJobSpecification

    response = seed.client.post(
        "/v1/training/jobs",
        headers=headers,
        json=TrainingJobSpecification(
            dataset_id=seed.manifest.dataset_id,
            dataset_version=seed.manifest.version,
        ).model_dump(mode="json"),
    )
    job = wait_training(seed.client, response.json()["specification"]["job_id"], headers)
    assert job.state == "succeeded"
    assert seed.client.post(
        "/v1/evaluations",
        headers=headers,
        json=seed.request.model_dump(mode="json"),
    ).json()["passed"]
    candidate = seed.request.model_copy(
        update={
            "evaluation_id": uid(),
            "candidate_deployment_id": job.model_version,
        }
    )
    report = seed.client.post(
        "/v1/evaluations",
        headers=headers,
        json=candidate.model_dump(mode="json"),
    )
    assert report.status_code == 200 and report.json()["passed"]
    for state in ["approved", "shadow"]:
        assert (
            seed.client.post(
                f"/v1/models/{job.model_version}/promotion-requests",
                headers=headers,
                json={
                    "model_version": job.model_version,
                    "target_state": state,
                    "reason": "SYNTHETIC shadow test",
                    "evaluation_id": candidate.evaluation_id,
                },
            ).status_code
            == 200
        )
    return seed, job.model_version


@dataclass
class EvaluationSeed:
    app: FastAPI
    client: TestClient
    directory: Path
    manifest: DatasetManifest
    request: EvaluationInput


@dataclass
class RouterSeed:
    seed: EvaluationSeed
    specialist: str
    router: str
    dataset: DatasetManifest


@pytest.fixture
def router_seed(shadow_seed: tuple[EvaluationSeed, str]) -> RouterSeed:
    from adaptive_llm.contracts import RoutePolicy, TrainingJobSpecification

    seed, specialist = shadow_seed
    operator = {"Authorization": "Bearer synthetic-operator-key"}
    cheaper = TrainingJobSpecification(
        registry_id="synthetic-live",
        dataset_id=seed.manifest.dataset_id,
        dataset_version=seed.manifest.version,
        input_micros_per_1000_tokens=250,
        output_micros_per_1000_tokens=500,
    )
    assert (
        seed.client.post(
            "/v1/training/jobs", headers=operator, json=cheaper.model_dump(mode="json")
        ).status_code
        == 200
    )
    trained = wait_training(seed.client, cheaper.job_id, operator)
    assert trained.state == "succeeded"
    specialist = trained.model_version
    candidate = seed.request.model_copy(
        update={"evaluation_id": uid(), "candidate_deployment_id": specialist}
    )
    assert seed.client.post(
        "/v1/evaluations", headers=operator, json=candidate.model_dump(mode="json")
    ).json()["passed"]
    for state in ["approved", "shadow"]:
        assert (
            seed.client.post(
                f"/v1/models/{specialist}/promotion-requests",
                headers=operator,
                json={
                    "model_version": specialist,
                    "target_state": state,
                    "evaluation_id": candidate.evaluation_id,
                    "reason": "SYNTHETIC cheaper live specialist",
                },
            ).status_code
            == 200
        )
    policy = RoutePolicy(
        eligible_specialist_versions=[specialist],
        shadow_enabled=True,
        tenant_enabled={"synthetic-a": True},
        task_enabled={"general": True},
    )
    assert (
        seed.client.post(
            "/v1/route-policies", headers=operator, json=policy.model_dump(mode="json")
        ).status_code
        == 200
    )
    assert (
        seed.client.post(
            f"/v1/route-policies/{policy.policy_id}/activate",
            headers=operator,
            json={"reason": "SYNTHETIC router observations"},
        ).status_code
        == 200
    )
    start = now()
    cutoffs = []
    for index in range(60):
        response = seed.client.post(
            "/v1/inference",
            headers={
                "Authorization": "Bearer synthetic-key-a",
                "X-Subject": f"synthetic-router-{index}",
            },
            json={
                "request_id": uid(),
                "application_id": "support-assistant",
                "max_output_tokens": 32,
                "messages": [{"role": "user", "content": f"SYNTHETIC routing example {index}"}],
            },
        )
        assert response.status_code == 200
        if index in {39, 49}:
            cutoffs.append(now())
    assert seed.client.portal is not None
    seed.client.portal.call(seed.app.state.shadow.queue.join)
    specification = DatasetSpecification(
        dataset_id="synthetic-router",
        purpose="router_training",
        tenant_ids=seed.manifest.tenant_ids,
        source_dataset_id=seed.manifest.dataset_id,
        source_dataset_version=seed.manifest.version,
        source_window=SourceWindow(start=start, end=now()),
        eligibility_policy_version="synthetic-dataset-policy-1",
        time_split=TimeSplit(train_end=cutoffs[0], validation_end=cutoffs[1]),
    )
    result = seed.client.post(
        "/v1/datasets/builds", headers=operator, json=specification.model_dump(mode="json")
    )
    assert result.status_code == 200, result.json()
    dataset = DatasetManifest.model_validate(result.json())
    assert dataset.examples == {"train": 40, "validation": 10, "test": 10}
    assert (
        seed.client.post(
            f"/v1/datasets/{dataset.dataset_id}/versions/{dataset.version}/approval",
            headers=operator,
            json={"reason": "SYNTHETIC routing dataset"},
        ).status_code
        == 200
    )
    specification_job = TrainingJobSpecification(
        job_type="router",
        registry_id="synthetic-router",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
    )
    result = seed.client.post(
        "/v1/training/jobs", headers=operator, json=specification_job.model_dump(mode="json")
    )
    assert result.status_code == 200, result.json()
    job = wait_training(seed.client, specification_job.job_id, operator)
    assert job.state == "succeeded", job.failure_code
    evaluation = EvaluationInput(
        candidate_deployment_id=job.model_version,
        baseline_deployment_id=None,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        suites=["routing"],
        minimum_sample_size=4,
    )
    result = seed.client.post(
        "/v1/evaluations", headers=operator, json=evaluation.model_dump(mode="json")
    )
    assert result.status_code == 200, result.json()
    assert result.json()["passed"], result.json()["suite_results"]
    assert (
        seed.client.post(
            f"/v1/models/{job.model_version}/promotion-requests",
            headers=operator,
            json={
                "model_version": job.model_version,
                "target_state": "approved",
                "evaluation_id": evaluation.evaluation_id,
                "reason": "SYNTHETIC calibrated router",
            },
        ).status_code
        == 200
    )
    return RouterSeed(seed, specialist, job.model_version, dataset)


@pytest.fixture
def evaluation_seed(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[EvaluationSeed]:
    no_context = getattr(request, "param", None) == "no_context"
    app = create_app(
        Settings(data_dir=tmp_path, policy=DatasetPolicy(), outbox_dispatch_enabled=False)
    )
    start = now() - timedelta(seconds=1)
    with TestClient(app) as client:
        cutoff = now()
        for i in range(9):
            result = client.post(
                "/v1/inference",
                headers={
                    "Authorization": "Bearer synthetic-key-a",
                    "X-Subject": f"synthetic-eval-{i}",
                },
                json={
                    "request_id": f"synthetic-eval-seed-{i}",
                    "application_id": "support-assistant",
                    "messages": [
                        {"role": "user", "content": f"SYNTHETIC unique case {i} unused receipt"}
                    ],
                    "rag": {"enabled": i > 0 and not no_context, "index_id": "synthetic-kb"},
                },
            )
            assert result.status_code == 200
            if i == 0:
                cutoff = now()
        spec = DatasetSpecification(
            dataset_id="synthetic-evaluation",
            tenant_ids=["synthetic-a", "synthetic-b"],
            source_window=SourceWindow(start=start, end=now()),
            eligibility_policy_version="synthetic-dataset-policy-1",
            near_duplicate_threshold=1,
            time_split=TimeSplit(
                train_end=cutoff, validation_end=cutoff + timedelta(microseconds=1)
            ),
        )
        result = client.post(
            "/v1/datasets/builds",
            headers={"Authorization": "Bearer synthetic-operator-key"},
            json=spec.model_dump(mode="json"),
        )
        assert result.status_code == 200
        manifest = DatasetManifest.model_validate(result.json())
        assert manifest.examples == {"train": 1, "validation": 0, "test": 8}
        request = EvaluationInput(
            candidate_deployment_id="fake-foundation-local-1",
            baseline_deployment_id="fake-foundation-local-1",
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.version,
            suites=["golden", "held_out", "safety", "retrieval", "performance"],
            minimum_sample_size=4,
            critical_segments=["citation", "safety"],
            performance_requests=12,
        )
        yield EvaluationSeed(app, client, tmp_path, manifest, request)


# Reuse the existing behavioral suites for both durable backends.
pytest_plugins = ["postgres_support"]
