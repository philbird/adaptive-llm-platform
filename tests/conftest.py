import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

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
    now,
    uid,
)
from adaptive_llm.gateway.identity import Identity, Keyring, LocalAuthenticator
from adaptive_llm.providers import ProviderRequest


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


@dataclass
class EvaluationSeed:
    app: FastAPI
    client: TestClient
    directory: Path
    manifest: DatasetManifest
    request: EvaluationInput


@pytest.fixture
def evaluation_seed(tmp_path: Path) -> Iterator[EvaluationSeed]:
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
                    "rag": {"enabled": i > 0, "index_id": "synthetic-kb"},
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
