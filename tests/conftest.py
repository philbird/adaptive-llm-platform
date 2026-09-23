import json
from pathlib import Path

import pytest

from adaptive_llm.app import Settings
from adaptive_llm.contracts import Chunk, InferenceRequest, Message, uid
from adaptive_llm.gateway.identity import Identity, Keyring, LocalAuthenticator
from adaptive_llm.providers import ProviderRequest


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
