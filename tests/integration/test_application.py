from pathlib import Path

from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import InferenceRequest
from adaptive_llm.gateway.identity import Identity


def test_lifespan_health_and_mounted_inference() -> None:
    application = create_app()
    with TestClient(application) as client:
        assert application.state.ready
        result = client.get("/healthz")
        assert result.status_code == 200
        assert result.json() == {"status": "ok", "stage": "slice-1a", "inference_enabled": True}
        assert client.post("/v1/inference", json={}).status_code == 401
    assert application.state.ready is False


def test_disabled_inference_is_truthful() -> None:
    with TestClient(create_app(Settings(inference_enabled=False))) as client:
        assert client.get("/healthz").json()["inference_enabled"] is False
        assert client.post("/v1/inference", json={}).status_code == 404


def test_custom_authenticator_and_secret_need_no_local_identity_file(
    tmp_path: Path, identity: Identity, inference_request: InferenceRequest
) -> None:
    class CustomAuthenticator:
        def authenticate(self, authorization: str | None, subject: str | None) -> Identity:
            return identity

    settings = Settings(
        identity_path=tmp_path / "missing-identity.json",
        secret=b"synthetic-injected-secret",
        authenticator=CustomAuthenticator(),
    )
    assert "synthetic-injected-secret" not in repr(settings)
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/inference", json=inference_request.model_dump())
    assert result.status_code == 200
