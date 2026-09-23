from fastapi.testclient import TestClient

from adaptive_llm.app import create_app


def test_lifespan_health_and_no_content_endpoints() -> None:
    application = create_app()
    with TestClient(application) as client:
        assert application.state.ready
        result = client.get("/healthz")
        assert result.status_code == 200
        assert result.json() == {"status": "ok", "stage": "scaffold", "inference_enabled": False}
        assert client.post("/v1/inference", json={}).status_code == 404
    assert application.state.ready is False
