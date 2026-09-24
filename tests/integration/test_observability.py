import errno
from typing import get_args
from urllib.request import urlopen

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from prometheus_client import generate_latest, start_http_server

from adaptive_llm.app import Settings, create_app
from adaptive_llm.metrics import Counter, Gauge, InProcessMetrics
from adaptive_llm.metrics.export import otlp_provider


def assert_metrics(text):
    for name in (*get_args(Counter), *get_args(Gauge)):
        assert f"# HELP {name}" in text
    assert "SYNTHETIC PRIVATE REASON" not in text
    for label in ("interaction_id", "trace_id", "request_id", "subject_id", "path"):
        assert label + "=" not in text


def bounded_metrics():
    metrics = InProcessMetrics()
    for i in range(200):
        metrics.increment(
            "requests",
            tenant_id=f"synthetic-{i}",
            deployment_id=f"synthetic-{i}",
            reason="SYNTHETIC PRIVATE REASON",
        )
    return metrics


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_metrics_separate_listener_lifecycle_and_api_has_no_route(bind, metrics_listener):
    metrics = bounded_metrics()
    app = create_app(Settings(inference_enabled=False, metrics=metrics, metrics_bind=bind))
    assert all(getattr(route, "path", None) != "/metrics" for route in app.routes)
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        assert client.get("/metrics").status_code == 404
        metrics_listener.assert_called_once_with(
            9464, addr=bind, registry=app.state.metrics_registry
        )
        assert_metrics(generate_latest(app.state.metrics_registry).decode())
        assert len(metrics._tenants) == len(metrics._deployments) == 64
        server, thread = metrics_listener.return_value
        server.shutdown.assert_not_called()
    server.shutdown.assert_called_once_with()
    server.server_close.assert_called_once_with()
    thread.join.assert_called_once_with()


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_metrics_http_server_answers_on_bound_port(bind, monkeypatch):
    monkeypatch.setattr("adaptive_llm.app.start_http_server", start_http_server)
    app = create_app(
        Settings(
            inference_enabled=False, metrics=bounded_metrics(), metrics_bind=bind, metrics_port=0
        )
    )
    try:
        with TestClient(app) as client:
            server = app.state.metrics_server
            assert server.server_address[0] == bind
            assert client.get("/metrics").status_code == 404
            with urlopen(f"http://127.0.0.1:{server.server_port}/metrics", timeout=3) as response:
                assert response.status == 200
                assert "text/plain" in response.headers["content-type"]
                assert_metrics(response.read().decode())
        assert server.socket.fileno() == -1
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM, errno.ENETUNREACH, errno.EADDRNOTAVAIL}:
            raise
        pytest.skip("Metrics HTTP test requires accessible local networking")


def test_metrics_server_stops_when_startup_fails(metrics_listener, monkeypatch):
    def fail(*args):
        raise ValueError("synthetic_startup_failure")

    monkeypatch.setattr("adaptive_llm.app._start_inference", fail)
    with pytest.raises(ValueError, match="synthetic_startup_failure"), TestClient(create_app()):
        pass
    server, thread = metrics_listener.return_value
    server.shutdown.assert_called_once_with()
    server.server_close.assert_called_once_with()
    thread.join.assert_called_once_with()


@pytest.mark.parametrize("port", [-1, 65536])
def test_metrics_port_validation(port):
    with pytest.raises(ValueError, match="invalid_metrics_port"):
        Settings(metrics_port=port)


def test_otlp_endpoint_export_and_shutdown_without_global_provider(monkeypatch):
    spans, endpoints = [], []

    class Exporter(SpanExporter):
        def __init__(self, endpoint, timeout):
            endpoints.append(endpoint)

        def export(self, batch):
            spans.extend(batch)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    monkeypatch.setattr("adaptive_llm.metrics.export.OTLPSpanExporter", Exporter)
    provider = otlp_provider("http://127.0.0.1:4318/v1/traces")
    with provider.get_tracer("synthetic").start_as_current_span("policy", record_exception=False):
        pass
    assert provider.force_flush()
    provider.shutdown()
    assert endpoints == ["http://127.0.0.1:4318/v1/traces"]
    assert [s.name for s in spans] == ["policy"]
    assert Settings().otlp_endpoint is None


def test_app_exporter_lifecycle_and_content_free_spans(monkeypatch, tmp_path):
    spans = []

    class Exporter(SpanExporter):
        def export(self, batch):
            spans.extend(batch)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(Exporter()))
    monkeypatch.setattr("adaptive_llm.app.otlp_provider", lambda _: provider)
    app = create_app(Settings(data_dir=tmp_path, otlp_endpoint="http://synthetic.invalid"))
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/inference",
                headers={"Authorization": "Bearer synthetic-key-a"},
                json={
                    "request_id": "synthetic-trace",
                    "application_id": "support-assistant",
                    "messages": [{"role": "user", "content": "SYNTHETIC PRIVATE INPUT"}],
                },
            ).status_code
            == 200
        )
    assert spans
    assert "SYNTHETIC PRIVATE INPUT" not in repr([(s.attributes, s.events) for s in spans])
