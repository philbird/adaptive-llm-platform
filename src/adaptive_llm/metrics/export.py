"""Isolated OTLP provider and Prometheus collection; no global SDK registration."""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def otlp_provider(endpoint: str) -> TracerProvider:
    provider = TracerProvider(resource=Resource({"service.name": "adaptive-llm"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=2)))
    return provider
