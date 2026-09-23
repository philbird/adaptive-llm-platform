# Observability plan (not deployed)

Instrumentation uses the OpenTelemetry API from slice 1a; metrics are exported in a
Prometheus-compatible form once a backend is chosen. Milestone 1 dashboards: traffic/errors/latency, input/cached/context/output tokens and total
attempt cost, retrieval empty rate and index versions, validation failures, telemetry lag,
retries/dropped events and dead letters. General traces contain versioned metadata, never text.
Metric labels must use bounded dimensions; never interaction, request, subject or trace IDs.

Alert on redaction failures, tenant isolation incidents, deadline/error spikes and telemetry
loss. Dashboards and executable alert definitions arrive with real metrics, rather than
presenting unimplemented series as a working dashboard at this checkpoint.
