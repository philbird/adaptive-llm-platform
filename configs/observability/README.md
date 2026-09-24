# Local observability export

`GET /metrics` is unauthenticated Prometheus text on a separate HTTP server, containing every
counter/gauge in the typed Metrics boundary, including zero-valued families. The application
lifespan starts this server with its own CollectorRegistry and shuts it down on exit.
`Settings.metrics_bind` defaults to `127.0.0.1`; `Settings.metrics_port` defaults to `9464`
(`0` selects an available port for tests). There is no `/metrics` route on the API listener,
including behind a local reverse proxy. Set `metrics_bind="0.0.0.0"` explicitly when deploying
behind an operator-managed scraping boundary. Give each application process a distinct
metrics port when sharing a host. Bounded labels are tenant, status class, method, deployment,
reason and check name; request, interaction, subject, trace ids and URL paths are never labels.

Set `Settings.otlp_endpoint` or `OTLP_ENDPOINT` to the full OTLP/HTTP traces endpoint, for
example `http://127.0.0.1:4318/v1/traces`. Unset means no exporter and preserves the previous
in-process instrumentation. Each application owns its SDK provider, bounded batch processor
and exporter; shutdown flushes/releases it. Explicitly injected tracers remain supported.
No global provider is replaced. Exported spans retain content-free identifiers and version
metadata and disable exception recording on serving/provider boundaries.

Dashboard deployment, alert routing, aggregation across application processes, workload/SLO
thresholds and an external trace backend remain platform/product decisions. No unimplemented
series or configured alerts are presented as working dashboards.
