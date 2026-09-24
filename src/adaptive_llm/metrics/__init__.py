"""Small, thread-safe metric boundary with a closed set of names and label dimensions."""

import re
from threading import Lock
from typing import Literal, Protocol, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

Counter = Literal[
    "requests",
    "fallback_free_attempts",
    "outbox_delivered",
    "outbox_retried",
    "dead_letter",
    "dropped_events",
    "persistence_failures",
    "degraded_emissions",
    "replay_hits",
    "dispatcher_failures",
    "route_distribution",
    "fallback_reasons",
    "shadow_opportunities",
    "shadow_completed",
    "shadow_drops",
    "shadow_cost_micros",
    "shadow_failures",
    "validation_advisory_failures",
]
Gauge = Literal[
    "outbox_pending",
    "outbox_dead",
    "dispatcher_lag_seconds",
    "shadow_queue_depth",
    "shadow_coverage",
    "breaker_state",
    "kill_switch",
]
StatusClass = Literal["2xx", "3xx", "4xx", "5xx"]
HttpMethod = Literal[
    "GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT", "OTHER"
]
HTTP_METHODS: frozenset[HttpMethod] = frozenset(
    {
        "GET",
        "HEAD",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "TRACE",
        "CONNECT",
        "OTHER",
    }
)


class Metrics(Protocol):
    def increment(
        self,
        name: Counter,
        value: int = 1,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
        deployment_id: str | None = None,
        reason: str | None = None,
        check_name: str | None = None,
    ) -> None: ...

    def gauge(self, name: Gauge, value: float, *, deployment_id: str | None = None) -> None: ...

    def get(
        self,
        name: Counter | Gauge,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
        deployment_id: str | None = None,
        reason: str | None = None,
        check_name: str | None = None,
    ) -> float: ...


class InProcessMetrics:
    def __init__(self) -> None:
        self._values: dict[tuple[str | None, ...], float] = {}
        self._lock = Lock()
        self._deployments: set[str] = set()
        self._domain_checks: set[str] = set()

    def increment(
        self,
        name: Counter,
        value: int = 1,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
        deployment_id: str | None = None,
        reason: str | None = None,
        check_name: str | None = None,
    ) -> None:
        if (
            value < 0
            or status_class not in (None, "2xx", "3xx", "4xx", "5xx")
            or (method is not None and method not in HTTP_METHODS)
        ):
            raise ValueError("invalid_metric")
        with self._lock:
            key = (
                name,
                tenant_id,
                status_class,
                method,
                self._deployment(deployment_id),
                self._reason(reason),
                self._check_name(check_name),
            )
            self._values[key] = self._values.get(key, 0) + value

    def gauge(self, name: Gauge, value: float, *, deployment_id: str | None = None) -> None:
        with self._lock:
            self._values[(name, None, None, None, self._deployment(deployment_id), None, None)] = (
                value
            )

    def get(
        self,
        name: Counter | Gauge,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
        deployment_id: str | None = None,
        reason: str | None = None,
        check_name: str | None = None,
    ) -> float:
        with self._lock:
            return self._values.get(
                (
                    name,
                    tenant_id,
                    status_class,
                    method,
                    self._deployment(deployment_id),
                    self._reason(reason),
                    self._check_name(check_name),
                ),
                0,
            )

    def _deployment(self, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in self._deployments and len(self._deployments) >= 64:
            return "other"
        self._deployments.add(value)
        return value

    def _check_name(self, value: str | None) -> str | None:
        # Builtins are fixed. Domain names come from trusted validator configuration;
        # cap their label inventory across applications, just like deployment labels.
        builtin = {
            "non_empty",
            "citation_ids",
            "json_object",
            "tool_allowlist",
            "citation_required",
            "groundedness",
            "language",
            "repetition",
            "truncation",
        }
        if value is None or value in builtin:
            return value
        if re.fullmatch(r"domain\.[\w.-]{1,100}", value):
            if value in self._domain_checks or len(self._domain_checks) < 32:
                self._domain_checks.add(value)
                return value
        return "other"

    @staticmethod
    def _reason(value: str | None) -> str | None:
        allowed = {
            "validation_failure",
            "endpoint_error",
            "deadline_risk",
            "policy_uncertainty",
            "unsupported_tool",
            "low_confidence",
            "out_of_distribution",
            "low_quality",
            "circuit_open",
            "live_specialists_disabled",
            "queue_full",
            "disabled",
            "unhealthy",
            "shutdown",
        }
        return value if value is None or value in allowed else "other"


class RequestMetrics:
    """Pure ASGI wrapper preserves the gateway's cancellation and reservation semantics."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app, self.metrics = app, metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        status: StatusClass = "5xx"
        method = cast(HttpMethod, scope["method"]) if scope["method"] in HTTP_METHODS else "OTHER"

        async def record(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                classes: dict[int, StatusClass] = {2: "2xx", 3: "3xx", 4: "4xx", 5: "5xx"}
                status = classes.get(message["status"] // 100, "5xx")
            await send(message)

        try:
            await self.app(scope, receive, record)
        finally:
            self.metrics.increment("requests", status_class=status, method=method)
