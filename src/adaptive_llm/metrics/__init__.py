"""Small, thread-safe metric boundary with a closed set of names and label dimensions."""

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
]
Gauge = Literal["outbox_pending", "outbox_dead", "dispatcher_lag_seconds"]
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
    ) -> None: ...

    def gauge(self, name: Gauge, value: float) -> None: ...

    def get(
        self,
        name: Counter | Gauge,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
    ) -> float: ...


class InProcessMetrics:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str | None, str | None, str | None], float] = {}
        self._lock = Lock()

    def increment(
        self,
        name: Counter,
        value: int = 1,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
    ) -> None:
        if (
            value < 0
            or status_class not in (None, "2xx", "3xx", "4xx", "5xx")
            or (method is not None and method not in HTTP_METHODS)
        ):
            raise ValueError("invalid_metric")
        key = (name, tenant_id, status_class, method)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def gauge(self, name: Gauge, value: float) -> None:
        with self._lock:
            self._values[(name, None, None, None)] = value

    def get(
        self,
        name: Counter | Gauge,
        *,
        tenant_id: str | None = None,
        status_class: StatusClass | None = None,
        method: HttpMethod | None = None,
    ) -> float:
        with self._lock:
            return self._values.get((name, tenant_id, status_class, method), 0)


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
