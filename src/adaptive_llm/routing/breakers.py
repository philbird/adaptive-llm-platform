"""Bounded, process-local sliding windows with one half-open probe per deployment."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Literal

from adaptive_llm.contracts import BreakerThresholds
from adaptive_llm.evaluation.stats import percentile
from adaptive_llm.metrics import Metrics

BreakerState = Literal["closed", "open", "half_open"]


@dataclass
class Window:
    samples: deque[tuple[float, bool, bool, float]] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    state: BreakerState = "closed"
    opened_at: float = 0
    probing: bool = False


class CircuitBreakers:
    def __init__(self, metrics: Metrics, clock: Callable[[], float] = monotonic) -> None:
        self.metrics, self.clock = metrics, clock
        self._windows: dict[str, Window] = {}
        self._lock = Lock()

    def _window(self, deployment: str) -> Window | None:
        if deployment not in self._windows:
            # Refuse additional deployments rather than evicting an open breaker.
            if len(self._windows) >= 64:
                return None
            self._windows[deployment] = Window()
        return self._windows[deployment]

    def available(self, deployment: str, thresholds: BreakerThresholds) -> bool:
        with self._lock:
            window = self._window(deployment)
            if window is None:
                return False
            self._advance(window, thresholds)
            self._publish(deployment, window)
            return window.state != "open" and not window.probing

    def acquire(self, deployment: str, thresholds: BreakerThresholds) -> bool:
        with self._lock:
            window = self._window(deployment)
            if window is None:
                return False
            self._advance(window, thresholds)
            if window.state == "open" or window.probing:
                return False
            if window.state == "half_open":
                window.probing = True
            self._publish(deployment, window)
            return True

    def _advance(self, window: Window, thresholds: BreakerThresholds) -> None:
        if (
            window.state == "open"
            and self.clock() - window.opened_at >= thresholds.cooldown_seconds
        ):
            window.state = "half_open"

    def record(
        self,
        deployment: str,
        thresholds: BreakerThresholds,
        *,
        error: bool,
        validation_failure: bool,
        latency_ms: float,
    ) -> None:
        with self._lock:
            window = self._window(deployment)
            if window is None:
                return
            at = self.clock()
            bad = error or validation_failure or latency_ms > thresholds.p95_latency_ms
            if window.state == "half_open":
                window.probing = False
                window.state = "open" if bad else "closed"
                window.opened_at = at
                window.samples.clear()
            elif window.state == "closed":
                window.samples.append((at, error, validation_failure, latency_ms))
                while window.samples and window.samples[0][0] < at - thresholds.window_seconds:
                    window.samples.popleft()
                count = len(window.samples)
                if count >= thresholds.minimum_samples and (
                    sum(s[1] for s in window.samples) / count >= thresholds.error_rate
                    or sum(s[2] for s in window.samples) / count
                    >= thresholds.validation_failure_rate
                    or percentile([s[3] for s in window.samples], 0.95) > thresholds.p95_latency_ms
                ):
                    window.state, window.opened_at = "open", at
            self._publish(deployment, window)

    def _publish(self, deployment: str, window: Window) -> None:
        self.metrics.gauge(
            "breaker_state",
            {"closed": 0, "open": 1, "half_open": 2}[window.state],
            deployment_id=deployment,
        )

    def states(self) -> dict[str, BreakerState]:
        with self._lock:
            return {key: window.state for key, window in self._windows.items()}
