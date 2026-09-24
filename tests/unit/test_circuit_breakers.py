import pytest

from adaptive_llm.contracts import BreakerThresholds
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.routing.breakers import CircuitBreakers


@pytest.mark.parametrize("failure", ["error", "validation", "latency"])
def test_closed_open_single_probe_recovery_and_window_expiry(failure: str) -> None:
    at = [0.0]
    metrics = InProcessMetrics()
    breakers = CircuitBreakers(metrics, lambda: at[0])
    thresholds = BreakerThresholds(minimum_samples=2, cooldown_seconds=2, window_seconds=3)
    assert breakers.acquire("synthetic", thresholds)
    for _ in range(2):
        breakers.record(
            "synthetic",
            thresholds,
            error=failure == "error",
            validation_failure=failure == "validation",
            latency_ms=6000 if failure == "latency" else 1,
        )
    assert breakers.states() == {"synthetic": "open"}
    assert metrics.get("breaker_state", deployment_id="synthetic") == 1
    assert not breakers.acquire("synthetic", thresholds)
    at[0] += 2
    assert breakers.available("synthetic", thresholds)
    assert breakers.states()["synthetic"] == "half_open"
    assert breakers.acquire("synthetic", thresholds)
    assert not breakers.acquire("synthetic", thresholds)
    breakers.record("synthetic", thresholds, error=False, validation_failure=False, latency_ms=1)
    assert breakers.states()["synthetic"] == "closed"
    breakers.record("synthetic", thresholds, error=True, validation_failure=False, latency_ms=1)
    at[0] += 4
    breakers.record("synthetic", thresholds, error=False, validation_failure=False, latency_ms=1)
    assert breakers.states()["synthetic"] == "closed"
    for _ in range(2):
        breakers.record("synthetic", thresholds, error=True, validation_failure=False, latency_ms=1)
    at[0] += 2
    assert breakers.acquire("synthetic", thresholds)
    breakers.record("synthetic", thresholds, error=True, validation_failure=False, latency_ms=1)
    assert breakers.states()["synthetic"] == "open"


def test_deployment_metric_labels_and_breaker_storage_are_bounded() -> None:
    metrics = InProcessMetrics()
    breakers = CircuitBreakers(metrics)
    for i in range(100):
        metrics.increment("route_distribution", deployment_id=f"synthetic-{i}")
        breakers.acquire(f"synthetic-{i}", BreakerThresholds())
    assert len(breakers.states()) == 64
    assert metrics.get("route_distribution", deployment_id="other") == 36
    metrics.increment("fallback_reasons", reason="synthetic-arbitrary")
    assert metrics.get("fallback_reasons", reason="other") == 1
