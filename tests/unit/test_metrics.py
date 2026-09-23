from concurrent.futures import ThreadPoolExecutor

import pytest

from adaptive_llm.metrics import InProcessMetrics


def test_metrics_are_thread_safe_and_labels_are_bounded() -> None:
    metrics = InProcessMetrics()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: metrics.increment("requests", status_class="2xx", method="POST"),
                range(1000),
            )
        )
    assert metrics.get("requests", status_class="2xx", method="POST") == 1000
    assert metrics.get("requests", status_class="4xx", method="POST") == 0
    metrics.increment("replay_hits", tenant_id="synthetic-a")
    assert metrics.get("replay_hits", tenant_id="synthetic-a") == 1
    metrics.gauge("outbox_pending", 3)
    metrics.gauge("outbox_pending", 1)
    assert metrics.get("outbox_pending") == 1
    with pytest.raises(ValueError, match="invalid_metric"):
        metrics.increment("requests", -1)
    with pytest.raises(TypeError):
        metrics.increment("requests", interaction_id="synthetic-forbidden")
    with pytest.raises(ValueError, match="invalid_metric"):
        metrics.increment("requests", status_class="synthetic-forbidden")

    with pytest.raises(TypeError):
        metrics.increment("requests", path="/v1/privacy/interactions/synthetic-forbidden")
    with pytest.raises(ValueError, match="invalid_metric"):
        metrics.increment("requests", method="synthetic-forbidden")
