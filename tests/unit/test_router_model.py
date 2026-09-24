import math

import pytest

from adaptive_llm.contracts import RoutingFeatures, RoutingObservation, RoutingRow
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.routing.model import ece, routing_suite, sigmoid, train_model


def row(index: int, *, covered: bool = True, task: str = "general") -> RoutingRow:
    observation = RoutingObservation(quality=1, validation_pass=True, cost_micros=10, latency_ms=1)
    return RoutingRow(
        interaction_id=f"synthetic-{index}",
        tenant_id="synthetic-a",
        source_dataset_id="synthetic",
        source_dataset_version="synthetic-1",
        foundation_id="foundation",
        features=RoutingFeatures(task=task, input_tokens=10),
        candidates={"foundation": observation, "specialist": observation if covered else None},
    )


def test_deterministic_training_calibration_and_centroid_distance() -> None:
    train = [row(i) for i in range(20)]
    validation = [row(i) for i in range(20, 30)]
    a = train_model(train, validation, lambda: None)
    b = train_model(train, validation, lambda: None)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.ood(train[0].features) == 0
    novel = row(50, task="novel").features
    # Six categorical groups with known and unknown columns; two changed columns, SD floor .25.
    distance = math.sqrt(32 / 12)
    assert a.ood(novel) == pytest.approx(distance / (1 + distance))
    estimate = a.estimate(train[0].features, "specialist")
    assert estimate.suitability > 0.99 and estimate.confidence > 0.99
    assert a.estimate(train[0].features, "unobserved").confidence == 0
    assert sigmoid(0) == 0.5
    assert ece([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1]) == pytest.approx(0.1)
    assert ece([1.0], [1.0]) == 0
    with pytest.raises(ValueError, match="calibration_samples_required"):
        ece([], [])


def test_missing_counterfactuals_train_abstention_and_disjoint_folds() -> None:
    a = train_model(
        [row(i, covered=False) for i in range(20)],
        [row(i, covered=False) for i in range(20, 30)],
        lambda: None,
    )
    assert a.estimate(row(100).features, "specialist").suitability < 0.01
    assert a.estimate(row(100).features, "specialist").confidence == 0
    with pytest.raises(GatewayError, match="routing_fold_overlap"):
        train_model([row(1)], [row(1)], lambda: None)
    with pytest.raises(GatewayError, match="routing_fold_overlap"):
        routing_suite(a, [row(1)])


def test_routing_suite_measures_known_false_selection_and_ood() -> None:
    model = train_model([row(i) for i in range(20)], [row(i) for i in range(20, 30)], lambda: None)
    result = routing_suite(model, [row(50), row(51, covered=False), row(52, task="novel")])
    assert result.metrics["false_specialist_rate"] == 0.5
    assert result.metrics["unnecessary_foundation_rate"] == 0.5
    assert result.metrics["ood_detection_rate"] == 1
    assert result.metrics["ood_samples"] == 1
    assert result.metrics["calibration_error"] > 0.1
