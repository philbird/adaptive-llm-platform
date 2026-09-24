import pytest

from adaptive_llm.contracts import (
    CanaryConfig,
    LiveObservation,
    RollbackThresholds,
    RoutePolicy,
    RoutingFeatures,
    uid,
)
from adaptive_llm.routing.canary import aggregate
from adaptive_llm.routing.live import assigned


def observation(
    *,
    specialist: bool,
    cost: int,
    success: bool = True,
    fallback: bool = False,
    task: str = "general",
    **updates: object,
) -> LiveObservation:
    return LiveObservation.model_validate(
        {
            "interaction_id": uid(),
            "policy_id": "synthetic",
            "tenant_id": "synthetic-a",
            "features": RoutingFeatures(task=task, input_tokens=10).model_dump(),
            "specialist_version": "specialist" if specialist else None,
            "specialist_served": specialist and not fallback and success,
            "fallback_used": fallback,
            "success": success,
            "quality": float(success),
            "total_cost_micros": cost,
            "latency_ms": 2,
            **updates,
        }
    )


def test_business_cost_counts_fallback_failed_outcomes_and_separate_shadow() -> None:
    rows = [observation(specialist=False, cost=100) for _ in range(10)]
    rows += [observation(specialist=True, cost=20, shadow_cost_micros=7) for _ in range(8)]
    rows += [observation(specialist=True, cost=120, fallback=True) for _ in range(2)]
    report = aggregate(rows, RollbackThresholds())
    assert report.specialist.total_cost_micros == 400
    assert report.specialist.cost_per_success_micros == 40
    assert report.foundation.cost_per_success_micros == 100
    assert report.cost_reduction_fraction == pytest.approx(0.6)
    assert report.cost_delta_micros.sample_size == 10
    assert report.cost_delta_micros.mean_delta == -60
    assert report.specialist_served.cost_per_success_micros == 20
    assert report.served_cost_delta_micros.sample_size == 8
    assert report.served_cost_delta_micros.mean_delta == -80
    assert report.foundation_served.total_cost_micros == 1240
    assert report.foundation_served.cost_per_success_micros == 104
    assert report.shadow_cost_micros == 56
    assert report.passed
    # Costs from failed outcomes stay in the numerator and bootstrap, not the success denominator.
    rows.append(observation(specialist=True, cost=100, success=False, error=True))
    report = aggregate(rows, RollbackThresholds(error_rate_max=1))
    assert report.specialist.cost_per_success_micros == 50
    assert report.cost_reduction_fraction == 0.5
    assert report.cost_delta_micros.mean_delta == -50
    print(
        "canary-cost: specialist_total=400 successes=10 per_success=40; "
        "foundation_per_success=100 reduction=0.600; shadow=56; failed_outcome_per_success=50"
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("validation_failure", True, "validation_failure_rate"),
        ("error", True, "error_rate"),
        ("latency_ms", 6000, "p95_latency_ms"),
        ("total_cost_micros", 1000, "cost_ratio"),
        ("critical_safety_incidents", 1, "critical_safety_incident"),
    ],
)
def test_every_rollback_threshold(field: str, value: object, reason: str) -> None:
    rows = [observation(specialist=False, cost=100) for _ in range(4)]
    rows += [observation(specialist=True, cost=20, **{field: value}) for _ in range(4)]
    report = aggregate(rows, RollbackThresholds())
    assert reason in report.breach_reasons and not report.passed


def test_missing_controls_unpaired_tasks_and_no_successes_fail_closed() -> None:
    rows = [observation(specialist=True, cost=20, success=False) for _ in range(4)]
    result = aggregate(rows, RollbackThresholds())
    assert result.specialist.cost_per_success_micros is None
    assert result.cost_reduction_fraction is None
    assert not result.passed
    rows = [observation(specialist=True, cost=20) for _ in range(4)]
    rows += [observation(specialist=False, cost=100, task="other") for _ in range(4)]
    result = aggregate(rows, RollbackThresholds())
    assert result.quality_delta.sample_size == 0 and not result.passed


def test_assignment_is_stable_and_bounded() -> None:
    policy = RoutePolicy(policy_id="synthetic", canary=CanaryConfig(traffic_fraction=0.05))
    assignments = [assigned(f"synthetic-{i}", policy) for i in range(10000)]
    assert 400 <= sum(assignments) <= 600
    assert assignments == [assigned(f"synthetic-{i}", policy) for i in range(10000)]
    assert not assigned("synthetic", policy.model_copy(update={"canary": CanaryConfig()}))
    assert assigned(
        "synthetic",
        policy.model_copy(
            update={
                "canary": CanaryConfig(traffic_fraction=1),
            }
        ),
    )
