import pytest

from adaptive_llm.app import Settings
from adaptive_llm.contracts import PolicyDecision, RoutingOptions
from adaptive_llm.providers import ProviderRequest
from adaptive_llm.routing import FoundationRouter, PriceList


def test_foundation_price_and_selection(
    settings: Settings, provider_request: ProviderRequest
) -> None:
    result = FoundationRouter(settings.routing_path).route(
        provider_request,
        RoutingOptions(),
        PolicyDecision(policy_version="local-1", retention_seconds=1),
        "interaction",
    )
    (candidate,) = result.decision.candidates
    assert candidate.eligible
    assert candidate.processing_region == "local"
    assert candidate.price_list_version == "synthetic-prices-1"
    assert (
        candidate.estimated_cost_micros
        == provider_request.input_tokens + 2 * provider_request.max_output_tokens
    )
    assert result.decision.selected_model_deployment_id == candidate.model_deployment_id
    assert not result.decision.fallback_deployment_ids


@pytest.mark.parametrize("constraint", ["residency", "budget", "processing"])
def test_hard_constraint_rejection(
    settings: Settings, provider_request: ProviderRequest, constraint: str
) -> None:
    policy = PolicyDecision(
        policy_version="local-1",
        retention_seconds=1,
        residency="eu-west" if constraint == "residency" else "local",
        processing_allowed=constraint != "processing",
    )
    result = FoundationRouter(settings.routing_path).route(
        provider_request,
        RoutingOptions(max_cost_micros=0 if constraint == "budget" else 20_000),
        policy,
        "interaction",
    )
    assert not result.decision.candidates[0].eligible
    assert result.decision.selected_model_deployment_id is None
    assert result.decision.model_dump(mode="json")["selected_model_deployment_id"] is None
    assert result.decision.candidates[0].reason_codes == [
        {
            "residency": "residency_mismatch",
            "budget": "cost_limit_exceeded",
            "processing": "processing_denied",
        }[constraint]
    ]


def test_cost_rounds_up_using_integers() -> None:
    prices = PriceList(
        version="synthetic-1", input_micros_per_1000_tokens=1, output_micros_per_1000_tokens=2
    )
    assert prices.estimate(0, 0) == 0
    assert prices.estimate(1, 1) == 1
    assert prices.estimate(1000, 1000) == 3
