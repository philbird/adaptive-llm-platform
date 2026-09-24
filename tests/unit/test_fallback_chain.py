import asyncio
from dataclasses import replace
from time import perf_counter

import pytest

from adaptive_llm.app import Settings
from adaptive_llm.contracts import GenerationAttempt, PolicyDecision, RoutePolicy, RoutingOptions
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.routing import FoundationRouter
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.chain import ChainPlan, ExecutionCandidate, run_chain
from adaptive_llm.validation import LocalValidator


class RecordingProvider:
    def __init__(self, calls: list[str], name: str, failure: str | None = None) -> None:
        self.calls, self.name, self.failure = calls, name, failure

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.calls.append(self.name)
        if self.failure == "error":
            raise RuntimeError("SYNTHETIC_PRIVATE_PROVIDER_BODY")
        if self.failure == "slow":
            await asyncio.sleep(0.2)
        result = await FakeProvider().generate(request)
        return replace(result, content="") if self.failure == "validation" else result


@pytest.mark.parametrize("failure", ["error", "validation"])
async def test_order_numbering_input_reuse_and_failed_output(
    provider_request: ProviderRequest,
    settings: Settings,
    failure: str,
) -> None:
    foundation = FoundationRouter(settings.routing_path).deployment
    calls: list[str] = []
    first = ExecutionCandidate(
        foundation.model_copy(update={"model_deployment_id": "synthetic-specialist"}),
        RecordingProvider(calls, "specialist", failure),
        specialist=True,
    )
    final = ExecutionCandidate(foundation, RecordingProvider(calls, "foundation"))
    plan = ChainPlan((first, first, final), RoutePolicy(), allow_live_specialists=True)
    metrics = InProcessMetrics()
    attempts: list[GenerationAttempt] = []
    result, attempt = await run_chain(
        plan,
        provider_request,
        RoutingOptions(),
        PolicyDecision(policy_version="test", retention_seconds=1),
        "synthetic-interaction",
        perf_counter() + 1,
        attempts,
        LocalValidator(),
        CircuitBreakers(metrics),
        metrics,
    )
    assert result.content and attempt.validation.passed
    assert calls == ["specialist", "foundation"]
    assert [a.attempt_number for a in attempts] == [1, 2]
    assert attempt.fallback_reason == (
        "endpoint_error" if failure == "error" else "validation_failure"
    )
    assert "SYNTHETIC_PRIVATE_PROVIDER_BODY" not in str(attempts)


@pytest.mark.parametrize("case", ["max_attempts", "no_hard_fallback", "deadline", "slow"])
async def test_chain_bounds(
    provider_request: ProviderRequest, settings: Settings, case: str
) -> None:
    foundation = FoundationRouter(settings.routing_path).deployment
    calls: list[str] = []
    first = ExecutionCandidate(
        foundation.model_copy(update={"model_deployment_id": "synthetic-specialist"}),
        RecordingProvider(calls, "specialist", "slow" if case == "slow" else "error"),
        estimated_latency_ms=1,
        specialist=True,
    )
    final = ExecutionCandidate(
        foundation,
        RecordingProvider(calls, "foundation"),
        estimated_latency_ms=1000 if case == "deadline" else 1,
    )
    policy = RoutePolicy(
        max_attempts=1 if case == "max_attempts" else 2,
        hard_fallback_on=[] if case == "no_hard_fallback" else ["endpoint_error", "deadline_risk"],
    )
    attempts: list[GenerationAttempt] = []
    metrics = InProcessMetrics()
    with pytest.raises(GatewayError):
        await run_chain(
            ChainPlan((first, final), policy, True),
            provider_request,
            RoutingOptions(),
            PolicyDecision(policy_version="test", retention_seconds=1),
            "synthetic",
            perf_counter() + 0.05,
            attempts,
            LocalValidator(),
            CircuitBreakers(metrics),
            metrics,
        )
    assert calls == ["specialist"]
    assert len(attempts) == 1
    if case == "slow":
        assert attempts[0].finish_reason == "deadline_exceeded"


@pytest.mark.parametrize("signal", ["quality", "confidence", "ood_score", "disabled", "breaker"])
async def test_candidate_admission(
    provider_request: ProviderRequest,
    settings: Settings,
    signal: str,
) -> None:
    foundation = FoundationRouter(settings.routing_path).deployment
    calls: list[str] = []
    first = ExecutionCandidate(
        foundation.model_copy(update={"model_deployment_id": "synthetic-specialist"}),
        RecordingProvider(calls, "specialist"),
        specialist=True,
        quality=0 if signal == "quality" else 1,
        confidence=0 if signal == "confidence" else 1,
        ood_score=1 if signal == "ood_score" else 0,
    )
    policy = RoutePolicy(
        hard_fallback_on=["low_quality", "low_confidence", "out_of_distribution", "circuit_open"]
    )
    metrics = InProcessMetrics()
    breakers = CircuitBreakers(metrics)
    if signal == "breaker":
        for _ in range(5):
            breakers.record(
                "synthetic-specialist",
                policy.breaker,
                error=True,
                validation_failure=False,
                latency_ms=1,
            )
    await run_chain(
        ChainPlan(
            (first, ExecutionCandidate(foundation, RecordingProvider(calls, "foundation"))),
            policy,
            signal != "disabled",
        ),
        provider_request,
        RoutingOptions(),
        PolicyDecision(policy_version="test", retention_seconds=1),
        "synthetic",
        perf_counter() + 1,
        [],
        LocalValidator(),
        breakers,
        metrics,
    )
    assert calls == ["foundation"]


async def test_advisories_do_not_fallback_or_open_breaker(
    provider_request: ProviderRequest,
    settings: Settings,
) -> None:
    foundation = FoundationRouter(settings.routing_path).deployment
    calls: list[str] = []

    class ParaphrasingProvider:
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            calls.append("specialist")
            return replace(
                await FakeProvider().generate(request),
                content="SYNTHETIC paraphrased answer.",
            )

    specialist = ExecutionCandidate(
        foundation.model_copy(update={"model_deployment_id": "synthetic-specialist"}),
        ParaphrasingProvider(),
        specialist=True,
    )
    plan = ChainPlan(
        (specialist, ExecutionCandidate(foundation, RecordingProvider(calls, "foundation"))),
        RoutePolicy(),
        allow_live_specialists=True,
    )
    metrics = InProcessMetrics()
    breakers = CircuitBreakers(metrics)
    for _ in range(6):
        attempts: list[GenerationAttempt] = []
        _, attempt = await run_chain(
            plan,
            provider_request,
            RoutingOptions(),
            PolicyDecision(policy_version="test", retention_seconds=1),
            "synthetic-interaction",
            perf_counter() + 1,
            attempts,
            LocalValidator(),
            breakers,
            metrics,
        )
        assert len(attempts) == 1 and attempt.validation.passed
        assert attempt.error_code is None
    assert calls == ["specialist"] * 6
    assert breakers.states()["synthetic-specialist"] == "closed"
    assert metrics.get("validation_advisory_failures", check_name="groundedness") == 6
    assert metrics.get("fallback_reasons", reason="validation_failure") == 0
