"""Bounded attempts on one canonical request. Live enablement is injection-only."""

import asyncio
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

from adaptive_llm.contracts import (
    GenerationAttempt,
    PolicyDecision,
    RequestParameters,
    RoutePolicy,
    RoutingOptions,
)
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.metrics import Metrics
from adaptive_llm.providers import Provider, ProviderError, ProviderRequest, ProviderResult
from adaptive_llm.routing import Deployment, RouteSelection
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.control import RoutePolicyStore, RouteSnapshot
from adaptive_llm.validation import Validator


@dataclass(frozen=True)
class ExecutionCandidate:
    deployment: Deployment
    provider: Provider
    estimated_latency_ms: float = 1
    specialist: bool = False
    quality: float = 1
    confidence: float = 1
    ood_score: float = 0
    rejection_reason: str | None = None
    context_limit: int = 1_000_000
    has_estimate: bool = True


@dataclass(frozen=True)
class ChainPlan:
    candidates: tuple[ExecutionCandidate, ...]
    policy: RoutePolicy
    allow_live_specialists: bool = False
    fallback_reason: str | None = None
    considered: tuple[ExecutionCandidate, ...] = ()


class ChainPlanner(Protocol):
    def plan(
        self,
        selection: RouteSelection,
        options: RoutingOptions,
        identity: Identity,
        task: str,
    ) -> ChainPlan: ...


class FoundationPlanner:
    def __init__(self, store: RoutePolicyStore, provider: Provider, metrics: Metrics) -> None:
        self.store, self.provider, self.metrics = store, provider, metrics

    def plan(
        self,
        selection: RouteSelection,
        options: RoutingOptions,
        identity: Identity,
        task: str,
    ) -> ChainPlan:
        try:
            snapshot = self.store.snapshot()
        except Exception:
            # A shadow-control outage fails closed for specialists, preserving foundation serving.
            snapshot = RouteSnapshot(killed=True)
            self.metrics.increment("shadow_failures")
        self.metrics.gauge(
            "kill_switch",
            int(snapshot.killed or bool(snapshot.policy and snapshot.policy.kill_switch)),
        )
        return ChainPlan(
            candidates=(ExecutionCandidate(selection.deployment, self.provider),),
            policy=snapshot.policy
            or RoutePolicy(
                policy_id="foundation-only",
                foundation_fallback=selection.deployment.model_deployment_id,
            ),
            fallback_reason="live_specialists_disabled" if options.mode == "specialist" else None,
        )


def rejection(
    candidate: ExecutionCandidate,
    plan: ChainPlan,
    request: ProviderRequest,
    options: RoutingOptions,
    policy: PolicyDecision,
    spent: int,
) -> str | None:
    if candidate.rejection_reason is not None:
        return candidate.rejection_reason
    if request.input_tokens + request.max_output_tokens > candidate.context_limit:
        return "context_length"
    if candidate.deployment.processing_region != policy.residency or not policy.processing_allowed:
        return "policy_uncertainty"
    if (
        candidate.deployment.price_list.estimate(request.input_tokens, request.max_output_tokens)
        + spent
        > options.max_cost_micros
    ):
        return "policy_uncertainty"
    if candidate.specialist:
        if not plan.allow_live_specialists:
            return "live_specialists_disabled"
        if candidate.ood_score > plan.policy.ood_threshold_max:
            return "out_of_distribution"
        if candidate.quality < plan.policy.quality_threshold:
            return "low_quality"
        if candidate.confidence < plan.policy.router_confidence_threshold:
            return "low_confidence"
    return None


async def run_chain(
    plan: ChainPlan,
    request: ProviderRequest,
    options: RoutingOptions,
    policy: PolicyDecision,
    interaction_id: str,
    deadline: float,
    attempts: list[GenerationAttempt],
    validator: Validator,
    breakers: CircuitBreakers,
    metrics: Metrics,
) -> tuple[ProviderResult, GenerationAttempt]:
    visited: set[str] = set()
    reason = plan.fallback_reason
    spent = 0
    error = GatewayError(503, "no_healthy_deployment")
    if reason:
        metrics.increment("fallback_reasons", reason=reason)
    for candidate in plan.candidates:
        deployment = candidate.deployment
        name = deployment.model_deployment_id
        if name in visited:
            continue
        visited.add(name)
        if len(attempts) >= plan.policy.max_attempts:
            break
        skip = rejection(candidate, plan, request, options, policy, spent)
        if skip is None and (deadline - perf_counter()) * 1000 <= candidate.estimated_latency_ms:
            skip = "deadline_risk"
            error = GatewayError(504, "provider_deadline_exceeded")
        if skip is None and not breakers.acquire(name, plan.policy.breaker):
            skip = "circuit_open"
        if skip:
            reason = skip
            metrics.increment("fallback_reasons", reason=reason)
            # Admission rejects are removed before execution; hard fallback settings govern
            # failed attempts and whether it is useful to try another deadline estimate.
            if reason == "deadline_risk" and reason not in plan.policy.hard_fallback_on:
                break
            continue
        result, attempt = await execute_attempt(
            candidate,
            request,
            interaction_id,
            len(attempts) + 1,
            max(0, deadline - perf_counter()),
            validator,
            metrics,
            fallback_reason=reason,
        )
        attempts.append(attempt)
        breakers.record(
            name,
            plan.policy.breaker,
            error=attempt.error_code
            in {
                "provider_failed",
                "provider_deadline_exceeded",
                "request_cancelled",
                "provider_timeout",
                "provider_rate_limited",
                "provider_unavailable",
                "provider_invalid_response",
            },
            validation_failure=bool(attempt.validation and not attempt.validation.passed),
            latency_ms=attempt.total_latency_ms,
        )
        spent += attempt.estimated_cost_micros or 0
        metrics.increment("route_distribution", deployment_id=name)
        if attempt.error_code is None and result is not None:
            if len(attempts) == 1:
                metrics.increment("fallback_free_attempts")
            return result, attempt
        if attempt.error_code == "request_cancelled":
            raise asyncio.CancelledError
        error = GatewayError(
            504
            if attempt.error_code in {"provider_deadline_exceeded", "provider_timeout"}
            else 503
            if attempt.error_code in {"provider_rate_limited", "provider_unavailable"}
            else 502,
            attempt.error_code or "provider_failed",
        )
        reason = (
            "unsupported_tool"
            if attempt.validation
            and any(c.name == "tool_allowlist" and not c.passed for c in attempt.validation.checks)
            else "validation_failure"
            if attempt.error_code == "validation_failed"
            else "deadline_risk"
            if attempt.error_code in {"provider_deadline_exceeded", "provider_timeout"}
            else "endpoint_error"
        )
        metrics.increment("fallback_reasons", reason=reason)
        if reason not in plan.policy.hard_fallback_on:
            break
    raise error


async def execute_attempt(
    candidate: ExecutionCandidate,
    request: ProviderRequest,
    interaction_id: str,
    number: int,
    remaining_seconds: float,
    validator: Validator,
    metrics: Metrics,
    *,
    fallback_reason: str | None = None,
    shadow: bool = False,
) -> tuple[ProviderResult | None, GenerationAttempt]:
    deployment = candidate.deployment
    attempt = GenerationAttempt(
        interaction_id=interaction_id,
        attempt_number=number,
        shadow=shadow,
        model_provider=deployment.model_provider,
        model_id=deployment.model_id,
        model_version=deployment.model_version,
        deployment_id=deployment.model_deployment_id,
        adapter_id=deployment.model_deployment_id if candidate.specialist else None,
        request_parameters=RequestParameters(
            max_output_tokens=request.max_output_tokens, temperature=request.temperature
        ),
        total_latency_ms=0,
        price_list_version=deployment.price_list.version,
        finish_reason="error",
        fallback_reason=fallback_reason,
    )
    started = perf_counter()
    result: ProviderResult | None = None
    code: str | None = None
    try:
        async with asyncio.timeout(remaining_seconds):
            result = await candidate.provider.generate(
                replace(request, deadline_ms=min(request.deadline_ms, remaining_seconds * 1000))
            )
        attempt = attempt.model_copy(
            update={
                "usage": result.usage,
                "tool_calls": list(result.tool_calls),
                "finish_reason": result.finish_reason,
                "estimated_cost_micros": deployment.price_list.estimate(
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                ),
            }
        )
        if result.finish_reason == "deadline_exceeded":
            code = "provider_deadline_exceeded"
        elif result.finish_reason not in ("stop", "length", "content_filter"):
            code = "provider_failed"
        else:
            validation = validator.validate(request, result)
            for check in validation.checks:
                if check.severity == "advisory" and not check.passed:
                    metrics.increment("validation_advisory_failures", check_name=check.name)
            attempt = attempt.model_copy(update={"validation": validation})
            if not validation.passed:
                code = "validation_failed"
            if perf_counter() - started >= remaining_seconds:
                code = "provider_deadline_exceeded"
                attempt = attempt.model_copy(update={"finish_reason": "deadline_exceeded"})
    except ProviderError as error:
        code = error.code
        if code == "provider_timeout":
            attempt = attempt.model_copy(update={"finish_reason": "deadline_exceeded"})
    except TimeoutError:
        code = "provider_deadline_exceeded"
        attempt = attempt.model_copy(update={"finish_reason": "deadline_exceeded"})
    except asyncio.CancelledError:
        code = "request_cancelled"
        attempt = attempt.model_copy(update={"finish_reason": "cancelled"})
    except Exception:
        code = "provider_failed"
    return result, attempt.model_copy(
        update={
            "error_code": code,
            "total_latency_ms": (perf_counter() - started) * 1000,
        }
    )
