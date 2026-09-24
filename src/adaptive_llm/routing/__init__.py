"""One foundation candidate with hard residency and budget constraints."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from adaptive_llm.contracts import (
    Candidate,
    PolicyDecision,
    Region,
    RouteDecision,
    RoutingFeatures,
    RoutingOptions,
)
from adaptive_llm.providers import ProviderRequest


class PriceList(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    input_micros_per_1000_tokens: int = Field(ge=0)
    output_micros_per_1000_tokens: int = Field(ge=0)

    def estimate(self, input_tokens: int, output_tokens: int) -> int:
        return (
            input_tokens * self.input_micros_per_1000_tokens
            + output_tokens * self.output_micros_per_1000_tokens
            + 999
        ) // 1000


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_deployment_id: str
    model_provider: str
    model_id: str
    model_version: str
    processing_region: Region
    price_list: PriceList


@dataclass(frozen=True)
class RouteSelection:
    decision: RouteDecision
    deployment: Deployment
    features: RoutingFeatures | None = None
    request: ProviderRequest | None = None
    policy: PolicyDecision | None = None


class Router(Protocol):
    def route(
        self,
        request: ProviderRequest,
        options: RoutingOptions,
        policy: PolicyDecision,
        interaction_id: str,
    ) -> RouteSelection: ...


class FoundationRouter:
    def __init__(self, path: Path) -> None:
        self.deployment = Deployment.model_validate(json.loads(path.read_text())["foundation"])

    def route(
        self,
        request: ProviderRequest,
        options: RoutingOptions,
        policy: PolicyDecision,
        interaction_id: str,
    ) -> RouteSelection:
        started = perf_counter()
        deployment = self.deployment
        cost = deployment.price_list.estimate(request.input_tokens, request.max_output_tokens)
        reasons = []
        if not policy.processing_allowed:
            reasons.append("processing_denied")
        if deployment.processing_region != policy.residency:
            reasons.append("residency_mismatch")
        if cost > options.max_cost_micros:
            reasons.append("cost_limit_exceeded")
        candidate = Candidate(
            model_deployment_id=deployment.model_deployment_id,
            eligible=not reasons,
            processing_region=deployment.processing_region,
            estimated_cost_micros=cost,
            price_list_version=deployment.price_list.version,
            reason_codes=reasons or ["foundation_only"],
        )
        return RouteSelection(
            decision=RouteDecision(
                interaction_id=interaction_id,
                candidates=[candidate],
                selected_model_deployment_id=deployment.model_deployment_id
                if not reasons
                else None,
                decision_latency_ms=(perf_counter() - started) * 1000,
                policy_constraints=["processing_allowed", "residency", "max_cost_micros"],
            ),
            deployment=deployment,
        )
