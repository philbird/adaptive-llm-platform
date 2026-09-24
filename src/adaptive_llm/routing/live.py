"""Hard admission precedes calibrated live estimates and deterministic canary assignment."""

import hashlib
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from threading import Lock

from adaptive_llm.contracts import ModelManifest, RoutePolicy, RoutingOptions
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.metrics import Metrics
from adaptive_llm.providers import Provider
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.registry.artifacts import verify_artifact
from adaptive_llm.routing import PriceList, RouteSelection
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.chain import ChainPlan, ExecutionCandidate, FoundationPlanner
from adaptive_llm.routing.control import RoutePolicyStore, RouteSnapshot
from adaptive_llm.routing.model import LogisticRouter
from adaptive_llm.routing.shadow import SpecialistLoader


def assigned(interaction_id: str, policy: RoutePolicy) -> bool:
    digest = hashlib.sha256(f"{interaction_id}:{policy.policy_id}".encode()).digest()
    return int.from_bytes(digest, "big") / (1 << 256) < policy.canary.traffic_fraction


class LivePlanner(FoundationPlanner):
    def __init__(
        self,
        store: RoutePolicyStore,
        provider: Provider,
        metrics: Metrics,
        registry: ModelRegistry,
        loader: SpecialistLoader,
        keyring: Keyring,
        data_dir: Path,
        tenants: frozenset[str],
        breakers: CircuitBreakers,
    ) -> None:
        super().__init__(store, provider, metrics)
        self.registry, self.loader, self.keyring = registry, loader, keyring
        self.data_dir, self.tenants, self.breakers = data_dir, tenants, breakers
        self._lock = Lock()
        self._snapshot: RouteSnapshot | None = None
        self._manifests: dict[str, ModelManifest] = {}
        self._routers: OrderedDict[tuple[str, str], tuple[ModelManifest, LogisticRouter]] = (
            OrderedDict()
        )

    def plan(
        self,
        selection: RouteSelection,
        options: RoutingOptions,
        identity: Identity,
        task: str,
    ) -> ChainPlan:
        foundation = super().plan(selection, options, identity, task)
        if options.mode == "foundation":
            return foundation
        with self._lock:
            try:
                return self._live(selection, options, identity, task, foundation)
            except Exception:
                self._snapshot = None
                self._manifests.clear()
                self._routers.clear()
                self.metrics.increment("live_planner_failures")
                return replace(foundation, fallback_reason="policy_uncertainty")

    def _refresh_manifests(self, snapshot: RouteSnapshot, identity: Identity) -> None:
        # The store returns the same immutable snapshot until its one-second refresh (or
        # explicit invalidation). Admission therefore shares the control pointer's cadence.
        if snapshot is self._snapshot:
            return
        policy = snapshot.policy
        assert policy is not None and policy.router_version is not None
        manifests = {
            version: self.registry.get(version, identity)
            for version in {policy.router_version, *policy.eligible_specialist_versions}
        }
        for key in list(self._routers):
            current = manifests.get(key[0])
            verified, _ = self._routers[key]
            if current is not None and (
                current.artifact_digest != key[1] or current.state != verified.state
            ):
                del self._routers[key]
        self._manifests = manifests
        self._snapshot = snapshot

    def _router(self, manifest: ModelManifest, identity: Identity) -> LogisticRouter:
        key = (manifest.version, manifest.artifact_digest)
        cached = self._routers.get(key)
        if cached is not None:
            self._routers.move_to_end(key)
            return cached[1]
        router = LogisticRouter.model_validate_json(
            verify_artifact(manifest, self.data_dir / manifest.storage_location, self.keyring)[
                "router.json"
            ]
        )
        # A slow cold load must not publish an artifact whose registry admission changed.
        latest = self.registry.get(manifest.version, identity)
        if latest != manifest:
            raise GatewayError(409, "live_router_changed")
        self._routers[key] = manifest, router
        if len(self._routers) > 8:
            self._routers.popitem(last=False)
        return router

    def _live(
        self,
        selection: RouteSelection,
        options: RoutingOptions,
        identity: Identity,
        task: str,
        foundation: ChainPlan,
    ) -> ChainPlan:
        snapshot = self.store.snapshot()
        p = snapshot.policy
        if p is None or not p.live_specialists_allowed:
            return foundation
        if not snapshot.enabled(identity.tenant_id, task):
            return replace(foundation, fallback_reason="live_specialists_disabled")
        request, features, decision = selection.request, selection.features, selection.policy
        if request is None or features is None or decision is None or p.router_version is None:
            return replace(foundation, fallback_reason="policy_uncertainty")
        internal = replace(identity, key_class="operator", dataset_tenants=self.tenants)
        self._refresh_manifests(snapshot, internal)
        router_manifest = self._manifests[p.router_version]
        if (
            router_manifest.state not in {"approved", "shadow", "canary", "production"}
            or router_manifest.adapter_architecture != "router-logistic-v1"
            or identity.tenant_id not in router_manifest.tenant_ids
        ):
            return replace(foundation, fallback_reason="policy_uncertainty")
        router = self._router(router_manifest, internal)
        candidates: list[ExecutionCandidate] = []
        for version in p.eligible_specialist_versions:
            model = self._manifests[version]
            reason: str | None = None
            if version in snapshot.disabled_specialists:
                reason = "deployment_disabled"
            elif model.state not in {"canary", "production"}:
                reason = "state_ineligible"
            elif not decision.processing_allowed or model.processing_region != decision.residency:
                reason = "policy_uncertainty"
            elif identity.tenant_id not in model.tenant_ids:
                reason = "policy_uncertainty"
            elif "text" not in model.modalities:
                reason = "unsupported_modality"
            elif request.input_tokens + request.max_output_tokens > model.context_limit:
                reason = "context_length"
            elif not self.breakers.available(version, p.breaker):
                reason = "circuit_open"
            elif model.state == "canary" and p.canary.traffic_fraction > 0.05:
                reason = "canary_not_assigned"
            elif (
                p.canary.tenant_allowlist
                and self.keyring.pseudonym(identity.tenant_id, "canary-tenant")
                not in p.canary.tenant_allowlist
            ):
                reason = "canary_not_assigned"
            elif not assigned(selection.decision.interaction_id, p):
                reason = "canary_not_assigned"
            deployment = selection.deployment.model_copy(
                update={
                    "model_deployment_id": version,
                    "model_provider": "local-specialist",
                    "model_id": model.base_model_id,
                    "model_version": model.version,
                    "processing_region": model.processing_region,
                    "price_list": PriceList(
                        version=f"model-{version}",
                        input_micros_per_1000_tokens=model.input_micros_per_1000_tokens,
                        output_micros_per_1000_tokens=model.output_micros_per_1000_tokens,
                    ),
                }
            )
            candidate = ExecutionCandidate(
                deployment,
                self.provider,
                specialist=True,
                rejection_reason=reason,
                context_limit=model.context_limit,
                has_estimate=False,
            )
            if reason is None:
                estimate = router.estimate(features, version)
                candidate = replace(
                    candidate,
                    quality=estimate.suitability,
                    confidence=estimate.confidence,
                    ood_score=estimate.ood,
                    estimated_latency_ms=estimate.latency_ms,
                    has_estimate=True,
                )
                if estimate.ood > p.ood_threshold_max:
                    reason = "out_of_distribution"
                elif estimate.suitability < p.quality_threshold:
                    reason = "low_quality"
                elif estimate.confidence < p.router_confidence_threshold:
                    reason = "low_confidence"
                elif deployment.price_list.estimate(
                    request.input_tokens, request.max_output_tokens
                ) > selection.deployment.price_list.estimate(
                    request.input_tokens, request.max_output_tokens
                ):
                    reason = "not_cheapest"
                else:
                    loaded = self.loader.load(version, identity)
                    candidate = replace(candidate, provider=loaded.provider)
                candidate = replace(candidate, rejection_reason=reason)
            candidates.append(candidate)
        candidates.sort(
            key=lambda c: (
                c.deployment.price_list.estimate(request.input_tokens, request.max_output_tokens),
                c.deployment.model_deployment_id,
            )
        )
        latest = self.store.snapshot()
        if latest.policy != p or not latest.enabled(identity.tenant_id, task):
            return replace(foundation, fallback_reason="live_specialists_disabled")
        candidates = [
            replace(c, rejection_reason="deployment_disabled")
            if c.deployment.model_deployment_id in latest.disabled_specialists
            else c
            for c in candidates
        ]
        selected = False
        for index, candidate in enumerate(candidates):
            if candidate.rejection_reason is None:
                if selected:
                    candidates[index] = replace(candidate, rejection_reason="not_selected")
                selected = True
        chosen_candidates = tuple(c for c in candidates if c.rejection_reason is None)
        reason = next(
            (c.rejection_reason for c in candidates if c.rejection_reason == "out_of_distribution"),
            None,
        )
        if not chosen_candidates and reason is None:
            reason = next((c.rejection_reason for c in candidates), "low_confidence")
        return ChainPlan(
            chosen_candidates + foundation.candidates,
            p,
            True,
            None if chosen_candidates else reason,
            tuple(candidates) + foundation.candidates,
        )
