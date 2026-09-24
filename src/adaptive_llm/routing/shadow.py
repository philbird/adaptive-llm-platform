"""Best-effort bounded shadow bulkhead; content exists only in memory or encrypted refs."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Protocol

from adaptive_llm.contracts import (
    GenerationAttempt,
    Interaction,
    LifecycleState,
    RoutePolicy,
    ShadowComparison,
    Validation,
    ValidationCheck,
)
from adaptive_llm.evaluation.judge import DeterministicJudge, JudgeInput, blinded_scores
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.metrics import Metrics
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.providers import CITATION_PATTERN, ProviderRequest, ProviderResult
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.routing import Deployment, PriceList
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.chain import ExecutionCandidate, execute_attempt
from adaptive_llm.routing.control import RoutePolicyStore
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.validation import Validator


@dataclass(frozen=True)
class ShadowWork:
    request: ProviderRequest = field(repr=False)
    foundation: ProviderResult = field(repr=False)
    attempt: GenerationAttempt
    interaction: Interaction
    identity: Identity
    input_redaction_failed: bool = False
    policy_id: str | None = None


class SpecialistLoader(Protocol):
    def load(self, version: str, identity: Identity) -> ExecutionCandidate: ...


class RegistrySpecialists:
    def __init__(
        self,
        registry: ModelRegistry,
        foundation: Deployment,
        keyring: Keyring,
        data_dir: Path,
        tenants: frozenset[str],
        allowed_states: frozenset[LifecycleState] = frozenset({"shadow"}),
    ) -> None:
        self.registry, self.foundation, self.keyring = registry, foundation, keyring
        self.data_dir, self.tenants = data_dir, tenants
        self.allowed_states = allowed_states
        self._cache: OrderedDict[tuple[str, str], ExecutionCandidate] = OrderedDict()
        self._lock = Lock()

    def load(self, version: str, identity: Identity) -> ExecutionCandidate:
        with self._lock:
            return self._load(version, identity)

    def _evict(self, version: str) -> None:
        for key in list(self._cache):
            if key[0] == version:
                del self._cache[key]

    def _load(self, version: str, identity: Identity) -> ExecutionCandidate:
        internal = replace(identity, key_class="operator", dataset_tenants=self.tenants)
        try:
            model = self.registry.get(version, internal)
        except Exception:
            self._evict(version)
            raise
        key = (version, model.artifact_digest)
        for existing in list(self._cache):
            if existing[0] == version and (
                existing != key or model.state not in self.allowed_states
            ):
                del self._cache[existing]
        if model.state not in self.allowed_states or identity.tenant_id not in model.tenant_ids:
            raise GatewayError(409, "shadow_model_ineligible")
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        provider = SpecialistProvider(
            model,
            self.data_dir / model.storage_location,
            self.keyring,
            self.data_dir,
        )
        try:
            latest = self.registry.get(version, internal)
        except Exception:
            self._evict(version)
            raise
        if (
            latest.state not in self.allowed_states
            or latest.artifact_digest != model.artifact_digest
        ):
            self._evict(version)
            raise GatewayError(409, "shadow_model_ineligible")
        if identity.tenant_id not in latest.tenant_ids:
            raise GatewayError(409, "shadow_model_ineligible")
        deployment = self.foundation.model_copy(
            update={
                "model_deployment_id": model.version,
                "model_id": model.base_model_id,
                "model_version": provider.model_version,
                "model_provider": "local-specialist",
                "processing_region": model.processing_region,
                "price_list": PriceList(
                    version=f"model-{model.version}",
                    input_micros_per_1000_tokens=model.input_micros_per_1000_tokens,
                    output_micros_per_1000_tokens=model.output_micros_per_1000_tokens,
                ),
            }
        )
        candidate = ExecutionCandidate(
            deployment, provider, specialist=True, context_limit=model.context_limit
        )
        self._cache[key] = candidate
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return candidate


class ShadowWorker:
    def __init__(
        self,
        store: RoutePolicyStore,
        loader: SpecialistLoader,
        policy: PolicyEngine,
        validator: Validator,
        persistence: Persistence,
        breakers: CircuitBreakers,
        metrics: Metrics,
        *,
        capacity: int = 32,
        timeout_seconds: float = 30,
    ) -> None:
        if capacity < 1 or timeout_seconds <= 0:
            raise ValueError("invalid_shadow_limits")
        self.store, self.loader, self.policy = store, loader, policy
        self.validator, self.persistence, self.breakers, self.metrics = (
            validator,
            persistence,
            breakers,
            metrics,
        )
        self.queue: asyncio.Queue[ShadowWork] = asyncio.Queue(maxsize=capacity)
        self.timeout_seconds = timeout_seconds
        self.stopping = False

    def segments(self, work: ShadowWork) -> list[str]:
        task = work.interaction.task
        tenant = self.persistence.keyring.pseudonym(work.identity.tenant_id, "shadow-segment")
        return [
            f"tenant.{tenant}",
            f"task.{task.label}",
            f"language.{task.language}",
            f"risk.{task.risk_tier}",
            "citation" if work.request.context else "no_context",
        ]

    async def submit(self, work: ShadowWork) -> None:
        """Called only after ASGI response body delivery; enqueue never waits for capacity."""
        try:
            snapshot = await asyncio.to_thread(self.store.snapshot)
            p = snapshot.policy
            if (
                p is None
                or p.policy_id != work.policy_id
                or not p.shadow_enabled
                or not p.eligible_specialist_versions
            ):
                return
            # Coverage includes disabled, unhealthy and queue-full opportunities.
            if not p.tenant_enabled.get(work.identity.tenant_id, False):
                return
            if not p.task_enabled.get(work.interaction.task.label, False):
                return
            self.metrics.increment("shadow_opportunities")
            await asyncio.to_thread(
                self.store.observe,
                p.policy_id,
                work.interaction.interaction_id,
                work.identity.tenant_id,
                self.segments(work),
            )
            if self.stopping or not snapshot.enabled(
                work.identity.tenant_id, work.interaction.task.label
            ):
                self.metrics.increment("shadow_drops", reason="disabled")
                return
            if not any(
                self.breakers.available(v, p.breaker) for v in p.eligible_specialist_versions
            ):
                self.metrics.increment("shadow_drops", reason="circuit_open")
                return
            try:
                self.queue.put_nowait(replace(work, policy_id=p.policy_id))
            except asyncio.QueueFull:
                self.metrics.increment("shadow_drops", reason="queue_full")
            self.metrics.gauge("shadow_queue_depth", self.queue.qsize())
        except Exception:
            self.metrics.increment("shadow_failures")
        finally:
            opportunities = self.metrics.get("shadow_opportunities")
            self.metrics.gauge(
                "shadow_coverage",
                self.metrics.get("shadow_completed") / opportunities if opportunities else 0,
            )

    async def run(self) -> None:
        while not self.stopping:
            try:
                work = await asyncio.wait_for(self.queue.get(), timeout=0.05)
            except TimeoutError:
                continue
            self.metrics.gauge("shadow_queue_depth", self.queue.qsize())
            try:
                # Loading, inference, validation, judging and writes share a separate thread.
                await asyncio.to_thread(self._process, work)
            except Exception:
                self.metrics.increment("shadow_failures")
            finally:
                self.queue.task_done()
                opportunities = self.metrics.get("shadow_opportunities")
                self.metrics.gauge(
                    "shadow_coverage",
                    self.metrics.get("shadow_completed") / opportunities if opportunities else 0,
                )
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
            self.metrics.increment("shadow_drops", reason="shutdown")
        self.metrics.gauge("shadow_queue_depth", 0)

    def stop(self) -> None:
        self.stopping = True

    def _process(self, work: ShadowWork) -> None:
        snapshot = self.store.snapshot()
        p = snapshot.policy
        if (
            p is None
            or p.policy_id != work.policy_id
            or not p.shadow_enabled
            or not snapshot.enabled(work.identity.tenant_id, work.interaction.task.label)
        ):
            self.metrics.increment("shadow_drops", reason="disabled")
            return
        current = self.policy.decide(work.identity, work.interaction.application_id)
        if not current.processing_allowed or current.residency != work.interaction.policy.residency:
            self.metrics.increment("shadow_drops", reason="disabled")
            return
        for version in p.eligible_specialist_versions:
            if version in snapshot.disabled_specialists:
                continue
            if not self.breakers.available(version, p.breaker):
                continue
            try:
                candidate = self.loader.load(version, work.identity)
            except Exception:
                continue
            # Recheck controls after potentially slow model loading.
            latest = self.store.snapshot()
            if (
                latest.policy is None
                or latest.policy.policy_id != p.policy_id
                or not latest.enabled(work.identity.tenant_id, work.interaction.task.label)
                or version in latest.disabled_specialists
            ):
                self.metrics.increment("shadow_drops", reason="disabled")
                return
            if candidate.deployment.processing_region != current.residency:
                continue
            if not self.breakers.acquire(version, p.breaker):
                continue
            result, attempt = asyncio.run(
                execute_attempt(
                    candidate,
                    work.request,
                    work.interaction.interaction_id,
                    0,
                    self.timeout_seconds,
                    self.validator,
                    self.metrics,
                    shadow=True,
                )
            )
            self.breakers.record(
                version,
                p.breaker,
                error=attempt.error_code not in (None, "validation_failed"),
                validation_failure=bool(attempt.validation and not attempt.validation.passed),
                latency_ms=attempt.total_latency_ms,
            )
            self.metrics.increment("shadow_cost_micros", attempt.estimated_cost_micros or 0)
            # Permissions may have changed while generation ran.
            current = self.policy.decide(work.identity, work.interaction.application_id)
            if (
                not current.processing_allowed
                or current.residency != work.interaction.policy.residency
            ):
                self.metrics.increment("shadow_drops", reason="disabled")
                return
            if self.persistence.save_shadow(
                work.interaction,
                attempt,
                result.content if result else None,
                current,
                input_redaction_failed=work.input_redaction_failed,
            ):
                self.store.compare(comparison(work, p, result, attempt, self.segments(work)))
                self.metrics.increment("shadow_completed")
            return
        self.metrics.increment("shadow_drops", reason="unhealthy")


def citation_scores(request: ProviderRequest, result: ProviderResult | None) -> tuple[float, float]:
    supplied = {(c.document_id, c.chunk_id) for c in request.context}
    cited = set(CITATION_PATTERN.findall(result.content)) if result else set()
    if result:
        cited.update((c.document_id, c.chunk_id) for c in result.citations)
    matched = len(supplied & cited)
    return (
        matched / len(cited) if cited else float(not supplied),
        matched / len(supplied) if supplied else float(not cited),
    )


def quality_proxy(request: ProviderRequest, result: ProviderResult) -> float:
    precision, _ = citation_scores(request, result)
    return (
        blinded_scores(
            DeterministicJudge(),
            [
                JudgeInput(
                    result.content,
                    tuple(c.content for c in request.context),
                    (),
                    precision == 1,
                )
            ],
            seed=23,
        )[0]
        / 5
    )


def comparison(
    work: ShadowWork,
    policy: RoutePolicy,
    result: ProviderResult | None,
    attempt: GenerationAttempt,
    segments: list[str],
) -> ShadowComparison:
    fp, fr = citation_scores(work.request, work.foundation)
    sp, sr = citation_scores(work.request, result)
    facts = tuple(c.content for c in work.request.context)
    scores = blinded_scores(
        DeterministicJudge(),
        [
            JudgeInput(work.foundation.content, facts, (), fp == 1),
            JudgeInput(result.content if result else "", facts, (), sp == 1),
        ],
        seed=23,
    )
    validation = attempt.validation or Validation(
        passed=False,
        checks=[ValidationCheck(name="generation_available", passed=False)],
    )
    return ShadowComparison(
        judge_version="shadow-chunk-overlap-1",
        interaction_id=work.interaction.interaction_id,
        policy_id=policy.policy_id,
        tenant_id=work.identity.tenant_id,
        specialist_version=attempt.deployment_id,
        segments=segments,
        foundation_score=scores[0] / 5,
        specialist_score=scores[1] / 5
        if result and attempt.error_code in (None, "validation_failed")
        else 0,
        foundation_citation_precision=fp,
        foundation_citation_recall=fr,
        specialist_citation_precision=sp,
        specialist_citation_recall=sr,
        input_token_delta=(attempt.usage.input_tokens if attempt.usage else 0)
        - work.foundation.usage.input_tokens,
        output_token_delta=(attempt.usage.output_tokens if attempt.usage else 0)
        - work.foundation.usage.output_tokens,
        cost_delta_micros=(attempt.estimated_cost_micros or 0)
        - (work.attempt.estimated_cost_micros or 0),
        specialist_cost_micros=attempt.estimated_cost_micros,
        latency_delta_ms=attempt.total_latency_ms - work.attempt.total_latency_ms,
        specialist_validation=validation,
    )
