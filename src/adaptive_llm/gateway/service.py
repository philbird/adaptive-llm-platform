"""One authenticated request, one foundation attempt, and content-free correlated events."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from threading import Lock
from time import perf_counter

from opentelemetry.trace import Tracer

from adaptive_llm.contracts import (
    Candidate,
    Chunk,
    GenerationAttempt,
    InferenceRequest,
    InferenceResponse,
    InputSummary,
    Interaction,
    LiveObservation,
    PolicyDecision,
    RetrievalRun,
    RouteDecision,
    RoutePolicy,
    RouteSummary,
    RoutingFeatures,
    Started,
    now,
    uid,
)
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.policy import PolicyEngine, ProcessingRedactor
from adaptive_llm.providers import TOKENIZER, Provider, ProviderRequest, ProviderResult, token_count
from adaptive_llm.rag import Retriever
from adaptive_llm.routing import Router
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.canary import LiveOutcomes
from adaptive_llm.routing.chain import (
    ChainPlan,
    ChainPlanner,
    ExecutionCandidate,
    rejection,
    run_chain,
)
from adaptive_llm.routing.shadow import ShadowWork, quality_proxy
from adaptive_llm.routing.tasks import RulesClassifier, TaskClassifier
from adaptive_llm.storage import ReplayRecord
from adaptive_llm.storage.persistence import InteractionGraph, Persistence, PersistenceContent
from adaptive_llm.validation import TracedValidator, Validator


class InferenceService:
    def __init__(
        self,
        *,
        keyring: Keyring,
        replay_capacity: int,
        policy: PolicyEngine,
        redactor: ProcessingRedactor,
        retriever: Retriever,
        router: Router,
        provider: Provider,
        validator: Validator,
        tracer: Tracer,
        persistence: Persistence,
        chain_planner: ChainPlanner | None = None,
        breakers: CircuitBreakers | None = None,
        classifier: TaskClassifier | None = None,
    ) -> None:
        if replay_capacity < 1:
            raise ValueError("invalid_replay_capacity")
        self._keyring = keyring
        self._replay_capacity = replay_capacity
        self._policy = policy
        self._redactor = redactor
        self._retriever = retriever
        self._router = router
        self._provider = provider
        self._validator = validator
        self.classifier = classifier or RulesClassifier()
        self._tracer = tracer
        self.persistence = persistence
        self.chain_planner = chain_planner
        self.live_outcomes: LiveOutcomes | None = None
        self.breakers = breakers or CircuitBreakers(persistence.metrics)
        self._replays: dict[tuple[str, str, str], str] = {}
        self._replay_lock = Lock()

    async def infer(
        self,
        request: InferenceRequest,
        identity: Identity,
        schedule_shadow: Callable[[ShadowWork], None] | None = None,
    ) -> InferenceResponse:
        if request.application_id not in identity.application_ids:
            raise GatewayError(403, "application_forbidden")
        if request.stream:
            raise GatewayError(501, "streaming_not_supported")
        key = (identity.tenant_id, request.application_id, request.request_id)
        fingerprint = self._keyring.fingerprint(
            json.dumps(request.model_dump(mode="json"), sort_keys=True)
        )
        at = self.persistence.clock()
        with self._replay_lock:
            entry = self._replays.get(key)
            if entry is not None:
                if entry != fingerprint:
                    raise GatewayError(409, "request_id_conflict")
                raise GatewayError(409, "request_in_progress")
            if len(self._replays) >= self._replay_capacity:
                raise GatewayError(429, "replay_capacity_exceeded")
            self._replays[key] = fingerprint
        try:
            replayed = await asyncio.to_thread(self._find_replay, key, fingerprint)
            if replayed is not None:
                self.persistence.metrics.increment("replay_hits")
                return replayed
            with self._tracer.start_as_current_span(
                "inference",
                attributes={"schema_version": "1.0"},
                record_exception=False,
                set_status_on_exception=False,
            ):
                response, _ = await self._execute(
                    request, identity, fingerprint, at, schedule_shadow
                )
            return response
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(500, "internal_error") from None
        finally:
            # Completed entries exist solely in SQLite; memory bounds only in-flight work.
            with self._replay_lock:
                del self._replays[key]

    def _find_replay(self, key: tuple[str, str, str], fingerprint: str) -> InferenceResponse | None:
        stored = self.persistence.metadata.get_replay(*key, self.persistence.clock())
        if stored is None:
            return None
        if stored.fingerprint != fingerprint:
            raise GatewayError(409, "request_id_conflict")
        return self.persistence.replay(stored)

    async def _execute(
        self,
        request: InferenceRequest,
        identity: Identity,
        fingerprint: str,
        reserved_at: datetime,
        schedule_shadow: Callable[[ShadowWork], None] | None = None,
    ) -> tuple[InferenceResponse, ReplayRecord | None]:
        started_at = now()
        started = perf_counter()
        interaction_id, trace_id = uid(), uid()
        attributes = {"interaction_id": interaction_id, "trace_id": trace_id}
        with self._tracer.start_as_current_span(
            "policy", attributes=attributes, record_exception=False, set_status_on_exception=False
        ) as span:
            policy = self._policy.decide(identity, request.application_id)
            if not policy.processing_allowed:
                raise GatewayError(403, "processing_forbidden")
            request, counts = self._redactor.redact(request)
            policy = policy.model_copy(
                update={
                    "processing_redaction_version": self._redactor.version,
                    "processing_redaction_counts": counts,
                }
            )
            span.set_attribute("policy_version", policy.policy_version)
        started_record = Started(
            interaction_id=interaction_id,
            application_id=request.application_id,
            policy_version=policy.policy_version,
        )
        content = self.persistence.prepare_input(request, policy)
        task = self.classifier.classify(request)
        retrieval_id: str | None = None
        route_id: str | None = None
        attempts: list[GenerationAttempt] = []
        retrieval_run: RetrievalRun | None = None
        route: RouteDecision | None = None
        response: InferenceResponse | None = None
        replay: ReplayRecord | None = None
        failure: str | None = None
        features: RoutingFeatures | None = None
        try:
            supplied_chunks: tuple[Chunk, ...] = ()
            if request.rag.enabled:
                with self._tracer.start_as_current_span(
                    "retrieval",
                    attributes=attributes,
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    retrieval = await self._retriever.retrieve(
                        request.messages[-1].content,
                        identity,
                        request.application_id,
                        request.rag,
                        interaction_id,
                        residency=policy.residency,
                    )
                    span.set_attribute("index_version", retrieval.run.index_version)
                retrieval_id = retrieval.run.retrieval_run_id
                retrieval_run = retrieval.run.model_copy(update={"query_hash": content.query_hash})
                supplied_chunks = retrieval.supplied_chunks
            provider_request = ProviderRequest(
                messages=tuple(request.messages),
                context=supplied_chunks,
                response_format=request.response_format,
                max_output_tokens=request.max_output_tokens,
                application_id=request.application_id,
                deadline_ms=request.routing.deadline_ms,
            )
            features = RoutingFeatures(
                task=task.label,
                input_tokens=provider_request.input_tokens,
                chunk_count=len(supplied_chunks),
                context_supplied=bool(supplied_chunks),
                index_id=request.rag.index_id,
                top_score=max(
                    (c.retrieval_score for c in retrieval_run.candidates if c.supplied_to_model),
                    default=0,
                )
                if retrieval_run
                else 0,
            )
            routing_started = perf_counter()
            with self._tracer.start_as_current_span(
                "routing",
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                selection = self._router.route(
                    provider_request, request.routing, policy, interaction_id
                )
                span.set_attribute("router_version", selection.decision.router_version)
            route_id = selection.decision.route_decision_id
            schema = request.response_format.json_schema
            route = selection.decision.model_copy(
                update={
                    "response_schema_name": schema.name if schema else None,
                    "response_schema_sha256": schema.sha256 if schema else None,
                }
            )
            selection = replace(
                selection,
                decision=route,
                features=features,
                request=provider_request,
                policy=policy,
            )
            plan = (
                await asyncio.to_thread(
                    self.chain_planner.plan,
                    selection,
                    request.routing,
                    identity,
                    task.label,
                )
                if self.chain_planner
                else ChainPlan(
                    candidates=(ExecutionCandidate(selection.deployment, self._provider),),
                    policy=RoutePolicy(
                        foundation_fallback=selection.deployment.model_deployment_id
                    ),
                    fallback_reason="live_specialists_disabled"
                    if request.routing.mode == "specialist"
                    else None,
                )
            )
            candidates = []
            for candidate in plan.considered or plan.candidates:
                reason = rejection(candidate, plan, provider_request, request.routing, policy, 0)
                if reason is None and not self.breakers.available(
                    candidate.deployment.model_deployment_id,
                    plan.policy.breaker,
                ):
                    reason = "circuit_open"
                candidates.append(
                    Candidate(
                        model_deployment_id=candidate.deployment.model_deployment_id,
                        eligible=reason is None,
                        processing_region=candidate.deployment.processing_region,
                        estimated_cost_micros=candidate.deployment.price_list.estimate(
                            provider_request.input_tokens,
                            provider_request.max_output_tokens,
                        ),
                        price_list_version=candidate.deployment.price_list.version,
                        estimated_latency_ms=candidate.estimated_latency_ms
                        if candidate.has_estimate
                        else None,
                        predicted_quality=candidate.quality
                        if candidate.specialist and candidate.has_estimate
                        else None,
                        confidence=candidate.confidence
                        if candidate.specialist and candidate.has_estimate
                        else None,
                        ood_score=candidate.ood_score
                        if candidate.specialist and candidate.has_estimate
                        else None,
                        reason_codes=[reason]
                        if reason
                        else [
                            "specialist_candidate" if candidate.specialist else "foundation_only"
                        ],
                    )
                )
            eligible = [c.model_deployment_id for c in candidates if c.eligible]
            route = route.model_copy(
                update={
                    "route_policy_id": plan.policy.policy_id
                    if self.chain_planner and plan.policy.policy_id != "foundation-only"
                    else None,
                    "fallback_reasons": [plan.fallback_reason] if plan.fallback_reason else [],
                    "candidates": candidates,
                    "router_version": plan.policy.router_version or route.router_version,
                    "decision_latency_ms": (perf_counter() - routing_started) * 1000,
                    "selected_model_deployment_id": eligible[0] if eligible else None,
                    "fallback_deployment_ids": eligible[1:],
                }
            )
            if not eligible and selection.decision.selected_model_deployment_id is None:
                reasons = {
                    reason
                    for candidate in selection.decision.candidates
                    for reason in candidate.reason_codes
                }
                if "processing_denied" in reasons:
                    raise GatewayError(403, "processing_forbidden")
                if "residency_mismatch" in reasons:
                    raise GatewayError(403, "residency_unavailable")
                if "cost_limit_exceeded" in reasons:
                    raise GatewayError(422, "cost_limit_exceeded")
                raise GatewayError(503, "no_healthy_deployment")
            response = await self._generate(
                provider_request,
                interaction_id,
                trace_id,
                started + request.routing.deadline_ms / 1000,
                request,
                plan,
                attempts,
                policy,
                content,
            )
        except GatewayError as error:
            failure = error.code
            raise
        except asyncio.CancelledError:
            failure = "request_cancelled"
            raise
        except Exception:
            failure = "pipeline_failed"
            raise GatewayError(502, failure) from None
        finally:
            if route is not None:
                route = route.model_copy(
                    update={
                        "fallback_reasons": list(
                            dict.fromkeys(
                                [
                                    *route.fallback_reasons,
                                    *(
                                        a.fallback_reason
                                        for a in attempts
                                        if a.fallback_reason is not None
                                    ),
                                ]
                            )
                        ),
                    }
                )
            interaction = Interaction(
                interaction_id=interaction_id,
                trace_id=trace_id,
                request_id=request.request_id,
                tenant_id=identity.tenant_id,
                subject_id_pseudonymous=identity.subject_id_pseudonymous,
                application_id=request.application_id,
                environment=identity.environment,
                started_at=started_at,
                completed_at=now(),
                task=task,
                policy=policy,
                input=self._input_summary(request, content),
                retrieval_run_id=retrieval_id,
                route_decision_id=route_id,
                generation_attempt_ids=[attempt.attempt_id for attempt in attempts],
                final_attempt_id=attempts[-1].attempt_id if attempts else None,
                status="completed" if failure is None else "failed",
                error_code=failure,
                total_cost_micros=sum(a.estimated_cost_micros or 0 for a in attempts),
            )
            cancelled_during_save = False
            try:
                saving = asyncio.create_task(
                    asyncio.to_thread(
                        self.persistence.save,
                        InteractionGraph(
                            interaction, started_record, retrieval_run, route, tuple(attempts)
                        ),
                        response,
                        content,
                        fingerprint,
                        reserved_at,
                    )
                )
                try:
                    interaction, replay = await asyncio.shield(saving)
                except asyncio.CancelledError:
                    # A worker cannot be cancelled once SQLite is writing. Keep the reservation
                    # until it finishes so a retry cannot race the still-running transaction.
                    cancelled_during_save = True
                    interaction, replay = await saving
            except Exception:
                self.persistence.record_failure()
            if (
                self.live_outcomes is not None
                and route is not None
                and route.route_policy_id
                and features
            ):
                try:
                    specialist = next((a for a in attempts if a.adapter_id is not None), None)
                    quality = (
                        quality_proxy(
                            provider_request,
                            ProviderResult(
                                content=response.content,
                                citations=tuple(response.citations),
                                usage=response.usage,
                                finish_reason=response.finish_reason,
                                latency_ms=attempts[-1].total_latency_ms,
                            ),
                        )
                        if response
                        else 0
                    )
                    await asyncio.to_thread(
                        self.live_outcomes.record_live,
                        LiveObservation(
                            interaction_id=interaction_id,
                            policy_id=route.route_policy_id,
                            tenant_id=identity.tenant_id,
                            features=features,
                            specialist_version=specialist.deployment_id if specialist else None,
                            specialist_served=bool(response and response.route.specialist_served),
                            fallback_used=bool(response and response.route.fallback_used),
                            success=response is not None,
                            quality=quality,
                            validation_failure=any(
                                a.validation is not None and not a.validation.passed
                                for a in attempts
                            ),
                            error=any(
                                a.error_code not in (None, "validation_failed") for a in attempts
                            ),
                            critical_safety_incidents=sum(
                                not c.passed
                                and c.severity == "hard"
                                and (c.critical_safety or c.name == "tool_allowlist")
                                for a in attempts
                                if a.validation
                                for c in a.validation.checks
                            ),
                            total_cost_micros=interaction.total_cost_micros,
                            latency_ms=(perf_counter() - started) * 1000,
                        ),
                    )
                except Exception:
                    self.persistence.record_failure()
            if cancelled_during_save:
                raise asyncio.CancelledError
        assert response is not None
        if schedule_shadow is not None and not response.route.specialist_served:
            schedule_shadow(
                ShadowWork(
                    request=provider_request,
                    foundation=ProviderResult(
                        content=response.content,
                        citations=tuple(response.citations),
                        usage=response.usage,
                        finish_reason=response.finish_reason,
                        latency_ms=attempts[-1].total_latency_ms,
                    ),
                    attempt=attempts[-1],
                    interaction=interaction,
                    identity=identity,
                    input_redaction_failed=content.failed,
                    policy_id=plan.policy.policy_id,
                )
            )
        return response, replay

    def _input_summary(
        self, request: InferenceRequest, content: PersistenceContent
    ) -> InputSummary:
        return InputSummary(
            content_hash=content.input_hash,
            token_count=sum(token_count(m.content) for m in request.messages),
            tokenizer=TOKENIZER,
        )

    async def _generate(
        self,
        request: ProviderRequest,
        interaction_id: str,
        trace_id: str,
        deadline: float,
        ingress: InferenceRequest,
        plan: ChainPlan,
        attempts: list[GenerationAttempt],
        policy: PolicyDecision,
        content: PersistenceContent,
    ) -> InferenceResponse:
        attributes = {"interaction_id": interaction_id, "trace_id": trace_id}
        with self._tracer.start_as_current_span(
            "generation",
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ):
            result, attempt = await run_chain(
                plan,
                request,
                ingress.routing,
                policy,
                interaction_id,
                deadline,
                attempts,
                TracedValidator(self._validator, self._tracer, attributes),
                self.breakers,
                self.persistence.metrics,
            )
        self.persistence.prepare_output(result.content, policy, content)
        attempts[-1] = attempt.model_copy(update={"output_hash": content.output_hash})
        assert result.finish_reason in ("stop", "length", "content_filter")
        return InferenceResponse(
            interaction_id=interaction_id,
            trace_id=trace_id,
            model_deployment_id=attempt.deployment_id,
            content=result.content,
            citations=list(result.citations),
            usage=result.usage,
            estimated_cost_micros=sum(a.estimated_cost_micros or 0 for a in attempts),
            route=RouteSummary(
                fallback_used=len(attempts) > 1 or attempt.fallback_reason is not None,
                specialist_served=attempt.adapter_id is not None,
            ),
            finish_reason=result.finish_reason,
        )
