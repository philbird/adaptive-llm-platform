"""One authenticated request, one foundation attempt, and content-free correlated events."""

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from opentelemetry.trace import Tracer

from adaptive_llm.contracts import (
    Chunk,
    Event,
    EventData,
    EventType,
    GenerationAttempt,
    InferenceRequest,
    InferenceResponse,
    InputSummary,
    Interaction,
    RequestParameters,
    Started,
    Task,
    now,
    uid,
)
from adaptive_llm.events import EventSink
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.policy import PolicyEngine, ProcessingRedactor
from adaptive_llm.providers import TOKENIZER, Provider, ProviderRequest, token_count
from adaptive_llm.rag import Retriever
from adaptive_llm.routing import Router, RouteSelection
from adaptive_llm.validation import Validator


@dataclass
class ReplayEntry:
    fingerprint: str
    response: InferenceResponse | None = None


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
        events: EventSink,
        tracer: Tracer,
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
        self._events = events
        self._tracer = tracer
        self._replays: OrderedDict[tuple[str, str, str], ReplayEntry] = OrderedDict()
        self._replay_lock = Lock()
        self.emission_failures = 0

    def _emit(
        self, event_type: EventType, data: EventData, identity: Identity, trace_id: str
    ) -> None:
        # Neither exporter exceptions nor sink exceptions may escape into serving.
        # OTel must not automatically record arbitrary exception messages.
        try:
            with self._tracer.start_as_current_span(
                "event_emission",
                attributes={"trace_id": trace_id, "schema_version": "1.0"},
                record_exception=False,
                set_status_on_exception=False,
            ):
                self._events.emit(
                    Event(
                        event_type=event_type,
                        tenant_id=identity.tenant_id,
                        trace_id=trace_id,
                        data=data,
                    )
                )
        except Exception:
            self.emission_failures += 1

    async def infer(self, request: InferenceRequest, identity: Identity) -> InferenceResponse:
        if request.application_id not in identity.application_ids:
            raise GatewayError(403, "application_forbidden")
        if request.stream:
            raise GatewayError(501, "streaming_not_supported")
        key = (identity.tenant_id, request.application_id, request.request_id)
        fingerprint = self._keyring.fingerprint(
            json.dumps(request.model_dump(mode="json"), sort_keys=True)
        )
        with self._replay_lock:
            entry = self._replays.get(key)
            if entry is not None:
                if entry.fingerprint != fingerprint:
                    raise GatewayError(409, "request_id_conflict")
                if entry.response is not None:
                    return entry.response.model_copy(update={"replayed": True}, deep=True)
                raise GatewayError(409, "request_in_progress")
            if len(self._replays) >= self._replay_capacity:
                # Active reservations cannot be evicted without allowing duplicate execution.
                oldest_completed = next(
                    (key for key, entry in self._replays.items() if entry.response is not None),
                    None,
                )
                if oldest_completed is None:
                    raise GatewayError(429, "replay_capacity_exceeded")
                del self._replays[oldest_completed]
            entry = ReplayEntry(fingerprint=fingerprint)
            self._replays[key] = entry
        try:
            with self._tracer.start_as_current_span(
                "inference",
                attributes={"schema_version": "1.0"},
                record_exception=False,
                set_status_on_exception=False,
            ):
                response = await self._execute(request, identity)
            with self._replay_lock:
                entry.response = response.model_copy(deep=True)
            return response
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(500, "internal_error") from None
        finally:
            # Includes cancellation and failures before/after the provider call. Only a
            # successful response survives, so a retry can reserve this id afresh.
            with self._replay_lock:
                if entry.response is None:
                    del self._replays[key]

    async def _execute(self, request: InferenceRequest, identity: Identity) -> InferenceResponse:
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
        self._emit(
            "interaction.started.v1",
            Started(
                interaction_id=interaction_id,
                application_id=request.application_id,
                policy_version=policy.policy_version,
            ),
            identity,
            trace_id,
        )
        retrieval_id: str | None = None
        route_id: str | None = None
        attempts: list[GenerationAttempt] = []
        failure: str | None = None
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
                supplied_chunks = retrieval.supplied_chunks
                self._emit("retrieval.completed.v1", retrieval.run, identity, trace_id)
            provider_request = ProviderRequest(
                messages=tuple(request.messages),
                context=supplied_chunks,
                response_format=request.response_format,
                max_output_tokens=request.max_output_tokens,
            )
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
            self._emit("route.decided.v1", selection.decision, identity, trace_id)
            if selection.decision.selected_model_deployment_id is None:
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
                raise GatewayError(500, "invalid_route_decision")
            remaining = request.routing.deadline_ms / 1000 - (perf_counter() - started)
            return await self._generate(
                provider_request, selection, identity, interaction_id, trace_id, remaining, attempts
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
            self._emit(
                "interaction.completed.v1",
                Interaction(
                    interaction_id=interaction_id,
                    trace_id=trace_id,
                    request_id=request.request_id,
                    tenant_id=identity.tenant_id,
                    subject_id_pseudonymous=identity.subject_id_pseudonymous,
                    application_id=request.application_id,
                    environment=identity.environment,
                    started_at=started_at,
                    completed_at=now(),
                    task=Task(
                        label="question_answering" if request.rag.enabled else "general",
                        classifier_version="placeholder-rag-flag-1",
                        confidence=0.5,
                        reason_codes=["rag_flag_only"],
                    ),
                    policy=policy,
                    input=self._input_summary(request),
                    retrieval_run_id=retrieval_id,
                    route_decision_id=route_id,
                    generation_attempt_ids=[attempt.attempt_id for attempt in attempts],
                    final_attempt_id=attempts[-1].attempt_id if attempts else None,
                    status="completed" if failure is None else "failed",
                    error_code=failure,
                ),
                identity,
                trace_id,
            )

    def _input_summary(self, request: InferenceRequest) -> InputSummary:
        return InputSummary(
            content_hash=self._keyring.content_hash(
                json.dumps([m.model_dump(mode="json") for m in request.messages]), purpose="input"
            ),
            token_count=sum(token_count(m.content) for m in request.messages),
            tokenizer=TOKENIZER,
        )

    async def _generate(
        self,
        request: ProviderRequest,
        selection: RouteSelection,
        identity: Identity,
        interaction_id: str,
        trace_id: str,
        remaining: float,
        attempts: list[GenerationAttempt],
    ) -> InferenceResponse:
        deployment = selection.deployment
        attributes = {
            "interaction_id": interaction_id,
            "trace_id": trace_id,
            "model_version": deployment.model_version,
        }
        attempt = GenerationAttempt(
            interaction_id=interaction_id,
            model_provider=deployment.model_provider,
            model_id=deployment.model_id,
            model_version=deployment.model_version,
            deployment_id=deployment.model_deployment_id,
            request_parameters=RequestParameters(max_output_tokens=request.max_output_tokens),
            total_latency_ms=0,
            price_list_version=deployment.price_list.version,
            finish_reason="error",
        )
        error_code: str | None = None
        generation_started = perf_counter()
        try:
            with self._tracer.start_as_current_span(
                "generation",
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            ):
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    result = await self._provider.generate(request)
                attempt = attempt.model_copy(
                    update={
                        "usage": result.usage,
                        "finish_reason": result.finish_reason,
                        "total_latency_ms": (perf_counter() - generation_started) * 1000,
                        "estimated_cost_micros": deployment.price_list.estimate(
                            result.usage.input_tokens, result.usage.output_tokens
                        ),
                    }
                )
            if result.finish_reason == "deadline_exceeded":
                raise GatewayError(504, "provider_deadline_exceeded")
            if result.finish_reason not in ("stop", "length", "content_filter"):
                raise GatewayError(502, "provider_failed")
            with self._tracer.start_as_current_span(
                "validation",
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                validation = self._validator.validate(request, result)
                span.set_attribute("validator_version", validation.validator_version)
            attempt = attempt.model_copy(
                update={
                    "validation": validation,
                    "output_hash": self._keyring.content_hash(result.content, purpose="output"),
                }
            )
            if not validation.passed:
                raise GatewayError(502, "validation_failed")
            return InferenceResponse(
                interaction_id=interaction_id,
                trace_id=trace_id,
                model_deployment_id=deployment.model_deployment_id,
                content=result.content,
                citations=list(result.citations),
                usage=result.usage,
                estimated_cost_micros=attempt.estimated_cost_micros or 0,
                finish_reason=result.finish_reason,
            )
        except TimeoutError:
            error_code = "provider_deadline_exceeded"
            attempt = attempt.model_copy(update={"finish_reason": "deadline_exceeded"})
            raise GatewayError(504, error_code) from None
        except GatewayError as error:
            error_code = error.code
            raise
        except asyncio.CancelledError:
            error_code = "request_cancelled"
            attempt = attempt.model_copy(update={"finish_reason": "cancelled"})
            raise
        except Exception:
            error_code = "provider_failed"
            raise GatewayError(502, error_code) from None
        finally:
            attempt = attempt.model_copy(
                update={
                    "error_code": error_code,
                    "total_latency_ms": attempt.total_latency_ms
                    or (perf_counter() - generation_started) * 1000,
                }
            )
            attempts.append(attempt)
            self._emit(
                "generation.completed.v1" if error_code is None else "generation.failed.v1",
                attempt,
                identity,
                trace_id,
            )
