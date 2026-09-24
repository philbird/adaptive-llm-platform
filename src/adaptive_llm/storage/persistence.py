"""Redact, encrypt and atomically persist the contracts from one interaction."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from adaptive_llm.contracts import (
    DeletionRequest,
    Event,
    EventData,
    EventType,
    GenerationAttempt,
    InferenceRequest,
    InferenceResponse,
    Interaction,
    PolicyDecision,
    RetrievalRun,
    RouteDecision,
    Started,
    now,
    uid,
)
from adaptive_llm.events import EventSink
from adaptive_llm.events.outbox import OutboxStore, publish_metrics
from adaptive_llm.gateway.identity import Keyring
from adaptive_llm.metrics import Metrics
from adaptive_llm.policy.persistence import PersistenceRedactor
from adaptive_llm.storage import EncryptedPayload, MetadataStore, PayloadStore, ReplayRecord
from adaptive_llm.storage.crypto import PayloadCipher


@dataclass(frozen=True)
class InteractionGraph:
    interaction: Interaction
    started: Started
    retrieval: RetrievalRun | None
    route: RouteDecision | None
    attempts: tuple[GenerationAttempt, ...]


@dataclass
class PersistenceContent:
    """Request-local redacted text; each hash is fixed before its record is emitted."""

    messages_json: str | None = None
    query: str | None = None
    output: str | None = None
    input_hash: str | None = None
    query_hash: str | None = None
    output_hash: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    failed: bool = False


class Persistence:
    def __init__(
        self,
        metadata: MetadataStore,
        payloads: PayloadStore,
        cipher: PayloadCipher,
        redactor: PersistenceRedactor,
        keyring: Keyring,
        *,
        outbox: OutboxStore,
        metrics: Metrics,
        fallback_sink: EventSink,
        outbox_pending_limit: int = 100_000,
        replay_ttl_seconds: int = 86_400,
        replay_capacity: int = 10_000,
        retention_seconds: int | None = None,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.outbox = outbox
        self.metrics = metrics
        self.fallback_sink = fallback_sink
        self.outbox_pending_limit = outbox_pending_limit
        self.metadata = metadata
        self.payloads = payloads
        self.cipher = cipher
        self.redactor = redactor
        self.keyring = keyring
        self.replay_ttl_seconds = replay_ttl_seconds
        self.replay_capacity = replay_capacity
        self.retention_seconds = retention_seconds
        self.clock = clock
        self.failures = 0

    def _redact(self, text: str, policy: PolicyDecision, content: PersistenceContent) -> str:
        redacted, counts = self.redactor.redact_text(text, policy)
        for name, count in counts.items():
            content.counts[name] = content.counts.get(name, 0) + count
        return redacted

    def prepare_input(
        self, request: InferenceRequest, policy: PolicyDecision
    ) -> PersistenceContent:
        content = PersistenceContent()
        try:
            messages = [
                message.model_copy(
                    update={"content": self._redact(message.content, policy, content)}
                )
                for message in request.messages
            ]
            for value in request.metadata.values():
                self._redact(value, policy, content)
            content.messages_json = json.dumps([m.model_dump(mode="json") for m in messages])
            content.query = messages[-1].content
            content.input_hash = self.keyring.content_hash(content.messages_json)
            content.query_hash = self.keyring.content_hash(content.query, purpose="query")
        except Exception:
            content.failed = True
        return content

    def prepare_output(
        self, output: str, policy: PolicyDecision, content: PersistenceContent
    ) -> None:
        if content.failed:
            return
        try:
            content.output = self._redact(output, policy, content)
            content.output_hash = self.keyring.content_hash(content.output, purpose="output")
        except Exception:
            # Previously emitted input/query evidence remains valid. Only this failed pass
            # has a null hash; no content from the interaction may be persisted afterward.
            content.failed = True

    def save(
        self,
        graph: InteractionGraph,
        response: InferenceResponse | None,
        content: PersistenceContent,
        fingerprint: str,
        reserved_at: datetime,
    ) -> tuple[Interaction, ReplayRecord | None]:
        interaction = graph.interaction.model_copy(
            update={
                "error_code": "persistence_redaction_failed"
                if content.failed
                else graph.interaction.error_code,
                "policy": graph.interaction.policy.model_copy(
                    update={
                        "persistence_redaction_version": self.redactor.version,
                        "persistence_redaction_counts": content.counts,
                    }
                ),
            }
        )
        graph = replace(graph, interaction=interaction)
        events = [
            Event(
                event_type=kind,
                tenant_id=interaction.tenant_id,
                trace_id=interaction.trace_id,
                data=record,
                occurred_at=interaction.started_at if kind == "interaction.started.v1" else now(),
            )
            for kind, record in self._graph_records(graph)
        ]
        try:
            return self._save(graph, response, content, fingerprint, reserved_at, events)
        except Exception:
            self.record_failure()
            self._emit_degraded(events)
            return interaction, None

    @staticmethod
    def _graph_records(graph: InteractionGraph) -> list[tuple[EventType, EventData]]:
        records: list[tuple[EventType, EventData]] = [("interaction.started.v1", graph.started)]
        if graph.retrieval is not None:
            records.append(("retrieval.completed.v1", graph.retrieval))
        if graph.route is not None:
            records.append(("route.decided.v1", graph.route))
        records.extend(
            ("generation.failed.v1" if attempt.error_code else "generation.completed.v1", attempt)
            for attempt in graph.attempts
        )
        records.append(("interaction.completed.v1", graph.interaction))
        return records

    def _emit_degraded(self, events: Sequence[Event]) -> None:
        # Called in the persistence worker, after rollback, with content-free envelopes only.
        # This is a non-durable attempt, not an acknowledgement of persistence or delivery.
        for event in events:
            self.metrics.increment("degraded_emissions")
            try:
                self.fallback_sink.emit(event)
            except Exception:
                pass

    def _save(
        self,
        graph: InteractionGraph,
        response: InferenceResponse | None,
        content: PersistenceContent,
        fingerprint: str,
        reserved_at: datetime,
        events: list[Event],
    ) -> tuple[Interaction, ReplayRecord | None]:
        interaction = graph.interaction
        tenant, iid = interaction.tenant_id, interaction.interaction_id
        expires = self.clock() + timedelta(
            seconds=self.retention_seconds or interaction.policy.retention_seconds
        )
        replay_expires = min(expires, self.clock() + timedelta(seconds=self.replay_ttl_seconds))
        retrieval = graph.retrieval
        attempts = list(graph.attempts)
        blobs: list[EncryptedPayload] = []
        replay: ReplayRecord | None = None

        def encrypt(content: str, field: str, expiry: datetime = expires) -> str:
            blob = self.cipher.encrypt(content.encode(), tenant, iid, field, expiry)
            blobs.append(blob)
            return blob.reference

        if not content.failed:
            assert content.messages_json is not None and content.query is not None
            logging = interaction.policy.content_logging_allowed
            interaction = interaction.model_copy(
                update={
                    "input": interaction.input.model_copy(
                        update={
                            "messages_ref": encrypt(content.messages_json, "messages")
                            if logging
                            else None,
                        }
                    ),
                }
            )
            if retrieval is not None:
                retrieval = retrieval.model_copy(
                    update={
                        "query_ref": encrypt(content.query, "query") if logging else None,
                    }
                )
            if content.output is not None and response is not None:
                attempts[-1] = attempts[-1].model_copy(
                    update={
                        "output_ref": encrypt(content.output, "output") if logging else None,
                    }
                )
                # Replay is an operational purpose, independent of content logging. Its content
                # still passes the persistence redactor; it expires no later than policy retention.
                redacted_response = response.model_copy(update={"content": content.output})
                replay = ReplayRecord(
                    tenant_id=tenant,
                    application_id=interaction.application_id,
                    request_id=interaction.request_id,
                    interaction_id=iid,
                    fingerprint=fingerprint,
                    response_ref=encrypt(
                        redacted_response.model_dump_json(), "replay", replay_expires
                    ),
                    reserved_at=reserved_at,
                    expires_at=replay_expires,
                )
        records = self._graph_records(
            replace(graph, interaction=interaction, retrieval=retrieval, attempts=tuple(attempts))
        )
        events[:] = [
            event.model_copy(update={"data": record})
            for event, (_, record) in zip(events, records, strict=True)
        ]
        with self.metadata.transaction():
            self.metadata.put(tenant, interaction, expires)
            self.metadata.put(tenant, graph.started, expires)
            if retrieval is not None:
                self.metadata.put(tenant, retrieval, expires)
            if graph.route is not None:
                self.metadata.put(tenant, graph.route, expires)
            for attempt in attempts:
                self.metadata.put(tenant, attempt, expires)
            for blob in blobs:
                self.payloads.put(blob)
            if replay is not None:
                self.metadata.delete_replay(
                    tenant, interaction.application_id, interaction.request_id
                )
                self.metadata.put_replay(replay, self.replay_capacity)
            dropped = self.outbox.enqueue(events, self.outbox_pending_limit)
        self.metrics.increment("dropped_events", dropped)
        self.refresh_metrics()
        return interaction, replay

    def refresh_metrics(self) -> None:
        # Publish after commit even if the consumer is slow or dispatch is disabled.
        # A failed gauge read cannot undo a successful interaction or privacy transaction.
        try:
            publish_metrics(self.outbox, self.metrics, now())
        except Exception:
            self.metrics.increment("dispatcher_failures")

    def save_shadow(
        self,
        interaction: Interaction,
        attempt: GenerationAttempt,
        output: str | None,
        policy: PolicyDecision,
        *,
        input_redaction_failed: bool,
    ) -> bool:
        """Separate attempt/event transaction. Never update the graph's final attempt or replay."""
        tenant, iid = interaction.tenant_id, interaction.interaction_id
        expires = (interaction.completed_at or interaction.started_at) + timedelta(
            seconds=self.retention_seconds or interaction.policy.retention_seconds
        )
        expires = min(expires, self.clock() + timedelta(seconds=policy.retention_seconds))
        if expires <= self.clock():
            return False
        content = PersistenceContent(failed=input_redaction_failed)
        if output is not None:
            self.prepare_output(output, policy, content)
        attempt = attempt.model_copy(update={"output_hash": content.output_hash})
        with self.metadata.transaction():
            if self.metadata.state(tenant, iid) != "active":
                return False
            if (
                not content.failed
                and content.output is not None
                and policy.content_logging_allowed
                and interaction.policy.content_logging_allowed
            ):
                blob = self.cipher.encrypt(
                    content.output.encode(),
                    tenant,
                    iid,
                    f"shadow_output.{attempt.attempt_id}",
                    expires,
                )
                self.payloads.put(blob)
                attempt = attempt.model_copy(update={"output_ref": blob.reference})
            self.metadata.put(tenant, attempt, expires)
            self.metadata.add_shadow_cost(tenant, iid, attempt.estimated_cost_micros or 0)
            self.outbox.enqueue(
                [
                    Event(
                        event_type="generation.failed.v1"
                        if attempt.error_code
                        else "generation.completed.v1",
                        tenant_id=tenant,
                        trace_id=interaction.trace_id,
                        data=attempt,
                    )
                ],
                self.outbox_pending_limit,
            )
        self.refresh_metrics()
        return True

    def record_failure(self) -> None:
        self.failures += 1
        self.metrics.increment("persistence_failures")

    def replay(self, record: ReplayRecord) -> InferenceResponse | None:
        blob = self.payloads.get(record.tenant_id, record.response_ref, self.clock())
        if blob is None:
            return None
        content = self.cipher.decrypt(blob, record.tenant_id, record.interaction_id, "replay")
        return InferenceResponse.model_validate_json(content).model_copy(update={"replayed": True})

    def delete(
        self, tenant_id: str, interaction_id: str, actor: str | None
    ) -> tuple[Interaction, DeletionRequest] | None:
        deletion = DeletionRequest(
            scope="interaction",
            target_id=interaction_id,
            actor_id_pseudonymous=actor,
            requested_at=self.clock(),
        )
        events = [self._deletion_event(tenant_id, uid(), deletion)]
        try:
            with self.metadata.transaction():
                interaction = self.metadata.get(tenant_id, Interaction, interaction_id)
                if interaction is None:
                    return None
                deletion = self.metadata.get_tombstone(tenant_id, interaction_id) or deletion
                events[:] = [self._deletion_event(tenant_id, interaction.trace_id, deletion)]
                self._delete(tenant_id, interaction, deletion)
                self.outbox.enqueue(events, self.outbox_pending_limit)
        except Exception:
            self.record_failure()
            self._emit_degraded(events)
            raise
        self.refresh_metrics()
        return interaction, deletion

    def _delete(self, tenant_id: str, interaction: Interaction, deletion: DeletionRequest) -> None:
        interaction_id = interaction.interaction_id
        self.metadata.tombstone(tenant_id, interaction_id, deletion)
        self.metadata.clear_refs(tenant_id, interaction_id, "deleted")
        self.payloads.delete_interaction(tenant_id, interaction_id)

    def delete_subject(
        self, tenant_id: str, pseudonym: str, actor: str | None
    ) -> tuple[DeletionRequest, list[tuple[Interaction, DeletionRequest]]]:
        deletion = DeletionRequest(
            scope="subject",
            target_id=pseudonym,
            actor_id_pseudonymous=actor,
            requested_at=self.clock(),
        )
        subject_trace_id = uid()
        events = [self._deletion_event(tenant_id, subject_trace_id, deletion)]
        try:
            with self.metadata.transaction():
                deletion = self.metadata.get_subject_tombstone(tenant_id, pseudonym) or deletion
                events[:] = [self._deletion_event(tenant_id, subject_trace_id, deletion)]
                deleted = []
                for interaction in self.metadata.for_subject(tenant_id, pseudonym):
                    interaction_deletion = self.metadata.get_tombstone(
                        tenant_id, interaction.interaction_id
                    ) or DeletionRequest(
                        scope="interaction",
                        target_id=interaction.interaction_id,
                        actor_id_pseudonymous=actor,
                        requested_at=self.clock(),
                    )
                    deleted.append((interaction, interaction_deletion))
                    events.insert(
                        -1,
                        self._deletion_event(tenant_id, interaction.trace_id, interaction_deletion),
                    )
                # Keep the subject summary after its interaction events in due-time order.
                events[-1] = self._deletion_event(tenant_id, subject_trace_id, deletion)
                self.metadata.tombstone_subject(tenant_id, deletion)
                for interaction, interaction_deletion in deleted:
                    self._delete(tenant_id, interaction, interaction_deletion)
                self.outbox.enqueue(events, self.outbox_pending_limit)
        except Exception:
            self.record_failure()
            self._emit_degraded(events)
            raise
        self.refresh_metrics()
        return deletion, deleted

    @staticmethod
    def _deletion_event(tenant_id: str, trace_id: str, deletion: DeletionRequest) -> Event:
        return Event(
            event_id=deletion.deletion_request_id,
            event_type="privacy.deletion.requested.v1",
            tenant_id=tenant_id,
            trace_id=trace_id,
            data=deletion,
        )
