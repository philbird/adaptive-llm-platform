"""Transactional, tenant-scoped user labels and persistence-redacted corrections."""

from adaptive_llm.contracts import (
    CorrectionInput,
    Event,
    Feedback,
    FeedbackInput,
    FeedbackValue,
    Interaction,
)
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.storage.persistence import Persistence


class FeedbackService:
    def __init__(self, persistence: Persistence, policy: PolicyEngine) -> None:
        self.persistence = persistence
        self.policy = policy

    def record(
        self, identity: Identity, interaction_id: str, body: FeedbackInput | CorrectionInput
    ) -> Feedback:
        persistence = self.persistence
        metadata = persistence.metadata
        tenant = identity.tenant_id
        with metadata.transaction():
            interaction = metadata.get(tenant, Interaction, interaction_id)
            if interaction is None:
                raise GatewayError(404, "interaction_not_found")
            if interaction.application_id not in identity.application_ids:
                raise GatewayError(403, "application_forbidden")
            policy = self.policy.decide(identity, interaction.application_id)
            if not policy.processing_allowed:
                raise GatewayError(403, "processing_forbidden")
            expires = metadata.expires_at(tenant, interaction_id)
            if (
                metadata.state(tenant, interaction_id) != "active"
                or metadata.get_tombstone(tenant, interaction_id) is not None
                or (
                    interaction.subject_id_pseudonymous is not None
                    and metadata.get_subject_tombstone(tenant, interaction.subject_id_pseudonymous)
                    is not None
                )
                or expires is None
                or expires <= persistence.clock()
            ):
                raise GatewayError(409, "interaction_inactive")
            correction = isinstance(body, CorrectionInput)
            feedback = Feedback(
                interaction_id=interaction_id,
                source="user",
                label_type="correction" if isinstance(body, CorrectionInput) else body.label_type,
                value=FeedbackValue(score=1, max_score=1)
                if isinstance(body, CorrectionInput)
                else body.value,
                rubric_version=None if isinstance(body, CorrectionInput) else body.rubric_version,
                training_authorised=body.training_authorised,
                actor_id_pseudonymous=identity.subject_id_pseudonymous,
                created_at=persistence.clock(),
            )
            content = body.correction if isinstance(body, CorrectionInput) else body.comment
            if content is not None and policy.content_logging_allowed:
                try:
                    redacted, _ = persistence.redactor.redact_text(content, policy)
                except Exception:
                    feedback = feedback.model_copy(
                        update={"error_code": "persistence_redaction_failed"}
                    )
                else:
                    field = "correction" if correction else "comment"
                    blob = persistence.cipher.encrypt(
                        redacted.encode(), tenant, interaction_id, field, expires
                    )
                    persistence.payloads.put(blob)
                    feedback = feedback.model_copy(
                        update={
                            f"{field}_ref": blob.reference,
                            "content_hash": persistence.keyring.content_hash(
                                redacted, purpose="output"
                            ),
                        }
                    )
            metadata.put(tenant, feedback, expires)
            metadata.append_feedback(tenant, interaction, feedback.feedback_id)
            persistence.outbox.enqueue(
                [
                    Event(
                        event_type="feedback.recorded.v1",
                        tenant_id=tenant,
                        trace_id=interaction.trace_id,
                        data=feedback,
                    )
                ],
                persistence.outbox_pending_limit,
            )
        persistence.refresh_metrics()
        return feedback
