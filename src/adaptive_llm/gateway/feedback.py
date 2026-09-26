"""Transactional, tenant-scoped user labels and persistence-redacted corrections."""

import json

from adaptive_llm.contracts import (
    CorrectionInput,
    Event,
    Feedback,
    FeedbackInput,
    FeedbackValue,
    Interaction,
    ResponseFormat,
)
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.structured import matches_schema, reject_constant


class FeedbackService:
    def __init__(self, persistence: Persistence, policy: PolicyEngine) -> None:
        self.persistence = persistence
        self.policy = policy

    def _response_format(self, interaction: Interaction) -> ResponseFormat | None:
        reference = interaction.input.response_format_ref
        if reference is None:
            return None
        persistence = self.persistence
        try:
            blob = persistence.payloads.get(interaction.tenant_id, reference, persistence.clock())
            if blob is None:
                raise ValueError
            return ResponseFormat.model_validate_json(
                persistence.cipher.decrypt(
                    blob, interaction.tenant_id, interaction.interaction_id, "response_format"
                )
            )
        except Exception:
            raise GatewayError(503, "correction_schema_unavailable") from None

    @staticmethod
    def _validate_target(content: str, response_format: ResponseFormat | None) -> None:
        if response_format is None or response_format.type == "text":
            return
        if response_format.json_schema is not None:
            valid = matches_schema(content, response_format.json_schema.schema_)
        else:
            try:
                valid = isinstance(json.loads(content, parse_constant=reject_constant), dict)
            except (ValueError, RecursionError):
                valid = False
        if not valid:
            raise GatewayError(422, "invalid_correction_target")

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
            response_format = self._response_format(interaction) if correction else None
            if isinstance(body, CorrectionInput):
                self._validate_target(body.correction, response_format)
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
                    if correction:
                        self._validate_target(redacted, response_format)
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
