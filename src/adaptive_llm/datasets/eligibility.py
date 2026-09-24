"""Select metadata-only candidates using current policy and lifecycle state."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from adaptive_llm.contracts import (
    DatasetSpecification,
    Feedback,
    GenerationAttempt,
    Interaction,
    PolicyDecision,
    RetrievalRun,
)
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.storage import MetadataStore


@dataclass(frozen=True)
class Example:
    """Only metadata, references and hashes; never decrypted text."""

    interaction: Interaction
    policy: PolicyDecision
    retrieval: RetrievalRun | None
    attempt: GenerationAttempt | None
    feedback: tuple[Feedback, ...]


@dataclass(frozen=True)
class Selection:
    examples: list[Example]
    considered: int
    exclusions: Counter[str]


def select(
    metadata: MetadataStore,
    policy: PolicyEngine,
    specification: DatasetSpecification,
    at: datetime,
) -> Selection:
    examples: list[Example] = []
    exclusions: Counter[str] = Counter()
    considered = 0
    for tenant in sorted(specification.tenant_ids):
        for interaction in metadata.in_window(
            tenant, specification.source_window.start, specification.source_window.end
        ):
            considered += 1
            identity = Identity(
                tenant,
                frozenset({interaction.application_id}),
                interaction.environment,
                interaction.subject_id_pseudonymous,
            )
            current = policy.decide(identity, interaction.application_id)
            if current.policy_version != specification.eligibility_policy_version:
                raise GatewayError(409, "eligibility_policy_version_mismatch")
            reasons: set[str] = set()
            if interaction.application_id == "evaluation":
                reasons.add("evaluation_interaction")
            if not current.training_allowed or not current.processing_allowed:
                reasons.add("training_forbidden")
            # Evaluation datasets also require their own purpose permission.
            if specification.purpose == "evaluation" and not current.evaluation_allowed:
                reasons.add("evaluation_forbidden")
            if interaction.status != "completed" and specification.purpose != "router_training":
                reasons.add("not_completed")
            if metadata.get_tombstone(tenant, interaction.interaction_id) is not None:
                reasons.add("interaction_deleted")
            subject = interaction.subject_id_pseudonymous
            if subject is not None and metadata.get_subject_tombstone(tenant, subject) is not None:
                reasons.add("subject_deleted")
            expires = metadata.expires_at(tenant, interaction.interaction_id)
            if (
                expires is None
                or expires <= at
                or metadata.state(tenant, interaction.interaction_id) == "expired"
            ):
                reasons.add("expired")
            if metadata.state(tenant, interaction.interaction_id) == "deleted":
                reasons.add("interaction_deleted")
            if interaction.error_code == "persistence_redaction_failed":
                reasons.add("persistence_redaction_failed")
            retrieval = (
                metadata.get(tenant, RetrievalRun, interaction.retrieval_run_id)
                if interaction.retrieval_run_id
                else None
            )
            if interaction.retrieval_run_id and retrieval is None:
                reasons.add("missing_retrieval")
            if retrieval is not None and any(
                chunk.licence_class not in {"synthetic", "internal-approved"}
                for chunk in retrieval.candidates
                if chunk.supplied_to_model
            ):
                reasons.add("unsupported_licence")
            feedback = tuple(
                record
                for fid in interaction.feedback_ids
                if (record := metadata.get(tenant, Feedback, fid)) is not None
            )
            if len(feedback) != len(interaction.feedback_ids):
                reasons.add("missing_feedback")
            corrections = [
                item
                for item in feedback
                if item.label_type == "correction"
                and item.training_authorised
                and item.correction_ref
                and not item.error_code
            ]
            if specification.purpose != "router_training" and any(
                item.source == "user"
                and item.label_type in {"thumb", "rubric"}
                and item.value.score * 2 < item.value.max_score
                and not any(c.created_at > item.created_at for c in corrections)
                for item in feedback
            ):
                reasons.add("unresolved_negative_feedback")
            attempt = (
                metadata.get(tenant, GenerationAttempt, interaction.final_attempt_id)
                if interaction.final_attempt_id
                else None
            )
            if specification.purpose != "router_training" and (
                not interaction.input.messages_ref or not interaction.input.content_hash
            ):
                reasons.add("missing_input")
            if reasons:
                exclusions.update(reasons)
            else:
                examples.append(Example(interaction, current, retrieval, attempt, feedback))
    examples.sort(key=lambda e: (e.interaction.started_at, e.interaction.interaction_id))
    return Selection(examples, considered, exclusions)
