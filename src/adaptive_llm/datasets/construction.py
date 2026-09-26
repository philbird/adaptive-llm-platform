"""Decrypt, verify, redact and construct canonical examples solely in memory."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import TypeAdapter

from adaptive_llm.contracts import DatasetSpecification, Message, TargetSource
from adaptive_llm.datasets.eligibility import Example
from adaptive_llm.datasets.sources import SourceResolver
from adaptive_llm.gateway.identity import Keyring
from adaptive_llm.policy.persistence import PersistenceRedactor
from adaptive_llm.storage import EncryptedPayload, PayloadReader, StorageError
from adaptive_llm.storage.crypto import PayloadCipher


class Excluded(Exception):
    """Fixed, content-free reason code."""


@dataclass(frozen=True)
class PayloadSnapshot:
    """Read-only encrypted blobs owned by one build, independent of the live store."""

    blobs: Mapping[tuple[str, str], EncryptedPayload] = field(repr=False)

    def get(self, tenant_id: str, reference: str, at: datetime) -> EncryptedPayload | None:
        blob = self.blobs.get((tenant_id, reference))
        return blob if blob is not None and blob.expires_at > at else None


@dataclass(frozen=True)
class BuiltExample:
    interaction_id: str
    tenant_id: str
    subject: str | None
    families: tuple[str, ...]
    started_at: datetime
    target_source: str
    language: str
    exact_hash: str
    text: str = field(repr=False)
    row: dict[str, object] = field(repr=False)
    benchmark_text: str | None = field(default=None, repr=False)


class Constructor:
    def __init__(
        self,
        payloads: PayloadReader,
        cipher: PayloadCipher,
        keyring: Keyring,
        redactor: PersistenceRedactor,
        sources: SourceResolver,
    ) -> None:
        self.payloads, self.cipher, self.keyring = payloads, cipher, keyring
        self.redactor, self.sources = redactor, sources

    def _read(
        self,
        example: Example,
        ref: str,
        field: str,
        digest: str,
        at: datetime,
        payloads: PayloadReader,
    ) -> str:
        interaction = example.interaction
        blob = payloads.get(interaction.tenant_id, ref, at)
        if blob is None:
            raise Excluded("missing_payload")
        try:
            value = self.cipher.decrypt(
                blob, interaction.tenant_id, interaction.interaction_id, field
            ).decode()
        except (StorageError, UnicodeError):
            raise Excluded("invalid_payload") from None
        purpose: Literal["input", "output"] = "input" if field == "messages" else "output"
        if self.keyring.content_hash(value, purpose=purpose) != digest:
            raise Excluded("payload_hash_mismatch")
        return value

    def build(
        self,
        example: Example,
        spec: DatasetSpecification,
        at: datetime,
        *,
        payloads: PayloadReader | None = None,
    ) -> BuiltExample:
        payloads = self.payloads if payloads is None else payloads
        interaction = example.interaction
        assert interaction.input.messages_ref and interaction.input.content_hash
        raw = self._read(
            example,
            interaction.input.messages_ref,
            "messages",
            interaction.input.content_hash,
            at,
            payloads,
        )
        try:
            messages = TypeAdapter(list[Message]).validate_json(raw)
        except Exception:
            raise Excluded("invalid_conversation") from None
        if not messages:
            raise Excluded("invalid_conversation")

        def redact(value: str) -> str:
            try:
                return self.redactor.redact_text(value, example.policy)[0]
            except Exception:
                raise Excluded("build_redaction_failed") from None

        conversation = [{"role": m.role, "content": redact(m.content)} for m in messages]
        sources: list[str] = []
        provenance: list[dict[str, object]] = []
        families: set[str] = set()
        if example.retrieval is not None:
            evidence = sorted(
                (c for c in example.retrieval.candidates if c.supplied_to_model),
                key=lambda c: c.context_position if c.context_position is not None else -1,
            )
            if [c.context_position for c in evidence] != list(range(len(evidence))):
                raise Excluded("invalid_source_order")
            for item in evidence:
                chunk = self.sources.resolve(
                    interaction, example.retrieval, item, example.policy.residency
                )
                if chunk is None:
                    raise Excluded("source_unavailable")
                if any(c in chunk.document_version for c in "<>\r\n\x00"):
                    raise Excluded("invalid_source_boundary")
                content = redact(chunk.content).replace("<<", "‹‹").replace(">>", "››")
                sources.append(
                    f"<<source {chunk.document_id}/{chunk.chunk_id} {chunk.document_version}>>"
                    f"{content}<</source>>"
                )
                families.add(chunk.document_family)
                provenance.append(
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "document_version": chunk.document_version,
                        "document_family": chunk.document_family,
                        "index_id": example.retrieval.index_id,
                        "index_version": example.retrieval.index_version,
                        "content_hash": item.content_hash,
                        "licence_class": chunk.licence_class,
                    }
                )
        target: str | None = None
        target_source: TargetSource = "production_output"
        target_feedback_id: str | None = None
        for preference in spec.target_preference_order:
            if preference == "correction":
                corrections = sorted(
                    (
                        f
                        for f in example.feedback
                        if f.label_type == "correction"
                        and f.training_authorised
                        and f.correction_ref
                        and f.content_hash
                        and not f.error_code
                    ),
                    key=lambda f: (f.created_at, f.feedback_id),
                    reverse=True,
                )
                if corrections:
                    chosen = corrections[0]
                    assert chosen.correction_ref and chosen.content_hash
                    target = self._read(
                        example,
                        chosen.correction_ref,
                        "correction",
                        chosen.content_hash,
                        at,
                        payloads,
                    )
                    target_feedback_id = chosen.feedback_id
            else:
                attempt = example.attempt
                positive = sorted(
                    (
                        f
                        for f in example.feedback
                        if f.label_type == "resolution" and f.value.score * 2 > f.value.max_score
                    ),
                    key=lambda f: (f.created_at, f.feedback_id),
                )
                if preference == "positive_resolution" and not positive:
                    continue
                if (
                    attempt
                    and attempt.validation
                    and attempt.validation.passed
                    and (
                        attempt.output_ref
                        and attempt.output_hash
                        and not attempt.error_code
                        and attempt.finish_reason == "stop"
                        and not attempt.tool_calls
                    )
                ):
                    target = self._read(
                        example, attempt.output_ref, "output", attempt.output_hash, at, payloads
                    )
                    if preference == "positive_resolution":
                        target_feedback_id = positive[-1].feedback_id
            if target is not None:
                target_source = preference
                break
        if target is None:
            raise Excluded("missing_eligible_target")
        target = redact(target)
        if not target.strip():
            raise Excluded("empty_target")
        inputs: dict[str, object] = {
            "system": "SYNTHETIC system instructions placeholder (local-1).",
            "policy": "SYNTHETIC policy instructions placeholder (local-1).",
            "messages": conversation,
            "sources": sources,
            "tool_results": [],
        }
        exact = self.keyring.dataset_hash(
            json.dumps([inputs, target], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        )
        text = "\n".join([*(m["content"] for m in conversation), *sources, target])
        row: dict[str, object] = {
            "input": inputs,
            "target": target,
            "target_source": target_source,
            "interaction_id": interaction.interaction_id,
            "tenant_id": interaction.tenant_id,
            "subject_id_pseudonymous": interaction.subject_id_pseudonymous,
            "document_families": sorted(families),
            "sources": provenance,
            "target_feedback_id": target_feedback_id,
            "target_attempt_id": interaction.final_attempt_id
            if target_source != "correction"
            else None,
            "input_hash": interaction.input.content_hash,
            "example_hash": exact,
            "eligibility_policy_version": example.policy.policy_version,
            "redaction_version": self.redactor.version,
            "labels": {
                "task": interaction.task.label,
                "language": interaction.task.language,
                "risk_tier": interaction.task.risk_tier,
                "difficulty": "unknown",
            },
        }
        return BuiltExample(
            interaction.interaction_id,
            interaction.tenant_id,
            interaction.subject_id_pseudonymous,
            tuple(sorted(families)),
            interaction.started_at,
            target_source,
            interaction.task.language,
            exact,
            text,
            row,
            "\n".join(m["content"] for m in conversation if m["role"] == "user"),
        )
