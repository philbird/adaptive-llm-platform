"""Tenant-scoped storage boundaries; implementations share an atomic unit of work."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeVar

from adaptive_llm.contracts import (
    DatasetManifest,
    DeletionRequest,
    Feedback,
    GenerationAttempt,
    Interaction,
    RetrievalRun,
    RouteDecision,
    Started,
)

StoredRecord = Interaction | Started | RetrievalRun | RouteDecision | GenerationAttempt | Feedback
RecordT = TypeVar("RecordT", bound=StoredRecord)
State = Literal["active", "expired", "deleted"]


@dataclass(frozen=True)
class ReplayRecord:
    tenant_id: str
    application_id: str
    request_id: str
    interaction_id: str
    fingerprint: str
    response_ref: str
    reserved_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class EncryptedPayload:
    reference: str
    tenant_id: str
    interaction_id: str
    field: str
    nonce: bytes
    ciphertext: bytes
    key_version: str
    expires_at: datetime


class MetadataStore(Protocol):
    def transaction(self) -> AbstractContextManager[None]: ...

    def read_transaction(self) -> AbstractContextManager[None]: ...

    def put(self, tenant_id: str, record: StoredRecord, expires_at: datetime) -> None: ...

    def get(self, tenant_id: str, kind: type[RecordT], record_id: str) -> RecordT | None: ...

    def state(self, tenant_id: str, interaction_id: str) -> State | None: ...

    def for_subject(self, tenant_id: str, pseudonym: str) -> list[Interaction]: ...

    def expired(self, tenant_id: str, at: datetime) -> list[Interaction]: ...

    def clear_refs(self, tenant_id: str, interaction_id: str, state: State) -> None: ...

    def tombstone(self, tenant_id: str, interaction_id: str, deletion: DeletionRequest) -> None: ...

    def get_tombstone(self, tenant_id: str, interaction_id: str) -> DeletionRequest | None: ...

    def tombstone_subject(self, tenant_id: str, deletion: DeletionRequest) -> None: ...

    def get_subject_tombstone(self, tenant_id: str, pseudonym: str) -> DeletionRequest | None: ...

    def put_replay(self, replay: ReplayRecord, capacity: int) -> None: ...

    def get_replay(
        self, tenant_id: str, application_id: str, request_id: str, at: datetime
    ) -> ReplayRecord | None: ...

    def delete_replay(self, tenant_id: str, application_id: str, request_id: str) -> None: ...

    def expire_replays(self, tenant_id: str, at: datetime) -> None: ...

    def expires_at(self, tenant_id: str, interaction_id: str) -> datetime | None: ...

    def append_feedback(
        self, tenant_id: str, interaction: Interaction, feedback_id: str
    ) -> None: ...

    def in_window(self, tenant_id: str, start: datetime, end: datetime) -> list[Interaction]: ...

    def put_manifest(self, manifest: DatasetManifest) -> None: ...

    def get_manifest(self, dataset_id: str, version: str) -> DatasetManifest | None: ...

    def approve_manifest(self, manifest: DatasetManifest) -> None: ...


class PayloadReader(Protocol):
    def get(self, tenant_id: str, reference: str, at: datetime) -> EncryptedPayload | None: ...


class PayloadStore(PayloadReader, Protocol):
    def put(self, payload: EncryptedPayload) -> None: ...

    def delete_interaction(self, tenant_id: str, interaction_id: str) -> None: ...


class StorageError(Exception):
    """Fixed codes only: never expose SQL bindings or content."""
