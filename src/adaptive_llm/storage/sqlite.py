"""Local SQLite stores. All data access binds an authenticated tenant explicitly."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, cast

from adaptive_llm.contracts import (
    DatasetManifest,
    DeletionRequest,
    Environment,
    Feedback,
    GenerationAttempt,
    Interaction,
    RetrievalRun,
    RouteDecision,
    Started,
)
from adaptive_llm.storage import (
    EncryptedPayload,
    RecordT,
    ReplayRecord,
    State,
    StorageError,
    StoredRecord,
)
from adaptive_llm.storage.database import Database
from adaptive_llm.storage.migrations import CONTROL_MIGRATIONS, MIGRATIONS, migrate

TABLES: dict[type[StoredRecord], tuple[str, str]] = {
    Interaction: ("interactions", "interaction_id"),
    Started: ("started", "interaction_id"),
    RetrievalRun: ("retrieval_runs", "retrieval_run_id"),
    RouteDecision: ("route_decisions", "route_decision_id"),
    GenerationAttempt: ("attempts", "attempt_id"),
    Feedback: ("feedback", "feedback_id"),
}


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise StorageError("naive_storage_timestamp")
    return value.astimezone(UTC).isoformat()


class SQLiteDatabase:
    def __init__(
        self,
        data_dir: Path,
        environment: Environment = "local",
        *,
        migrate_on_startup: bool = True,
        migrations: Path = MIGRATIONS,
        in_memory: bool = False,
    ) -> None:
        if environment not in ("local", "development", "staging", "production"):
            raise ValueError("invalid_environment")
        if not in_memory:
            data_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(":memory:") if in_memory else data_dir / f"{environment}.sqlite3"
        self.lock = RLock()
        self.connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA secure_delete = ON")
        # DELETE journals disappear after commit; no retained WAL copy of deleted blobs.
        self.connection.execute("PRAGMA journal_mode = DELETE")
        if migrate_on_startup:
            try:
                migrate(self.connection, migrations)
            except BaseException:
                self.connection.close()
                raise

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[None]:
        with self.lock:
            try:
                self.connection.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")
                yield
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def writable(self, tenant_id: str, interaction_id: str) -> None:
        if not self.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        if self.connection.execute(
            "SELECT 1 FROM deletion_tombstones WHERE tenant_id = ? AND interaction_id = ?",
            (tenant_id, interaction_id),
        ).fetchone():
            raise StorageError("interaction_deleted")
        row = self.connection.execute(
            "SELECT state FROM interactions WHERE tenant_id = ? AND record_id = ?",
            (tenant_id, interaction_id),
        ).fetchone()
        if row and row["state"] != "active":
            raise StorageError("interaction_inactive")


def control_database(
    data_dir: Path, environment: Environment = "local", *, migrate_on_startup: bool = True
) -> SQLiteDatabase:
    legacy, destination = data_dir / "evaluations" / "control", data_dir / "control"
    if legacy.exists():
        if destination.exists():
            raise StorageError("ambiguous_control_database")
        legacy.rename(destination)
    return SQLiteDatabase(
        destination,
        environment,
        migrate_on_startup=migrate_on_startup,
        migrations=CONTROL_MIGRATIONS,
    )


class SQLiteMetadataStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.database.transaction():
            yield

    @contextmanager
    def read_transaction(self) -> Iterator[None]:
        with self.database.transaction(read_only=True):
            yield

    def put(self, tenant_id: str, record: StoredRecord, expires_at: datetime) -> None:
        with self.database.lock:
            self.database.writable(tenant_id, record.interaction_id)
            if isinstance(record, Interaction) and record.tenant_id != tenant_id:
                raise StorageError("tenant_mismatch")
            if (
                isinstance(record, Interaction)
                and record.subject_id_pseudonymous is not None
                and self.get_subject_tombstone(tenant_id, record.subject_id_pseudonymous)
                is not None
            ):
                raise StorageError("subject_deleted")
            table, id_field = TABLES[type(record)]
            columns = "tenant_id, record_id, interaction_id, data, expires_at"
            values = [
                tenant_id,
                getattr(record, id_field),
                record.interaction_id,
                record.model_dump_json(),
                timestamp(expires_at),
            ]
            if isinstance(record, Interaction):
                columns += ", subject_id, started_at"
                values.extend([record.subject_id_pseudonymous, timestamp(record.started_at)])
            placeholders = ", ".join("?" for _ in values)
            self.database.connection.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values
            )

    def get(self, tenant_id: str, kind: type[RecordT], record_id: str) -> RecordT | None:
        table, _ = TABLES[cast(type[StoredRecord], kind)]
        with self.database.lock:
            row = self.database.connection.execute(
                f"SELECT data FROM {table} WHERE tenant_id = ? AND record_id = ?",
                (tenant_id, record_id),
            ).fetchone()
        return cast(RecordT, kind.model_validate_json(row["data"])) if row else None

    def state(self, tenant_id: str, interaction_id: str) -> State | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT state FROM interactions WHERE tenant_id = ? AND record_id = ?",
                (tenant_id, interaction_id),
            ).fetchone()
        return cast(State, row["state"]) if row else None

    def add_shadow_cost(self, tenant_id: str, interaction_id: str, cost_micros: int) -> None:
        self.database.writable(tenant_id, interaction_id)
        record = self.get(tenant_id, Interaction, interaction_id)
        if record is not None:
            updated = record.model_copy(
                update={"shadow_cost_micros": record.shadow_cost_micros + cost_micros}
            )
            self.database.connection.execute(
                "UPDATE interactions SET data=? WHERE tenant_id=? AND record_id=?",
                (updated.model_dump_json(), tenant_id, interaction_id),
            )

    def for_subject(self, tenant_id: str, pseudonym: str) -> list[Interaction]:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT data FROM interactions WHERE tenant_id = ? AND subject_id = ? "
                "AND state != 'deleted'",
                (tenant_id, pseudonym),
            ).fetchall()
        return [Interaction.model_validate_json(row["data"]) for row in rows]

    def expired(self, tenant_id: str, at: datetime) -> list[Interaction]:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT data FROM interactions WHERE tenant_id = ? AND expires_at <= ? "
                "AND state = 'active'",
                (tenant_id, timestamp(at)),
            ).fetchall()
        return [Interaction.model_validate_json(row["data"]) for row in rows]

    def clear_refs(self, tenant_id: str, interaction_id: str, state: State) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")

        # Walk JSON solely to remove reference fields, retaining the original contracts.
        def clear(value: object) -> object:
            if isinstance(value, dict):
                return {k: None if k.endswith("_ref") else clear(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clear(item) for item in value]
            return value

        for table, _ in TABLES.values():
            rows = self.database.connection.execute(
                f"SELECT record_id, data FROM {table} WHERE tenant_id = ? AND interaction_id = ?",
                (tenant_id, interaction_id),
            ).fetchall()
            for row in rows:
                self.database.connection.execute(
                    f"UPDATE {table} SET data = ?, state = ? WHERE tenant_id = ? AND record_id = ?",
                    (
                        json.dumps(clear(json.loads(row["data"]))),
                        state,
                        tenant_id,
                        row["record_id"],
                    ),
                )
        self.database.connection.execute(
            "DELETE FROM replay_entries WHERE tenant_id = ? AND interaction_id = ?",
            (tenant_id, interaction_id),
        )

    def tombstone(self, tenant_id: str, interaction_id: str, deletion: DeletionRequest) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        self.database.connection.execute(
            "INSERT OR IGNORE INTO deletion_tombstones VALUES (?, ?, ?)",
            (tenant_id, interaction_id, deletion.model_dump_json()),
        )

    def get_tombstone(self, tenant_id: str, interaction_id: str) -> DeletionRequest | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT data FROM deletion_tombstones WHERE tenant_id = ? AND interaction_id = ?",
                (tenant_id, interaction_id),
            ).fetchone()
        return DeletionRequest.model_validate_json(row["data"]) if row else None

    def tombstone_subject(self, tenant_id: str, deletion: DeletionRequest) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        if deletion.scope != "subject":
            raise StorageError("invalid_deletion_scope")
        self.database.connection.execute(
            "INSERT OR IGNORE INTO subject_tombstones VALUES (?, ?, ?)",
            (tenant_id, deletion.target_id, deletion.model_dump_json()),
        )

    def get_subject_tombstone(self, tenant_id: str, pseudonym: str) -> DeletionRequest | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT data FROM subject_tombstones WHERE tenant_id = ? AND subject_id = ?",
                (tenant_id, pseudonym),
            ).fetchone()
        return DeletionRequest.model_validate_json(row["data"]) if row else None

    def put_replay(self, replay: ReplayRecord, capacity: int) -> None:
        if capacity < 1:
            raise StorageError("invalid_replay_capacity")
        self.database.writable(replay.tenant_id, replay.interaction_id)
        self.database.connection.execute(
            "INSERT INTO replay_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                replay.tenant_id,
                replay.application_id,
                replay.request_id,
                replay.interaction_id,
                replay.fingerprint,
                replay.response_ref,
                timestamp(replay.reserved_at),
                timestamp(replay.expires_at),
            ),
        )
        # The tenant index supplies reservation order without loading replay rows into Python.
        # Delete entries first (their FK references payloads), then just their replay blobs.
        evicted = self.database.connection.execute(
            "DELETE FROM replay_entries WHERE tenant_id = ? AND (application_id, request_id) IN ("
            "SELECT application_id, request_id FROM replay_entries WHERE tenant_id = ? "
            "ORDER BY reserved_at DESC, application_id DESC, request_id DESC LIMIT "
            "(SELECT count(*) FROM replay_entries WHERE tenant_id = ?) OFFSET ?) "
            "RETURNING response_ref",
            (replay.tenant_id, replay.tenant_id, replay.tenant_id, capacity),
        ).fetchall()
        for row in evicted:
            self.database.connection.execute(
                "DELETE FROM payloads WHERE tenant_id = ? AND reference = ?",
                (replay.tenant_id, row["response_ref"]),
            )

    @staticmethod
    def _replay(row: Any) -> ReplayRecord:
        return ReplayRecord(
            tenant_id=row["tenant_id"],
            application_id=row["application_id"],
            request_id=row["request_id"],
            interaction_id=row["interaction_id"],
            fingerprint=row["fingerprint"],
            response_ref=row["response_ref"],
            reserved_at=datetime.fromisoformat(row["reserved_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )

    def get_replay(
        self, tenant_id: str, application_id: str, request_id: str, at: datetime
    ) -> ReplayRecord | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT * FROM replay_entries WHERE tenant_id = ? AND application_id = ? "
                "AND request_id = ? AND expires_at > ?",
                (tenant_id, application_id, request_id, timestamp(at)),
            ).fetchone()
        return self._replay(row) if row else None

    def delete_replay(self, tenant_id: str, application_id: str, request_id: str) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        row = self.database.connection.execute(
            "SELECT response_ref FROM replay_entries WHERE tenant_id = ? "
            "AND application_id = ? AND request_id = ?",
            (tenant_id, application_id, request_id),
        ).fetchone()
        self.database.connection.execute(
            "DELETE FROM replay_entries WHERE tenant_id = ? "
            "AND application_id = ? AND request_id = ?",
            (tenant_id, application_id, request_id),
        )
        if row:
            self.database.connection.execute(
                "DELETE FROM payloads WHERE tenant_id = ? AND reference = ?",
                (tenant_id, row["response_ref"]),
            )

    def expire_replays(self, tenant_id: str, at: datetime) -> None:
        rows = self.database.connection.execute(
            "SELECT application_id, request_id FROM replay_entries "
            "WHERE tenant_id = ? AND expires_at <= ?",
            (tenant_id, timestamp(at)),
        ).fetchall()
        for row in rows:
            self.delete_replay(tenant_id, row["application_id"], row["request_id"])

    def expires_at(self, tenant_id: str, interaction_id: str) -> datetime | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT expires_at FROM interactions WHERE tenant_id = ? AND record_id = ?",
                (tenant_id, interaction_id),
            ).fetchone()
        return datetime.fromisoformat(row["expires_at"]) if row else None

    def append_feedback(self, tenant_id: str, interaction: Interaction, feedback_id: str) -> None:
        self.database.writable(tenant_id, interaction.interaction_id)
        updated = interaction.model_copy(
            update={"feedback_ids": [*interaction.feedback_ids, feedback_id]}
        )
        self.database.connection.execute(
            "UPDATE interactions SET data = ? WHERE tenant_id = ? AND record_id = ?",
            (updated.model_dump_json(), tenant_id, interaction.interaction_id),
        )

    def in_window(self, tenant_id: str, start: datetime, end: datetime) -> list[Interaction]:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT data FROM interactions WHERE tenant_id = ? "
                "AND started_at >= ? AND started_at < ? ORDER BY started_at, record_id",
                (tenant_id, timestamp(start), timestamp(end)),
            ).fetchall()
        return [Interaction.model_validate_json(row["data"]) for row in rows]

    def put_manifest(self, manifest: DatasetManifest) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        self.database.connection.execute(
            "INSERT INTO dataset_manifests VALUES (?, ?, ?)",
            (manifest.dataset_id, manifest.version, manifest.model_dump_json()),
        )

    def approve_manifest(self, manifest: DatasetManifest) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        self.database.connection.execute(
            "UPDATE dataset_manifests SET data=? WHERE dataset_id=? AND version=?",
            (manifest.model_dump_json(), manifest.dataset_id, manifest.version),
        )

    def get_manifest(self, dataset_id: str, version: str) -> DatasetManifest | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT data FROM dataset_manifests WHERE dataset_id = ? AND version = ?",
                (dataset_id, version),
            ).fetchone()
        return DatasetManifest.model_validate_json(row["data"]) if row else None


class SQLitePayloadStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def put(self, payload: EncryptedPayload) -> None:
        self.database.writable(payload.tenant_id, payload.interaction_id)
        self.database.connection.execute(
            "INSERT INTO payloads VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                payload.tenant_id,
                payload.reference,
                payload.interaction_id,
                payload.field,
                payload.nonce,
                payload.ciphertext,
                payload.key_version,
                timestamp(payload.expires_at),
            ),
        )

    def get(self, tenant_id: str, reference: str, at: datetime) -> EncryptedPayload | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT * FROM payloads WHERE tenant_id = ? AND reference = ? AND expires_at > ?",
                (tenant_id, reference, timestamp(at)),
            ).fetchone()
        if row is None:
            return None
        return EncryptedPayload(
            reference=row["reference"],
            tenant_id=row["tenant_id"],
            interaction_id=row["interaction_id"],
            field=row["field"],
            nonce=row["nonce"],
            ciphertext=row["ciphertext"],
            key_version=row["key_version"],
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )

    def delete_interaction(self, tenant_id: str, interaction_id: str) -> None:
        if not self.database.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        self.database.connection.execute(
            "DELETE FROM payloads WHERE tenant_id = ? AND interaction_id = ?",
            (tenant_id, interaction_id),
        )
