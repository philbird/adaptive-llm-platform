"""PostgreSQL control repositories and independent transactional dispatch/worker claims."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Literal, cast

from adaptive_llm.contracts import TrainingJob
from adaptive_llm.distillation.benchmark import SQLiteBenchmarkStore
from adaptive_llm.evaluation.storage import EvaluationStore, SQLiteEvaluationStore
from adaptive_llm.events.outbox import OutboxRow, OutboxStore
from adaptive_llm.gateway.identity import Identity, Keyring
from adaptive_llm.registry.sqlite import SQLiteModelRegistry
from adaptive_llm.routing.control import SQLiteRoutePolicies
from adaptive_llm.storage.database import Database
from adaptive_llm.storage.outbox import SQLiteOutboxStore
from adaptive_llm.storage.postgres import PostgresConnection, PostgresDatabase
from adaptive_llm.storage.sqlite import timestamp


class PostgresOutboxStore(SQLiteOutboxStore):
    database: PostgresDatabase

    def __init__(self, database: PostgresDatabase) -> None:
        super().__init__(database)
        self._claims = local()
        self._connections = database.claim_connections()

    def close(self) -> None:
        self._connections.close()

    @property
    def _claimed(self) -> PostgresConnection | None:
        return cast(PostgresConnection | None, getattr(self._claims, "connection", None))

    @contextmanager
    def claim(self, at: datetime) -> Iterator[OutboxRow | None]:
        connection = self._connections.get()
        previous = self._claimed
        try:
            with connection.raw.transaction():
                row = connection.execute(
                    "SELECT * FROM outbox o WHERE state='pending' AND next_attempt_at<=? "
                    "AND NOT EXISTS (SELECT 1 FROM outbox earlier "
                    "WHERE earlier.tenant_id=o.tenant_id "
                    "AND earlier.interaction_id=o.interaction_id AND earlier.state='pending' "
                    "AND earlier.sequence<o.sequence) ORDER BY next_attempt_at, sequence "
                    "LIMIT 1 FOR UPDATE OF o SKIP LOCKED",
                    (timestamp(at),),
                ).fetchone()
                self._claims.connection = connection
                yield (
                    OutboxRow(
                        row["event_id"],
                        row["tenant_id"],
                        row["trace_id"],
                        row["interaction_id"],
                        row["event_type"],
                        row["envelope"],
                        row["attempts"],
                    )
                    if row
                    else None
                )
        finally:
            self._claims.connection = previous

    def finish(
        self,
        event_id: str,
        state: Literal["pending", "delivered", "dead"],
        attempts: int,
        next_attempt_at: datetime,
        error: Literal["sink_unavailable", "invalid_event"] | None,
    ) -> None:
        if self._claimed is None:
            super().finish(event_id, state, attempts, next_attempt_at, error)
        else:
            self._claimed.execute(
                "UPDATE outbox SET state=?, attempts=?, next_attempt_at=?, "
                "last_error_code=? WHERE event_id=? AND state='pending'",
                (state, attempts, timestamp(next_attempt_at), error, event_id),
            )


class PostgresModelRegistry(SQLiteModelRegistry):
    def __init__(
        self,
        database: Database,
        outbox: OutboxStore,
        keyring: Keyring,
        evaluations: EvaluationStore,
        data_dir: Path,
        pending_limit: int,
    ) -> None:
        assert isinstance(database, PostgresDatabase)
        super().__init__(database, outbox, keyring, evaluations, data_dir, pending_limit)
        self._connections = database.claim_connections()

    def close(self) -> None:
        self._connections.close()

    @contextmanager
    def worker_claim(
        self, architectures: tuple[str, ...]
    ) -> Iterator[list[tuple[TrainingJob, Identity]]]:
        connection = self._connections.get()
        with connection.raw.transaction():
            # Lock submitters so progress and cancellation updates remain live.
            row = connection.execute(
                "SELECT j.data, s.identity FROM training_jobs j JOIN training_submitters s "
                "USING(job_id) WHERE json_extract(j.data, '$.state') IN ('queued','running') "
                "AND json_extract(j.data, '$.trainer_architecture')=ANY(?) "
                "ORDER BY json_extract(j.data, '$.created_at'), j.job_id "
                "LIMIT 1 FOR UPDATE OF s SKIP LOCKED",
                (list(architectures),),
            ).fetchone()
            if row is None:
                yield []
            else:
                trusted = json.loads(row["identity"])
                for key in ("application_ids", "dataset_tenants", "capabilities"):
                    trusted[key] = frozenset(trusted.get(key, []))
                yield [(TrainingJob.model_validate_json(row["data"]), Identity(**trusted))]


class PostgresEvaluationStore(SQLiteEvaluationStore):
    """Atomic report/baseline publication using the control schema."""


class PostgresBenchmarkStore(SQLiteBenchmarkStore):
    """Authenticated benchmark reports using the control schema."""


class PostgresRoutePolicies(SQLiteRoutePolicies):
    """Transactional route policies, audit history and outcome observations."""
