"""SQLite outbox operations; enqueue participates in the caller's transaction."""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from adaptive_llm.contracts import Event
from adaptive_llm.events.outbox import OutboxRow, OutboxStats, interaction_key
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.sqlite import SQLiteDatabase, timestamp

OPTIONAL_EVENTS = frozenset(
    {
        "interaction.started.v1",
        "retrieval.completed.v1",
        "route.decided.v1",
    }
)


class SQLiteOutboxStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def enqueue(self, events: Sequence[Event], pending_limit: int) -> int:
        with self.database.lock:
            connection = self.database.connection
            if not connection.in_transaction:
                raise StorageError("storage_transaction_required")
            pending = connection.execute(
                "SELECT count(*) FROM outbox WHERE state = 'pending'"
            ).fetchone()[0]
            dropped = 0
            for event in events:
                if connection.execute(
                    "SELECT 1 FROM outbox WHERE event_id = ?", (event.event_id,)
                ).fetchone():
                    continue
                if pending >= pending_limit and event.event_type in OPTIONAL_EVENTS:
                    dropped += 1
                    continue
                connection.execute(
                    "INSERT INTO outbox (event_id, tenant_id, trace_id, interaction_id, "
                    "event_type, envelope, next_attempt_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.tenant_id,
                        event.trace_id,
                        interaction_key(event),
                        event.event_type,
                        event.model_dump_json(),
                        timestamp(event.occurred_at),
                    ),
                )
                pending += 1
            return dropped

    def next_due(self, at: datetime) -> OutboxRow | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT * FROM outbox AS o WHERE state = 'pending' AND next_attempt_at <= ? "
                "AND NOT EXISTS (SELECT 1 FROM outbox AS earlier "
                "WHERE earlier.tenant_id = o.tenant_id "
                "AND earlier.interaction_id = o.interaction_id AND earlier.state = 'pending' "
                "AND earlier.sequence < o.sequence) ORDER BY next_attempt_at, sequence LIMIT 1",
                (timestamp(at),),
            ).fetchone()
        return (
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

    def finish(
        self,
        event_id: str,
        state: Literal["pending", "delivered", "dead"],
        attempts: int,
        next_attempt_at: datetime,
        error: Literal["sink_unavailable", "invalid_event"] | None,
    ) -> None:
        with self.database.transaction():
            self.database.connection.execute(
                "UPDATE outbox SET state = ?, attempts = ?, next_attempt_at = ?, "
                "last_error_code = ? WHERE event_id = ? AND state = 'pending'",
                (state, attempts, timestamp(next_attempt_at), error, event_id),
            )

    def stats(self, at: datetime) -> OutboxStats:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT state, count(*) AS count, min(next_attempt_at) AS oldest "
                "FROM outbox WHERE state IN ('pending', 'dead') GROUP BY state"
            ).fetchall()
        counts = {row["state"]: row["count"] for row in rows}
        oldest = next((row["oldest"] for row in rows if row["state"] == "pending"), None)
        lag = max(0.0, (at - datetime.fromisoformat(oldest)).total_seconds()) if oldest else 0.0
        return OutboxStats(counts.get("pending", 0), counts.get("dead", 0), lag)

    def dead_letters(self, tenant_id: str) -> list[tuple[str, str, str]]:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT event_id, event_type, last_error_code FROM outbox "
                "WHERE tenant_id = ? AND state = 'dead' ORDER BY sequence",
                (tenant_id,),
            ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def redeliver(self, event_id: str, at: datetime) -> bool:
        with self.database.transaction():
            return (
                self.database.connection.execute(
                    "UPDATE outbox SET state = 'pending', attempts = 0, next_attempt_at = ?, "
                    "last_error_code = NULL WHERE event_id = ? AND state = 'dead'",
                    (timestamp(at), event_id),
                ).rowcount
                == 1
            )
