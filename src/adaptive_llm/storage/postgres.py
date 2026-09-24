"""PostgreSQL units of work. Tenant and control schemas form one atomic backup pair."""

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, local
from typing import Any, Literal, cast

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row

from adaptive_llm.contracts import Environment, now
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.migrations import CONTROL_MIGRATIONS, MIGRATIONS
from adaptive_llm.storage.sqlite import SQLiteMetadataStore, SQLitePayloadStore


class Row(dict[str, Any]):
    def __getitem__(self, key: str | int) -> Any:
        return list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


class Cursor:
    def __init__(self, cursor: psycopg.Cursor[dict[str, Any]]) -> None:
        self._cursor = cursor
        self.rowcount = cursor.rowcount

    def fetchone(self) -> Any:
        row = self._cursor.fetchone()
        return Row(row) if row is not None else None

    def fetchall(self) -> list[Row]:
        return [Row(row) for row in self._cursor.fetchall()]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.fetchall())


class PostgresConnection:
    """Bound parameters only; adapt the repository's small SQL dialect at one boundary."""

    def __init__(self, raw: psycopg.Connection[dict[str, Any]]) -> None:
        self.raw = raw

    @property
    def in_transaction(self) -> bool:
        return self.raw.info.transaction_status != TransactionStatus.IDLE

    def execute(self, query: str, parameters: Sequence[Any] = ()) -> Cursor:
        # Only source-controlled statements reach this boundary. No bindings enter SQL text.
        if "INSERT OR IGNORE INTO" in query:
            query = (
                query.replace("INSERT OR IGNORE INTO", "INSERT INTO") + " ON CONFLICT DO NOTHING"
            )
        query = re.sub(
            r"json_extract\((\w+(?:\.\w+)?), '\$\.([\w.]+)'\)",
            lambda m: f"({m[1]}::jsonb #>> '{{{m[2].replace('.', ',')}}}')",
            query,
        )
        query = query.replace("?", "%s")
        try:
            return Cursor(self.raw.execute(query, parameters))
        except psycopg.Error:
            raise StorageError("database_operation_failed") from None


class PostgresDatabase:
    def __init__(
        self,
        database_url: str,
        environment: Environment = "local",
        *,
        role: Literal["tenant", "control"] = "tenant",
        migrate_on_startup: bool = True,
    ) -> None:
        if environment not in {"local", "development", "staging", "production"}:
            raise ValueError("invalid_environment")
        self.database_url = database_url
        self.schema = f"adaptive_{environment}_{role}"
        self.lock = RLock()
        self._claim_connections: list[ClaimConnections] = []
        self.connection = self.connect()
        try:
            if migrate_on_startup:
                self.migrate(CONTROL_MIGRATIONS if role == "control" else MIGRATIONS)
        except BaseException:
            self.close()
            raise

    def connect(self) -> PostgresConnection:
        try:
            raw = psycopg.connect(
                self.database_url, autocommit=True, row_factory=dict_row, connect_timeout=3
            )
            raw.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            return PostgresConnection(raw)
        except psycopg.Error:
            raise StorageError("database_unavailable") from None

    def migrate(self, directory: Path) -> None:
        raw = self.connection.raw
        try:
            with raw.transaction():
                raw.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self.schema,))
                raw.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
                )
                raw.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations "
                    "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
                )
                versions = {
                    r["version"] for r in raw.execute("SELECT version FROM schema_migrations")
                }
                overlay = (
                    MIGRATIONS / "postgres" / ("control" if directory == CONTROL_MIGRATIONS else "")
                )
                sources = list(directory.glob("[0-9]*_*.sql"))
                files = {int(p.name.split("_")[0]): p for p in sources}
                if len(files) != len(sources):
                    raise StorageError("duplicate_migration_version")
                if directory in (MIGRATIONS, CONTROL_MIGRATIONS):
                    files.update(
                        {int(p.name.split("_")[0]): p for p in overlay.glob("[0-9]*_*.sql")}
                    )
                for version, path in sorted(files.items()):
                    if version in versions:
                        continue
                    raw.execute(path.read_text())
                    raw.execute(
                        "INSERT INTO schema_migrations VALUES (%s, %s)",
                        (version, now().isoformat()),
                    )
        except Exception:
            raise StorageError("migration_failed") from None

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[None]:
        try:
            with self.lock, self.connection.raw.transaction():
                if read_only:
                    self.connection.raw.execute(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                    )
                else:
                    # Preserve read/check/write invariants (tombstones, replay cap, promotions).
                    # Dispatch and worker claims use independent row locks, not this fence.
                    self.connection.raw.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))", (self.schema,)
                    )
                yield
        except psycopg.Error:
            raise StorageError("database_transaction_failed") from None

    def writable(self, tenant_id: str, interaction_id: str) -> None:
        if not self.connection.in_transaction:
            raise StorageError("storage_transaction_required")
        if self.connection.execute(
            "SELECT 1 FROM deletion_tombstones WHERE tenant_id=? AND interaction_id=?",
            (tenant_id, interaction_id),
        ).fetchone():
            raise StorageError("interaction_deleted")
        row = self.connection.execute(
            "SELECT state FROM interactions WHERE tenant_id=? AND record_id=?",
            (tenant_id, interaction_id),
        ).fetchone()
        if row and row["state"] != "active":
            raise StorageError("interaction_inactive")

    def close(self) -> None:
        with self.lock:
            for connections in self._claim_connections:
                connections.close()
            self._claim_connections.clear()
            self.connection.raw.close()

    def claim_connections(self) -> "ClaimConnections":
        with self.lock:
            connections = ClaimConnections(self)
            self._claim_connections.append(connections)
            return connections


class ClaimConnections:
    """One lazy connection per store/thread, closed by the store or its owning database."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database
        self._local = local()
        self._lock = RLock()
        self._connections: list[PostgresConnection] = []
        self._closed = False

    def get(self) -> PostgresConnection:
        with self._lock:
            if self._closed:
                raise StorageError("claim_connections_closed")
            connection = cast(PostgresConnection | None, getattr(self._local, "connection", None))
            if connection is None or connection.raw.closed:
                connection = self.database.connect()
                self._connections = [c for c in self._connections if not c.raw.closed]
                self._connections.append(connection)
                self._local.connection = connection
            if connection.in_transaction:
                raise StorageError("claim_already_active")
            return connection

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for connection in self._connections:
                connection.raw.close()
            self._connections.clear()


class PostgresMetadataStore(SQLiteMetadataStore):
    """Shared tenant predicates and privacy transactions, PostgreSQL SQL execution."""


class PostgresPayloadStore(SQLitePayloadStore):
    """Shared authenticated ciphertext records, stored as PostgreSQL bytea."""
