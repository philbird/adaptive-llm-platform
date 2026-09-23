"""Ordered SQL migrations, including their version records, commit as one transaction."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from adaptive_llm.storage import StorageError

MIGRATIONS = Path(__file__).resolve().parents[3] / "migrations"
CONTROL_MIGRATIONS = MIGRATIONS / "control"


def migrate(connection: sqlite3.Connection, directory: Path = MIGRATIONS) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        # Upgrade the migration ledger created by the first slice-1b implementation too.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(schema_migrations)")}
        if "tenant_id" in columns:
            connection.execute("ALTER TABLE schema_migrations DROP COLUMN tenant_id")
        versions = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
        files = sorted(directory.glob("[0-9]*_*.sql"), key=lambda p: int(p.name.split("_")[0]))
        numbered = [(int(path.name.split("_")[0]), path) for path in files]
        if len({version for version, _ in numbered}) != len(numbered):
            raise StorageError("duplicate_migration_version")
        for version, path in numbered:
            if version in versions:
                continue
            statement = ""
            for line in path.read_text().splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    connection.execute(statement)
                    statement = ""
            if statement.strip():
                raise StorageError("incomplete_migration")
            connection.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(UTC).isoformat()),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise StorageError("migration_failed") from None
