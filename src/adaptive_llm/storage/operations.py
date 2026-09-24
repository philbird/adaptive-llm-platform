"""Local backup, offline restore and batched tenant payload key rotation."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path

from adaptive_llm.contracts import now, uid
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.database import Database
from adaptive_llm.storage.migrations import CONTROL_MIGRATIONS, MIGRATIONS, migrate
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLitePayloadStore, timestamp


def rotate_key(
    database: Database, tenant_id: str, cipher: PayloadCipher, *, batch_size: int = 100
) -> int:
    if batch_size < 1:
        raise ValueError("invalid_rotation_batch_size")
    payloads = SQLitePayloadStore(database)
    with database.lock:
        ceiling = database.connection.execute(
            "SELECT coalesce(max(reference), '') FROM payloads WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()[0]
    cursor, changed = "", 0
    while True:
        with database.transaction():
            at = now()
            rows = database.connection.execute(
                "SELECT reference FROM payloads WHERE tenant_id = ? "
                "AND reference > ? AND reference <= ? AND expires_at > ? AND key_version != ? "
                "ORDER BY reference LIMIT ?",
                (tenant_id, cursor, ceiling, timestamp(at), cipher.key_version, batch_size),
            ).fetchall()
            if not rows:
                return changed
            for row in rows:
                blob = payloads.get(tenant_id, row["reference"], at)
                assert blob is not None
                plaintext = cipher.decrypt(blob, tenant_id, blob.interaction_id, blob.field)
                replacement = cipher.encrypt(
                    plaintext, tenant_id, blob.interaction_id, blob.field, blob.expires_at
                )
                database.connection.execute(
                    "UPDATE payloads SET nonce = ?, ciphertext = ?, key_version = ? "
                    "WHERE tenant_id = ? AND reference = ?",
                    (
                        replacement.nonce,
                        replacement.ciphertext,
                        replacement.key_version,
                        tenant_id,
                        blob.reference,
                    ),
                )
            cursor = rows[-1]["reference"]
        changed += len(rows)


def backup(database: SQLiteDatabase, output: Path) -> None:
    if output.resolve() == database.path.resolve() or output.exists():
        raise StorageError("backup_destination_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(output)
    try:
        with database.lock:
            database.connection.backup(target)
    except Exception:
        output.unlink(missing_ok=True)
        raise StorageError("backup_failed") from None
    finally:
        target.close()


def restore(source: Path, destination: Path, *, force: bool = False) -> None:
    """Offline replacement: validate a temporary copy, then atomically publish it."""
    if source.resolve() == destination.resolve():
        raise StorageError("invalid_restore_source")
    if destination.exists() and not force:
        raise StorageError("restore_destination_exists")
    if not source.is_file():
        raise StorageError("backup_not_found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".restore-{uid()}.sqlite3")
    try:
        original = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
        target = sqlite3.connect(temporary, isolation_level=None)
        try:
            original.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise StorageError("invalid_backup")
            if not target.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'schema_migrations'"
            ).fetchone():
                raise StorageError("invalid_backup")
            migrate(target)
        finally:
            original.close()
            target.close()
        if force:
            os.replace(temporary, destination)
        else:
            # The link fails if another process created the destination during validation.
            os.link(temporary, destination)
    except Exception:
        raise StorageError("restore_failed") from None
    finally:
        temporary.unlink(missing_ok=True)


def backup_pair(tenant: SQLiteDatabase, control: SQLiteDatabase, output: Path) -> None:
    """Hold both local writer locks; publish the tenant-first snapshot pair by one rename."""
    if output.exists():
        raise StorageError("backup_destination_exists")
    staging = output.with_name(f".{output.name}.{uid()}.backup")
    version = uid()
    try:
        staging.mkdir(parents=True, mode=0o700)
        hashes: dict[str, str] = {}
        # BEGIN IMMEDIATE on separate connections fences writers in other processes too.
        # Read connections make online backups without backing up an active write transaction.
        with tenant.transaction(), control.transaction():
            for role, database in (("tenant", tenant), ("control", control)):
                path = staging / f"{role}.sqlite3"
                source = sqlite3.connect(database.path)
                target = sqlite3.connect(path)
                try:
                    source.backup(target)
                    target.execute("CREATE TABLE backup_pair (version TEXT, role TEXT)")
                    target.execute("INSERT INTO backup_pair VALUES (?, ?)", (version, role))
                    target.commit()
                finally:
                    source.close()
                    target.close()
                hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
        (staging / "pair.json").write_text(json.dumps({"version": version, "hashes": hashes}))
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise StorageError("backup_failed") from None


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def restore_pair(source: Path, data_dir: Path, environment: str, *, force: bool = False) -> None:
    """Offline restore using SQLite's atomic multi-database rollback-journal transaction."""
    destinations = {
        "tenant": data_dir / f"{environment}.sqlite3",
        "control": data_dir / "control" / f"{environment}.sqlite3",
    }
    if any(p.exists() for p in destinations.values()) and not force:
        raise StorageError("restore_destination_exists")
    if not source.is_dir():
        raise StorageError("backup_not_found")
    existing = {role: p.exists() for role, p in destinations.items()}
    connection: sqlite3.Connection | None = None
    try:
        pair = json.loads((source / "pair.json").read_text())
        for role, migrations in (("tenant", MIGRATIONS), ("control", CONTROL_MIGRATIONS)):
            path = source / f"{role}.sqlite3"
            if path.resolve() in {p.resolve() for p in destinations.values()}:
                raise ValueError
            if hashlib.sha256(path.read_bytes()).hexdigest() != pair["hashes"][role]:
                raise ValueError
            db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            try:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError
                if db.execute("SELECT version, role FROM backup_pair").fetchall() != [
                    (pair["version"], role)
                ]:
                    raise ValueError
                versions = {r[0] for r in db.execute("SELECT version FROM schema_migrations")}
                expected = {int(p.name.split("_")[0]) for p in migrations.glob("[0-9]*_*.sql")}
                # Tenant ledgers may retain the old, now split, migration 6.
                if not expected <= versions or versions - expected - (
                    {6} if role == "tenant" else set()
                ):
                    raise ValueError
            finally:
                db.close()
        for path in destinations.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(destinations["tenant"], isolation_level=None)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("ATTACH DATABASE ? AS control", (str(destinations["control"]),))
        connection.execute("PRAGMA control.journal_mode=DELETE")
        connection.execute("PRAGMA control.synchronous=FULL")
        connection.execute("ATTACH DATABASE ? AS tenant_source", (str(source / "tenant.sqlite3"),))
        connection.execute(
            "ATTACH DATABASE ? AS control_source", (str(source / "control.sqlite3"),)
        )
        connection.execute("BEGIN IMMEDIATE")
        for role, target in (("tenant", "main"), ("control", "control")):
            origin = f"{role}_source"
            for (name,) in connection.execute(
                f"SELECT name FROM {target}.sqlite_master WHERE type='table'"
            ).fetchall():
                connection.execute(f"DROP TABLE {target}.{_quote(name)}")
            schema = connection.execute(
                f"SELECT type, name, sql FROM {origin}.sqlite_master WHERE sql IS NOT NULL "
                "AND name != 'backup_pair' ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END"
            ).fetchall()
            for kind, name, sql in schema:
                # Rebind the validated backup schema to the destination.
                pattern = r'^(CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+)(?:"[^"]+"|\w+)'
                match = re.match(pattern, sql)
                if match is None:
                    raise ValueError
                rebound = f"{match[1]}{target}.{_quote(name)}{sql[match.end() :]}"
                connection.execute(rebound)
                if kind == "table":
                    connection.execute(
                        f"INSERT INTO {target}.{_quote(name)} SELECT * FROM {origin}.{_quote(name)}"
                    )
        connection.commit()
    except Exception:
        if connection is not None:
            connection.rollback()
            connection.close()
            connection = None
        for role, path in destinations.items():
            if not existing[role]:
                path.unlink(missing_ok=True)
        raise StorageError("restore_pair_failed") from None
    finally:
        if connection is not None:
            connection.close()
