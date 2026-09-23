"""Local backup, offline restore and batched tenant payload key rotation."""

import os
import sqlite3
from pathlib import Path

from adaptive_llm.contracts import now, uid
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.migrations import migrate
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLitePayloadStore, timestamp


def rotate_key(
    database: SQLiteDatabase, tenant_id: str, cipher: PayloadCipher, *, batch_size: int = 100
) -> int:
    if batch_size < 1:
        raise ValueError("invalid_rotation_batch_size")
    payloads = SQLitePayloadStore(database)
    with database.lock:
        ceiling = database.connection.execute(
            "SELECT coalesce(max(rowid), 0) FROM payloads WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()[0]
    cursor, changed = 0, 0
    while True:
        with database.transaction():
            at = now()
            rows = database.connection.execute(
                "SELECT rowid, reference FROM payloads WHERE tenant_id = ? "
                "AND rowid > ? AND rowid <= ? AND expires_at > ? AND key_version != ? "
                "ORDER BY rowid LIMIT ?",
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
            cursor = rows[-1]["rowid"]
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
