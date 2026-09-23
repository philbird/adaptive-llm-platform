import shutil
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from adaptive_llm.contracts import Interaction, uid
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.__main__ import main
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.migrations import MIGRATIONS, migrate
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, timestamp

if TYPE_CHECKING:
    from conftest import DatasetSeed

AT = datetime(2026, 9, 23, tzinfo=UTC)


def test_migrations_fresh_repeat_and_failed_batch_rollback(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "migration.sqlite3", isolation_level=None)
    migrate(connection)
    assert [row[1] for row in connection.execute("PRAGMA table_info(schema_migrations)")] == [
        "version",
        "applied_at",
    ]
    versions = connection.execute("SELECT * FROM schema_migrations").fetchall()
    assert len(versions) == len(list(MIGRATIONS.glob("[0-9]*_*.sql")))
    migrate(connection)
    assert connection.execute("SELECT * FROM schema_migrations").fetchall() == versions
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "9998_good.sql").write_text("CREATE TABLE rolled_back (tenant_id TEXT);\n")
    (migrations / "9999_bad.sql").write_text(
        "CREATE TABLE also_rolled_back (tenant_id TEXT);\nBAD SQL;\n"
    )
    with pytest.raises(StorageError, match="^migration_failed$"):
        migrate(connection, migrations)
    assert connection.execute("SELECT * FROM schema_migrations").fetchall() == versions
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert "rolled_back" not in tables
    assert "also_rolled_back" not in tables
    connection.close()


def test_failed_first_migration_leaves_no_schema(tmp_path: Path) -> None:
    (tmp_path / "0001_bad.sql").write_text("CREATE TABLE partial (tenant_id TEXT);\nBAD SQL;\n")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    with pytest.raises(StorageError):
        migrate(connection, tmp_path)
    assert not connection.execute("SELECT name FROM sqlite_master").fetchall()
    connection.close()


def test_upgrade_removes_legacy_schema_tenant_without_changing_applied_version(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    shutil.copyfile(MIGRATIONS / "0001_persistence.sql", legacy / "0001_persistence.sql")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    migrate(connection, legacy)
    first = connection.execute("SELECT version, applied_at FROM schema_migrations").fetchone()
    connection.execute(
        "ALTER TABLE schema_migrations ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '__schema__'"
    )
    migrate(connection)
    assert [row[1] for row in connection.execute("PRAGMA table_info(schema_migrations)")] == [
        "version",
        "applied_at",
    ]
    assert (
        connection.execute(
            "SELECT version, applied_at FROM schema_migrations WHERE version = 1"
        ).fetchone()
        == first
    )
    assert connection.execute("SELECT count(*) FROM subject_tombstones").fetchone()[0] == 0
    connection.close()


def test_cipher_round_trip_unique_nonces_and_binding() -> None:
    cipher = PayloadCipher(b"s" * 32)
    iid = uid()
    first = cipher.encrypt(b"SYNTHETIC private content", "synthetic-a", iid, "messages", AT)
    second = cipher.encrypt(b"SYNTHETIC private content", "synthetic-a", iid, "messages", AT)
    assert cipher.decrypt(first, "synthetic-a", iid, "messages") == b"SYNTHETIC private content"
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    for tenant, interaction, field in (
        ("synthetic-b", iid, "messages"),
        ("synthetic-a", uid(), "messages"),
        ("synthetic-a", iid, "output"),
    ):
        with pytest.raises(StorageError, match="^payload_authentication_failed$"):
            cipher.decrypt(first, tenant, interaction, field)
    tampered = replace(first, ciphertext=bytes([first.ciphertext[0] ^ 1]) + first.ciphertext[1:])
    with pytest.raises(StorageError, match="^payload_authentication_failed$"):
        cipher.decrypt(tampered, "synthetic-a", iid, "messages")
    with pytest.raises(ValueError, match="^invalid_payload_key$"):
        PayloadCipher(b"short")
    with pytest.raises(StorageError, match="^unknown_payload_key_version$"):
        cipher.decrypt(replace(first, key_version="other"), "synthetic-a", iid, "messages")


def test_storage_cli_and_disabled_migration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = SQLiteDatabase(tmp_path, migrate_on_startup=False)
    assert not database.connection.execute("SELECT name FROM sqlite_master").fetchall()
    database.close()
    main(["migrate", "--data-dir", str(tmp_path)])
    main(["retention-sweep", "--data-dir", str(tmp_path), "--tenant", "synthetic-a"])
    assert capsys.readouterr().out == "migrations_applied\nexpired_interactions=0\n"
    with pytest.raises(SystemExit):
        main(["retention-sweep"])


def test_payload_expiration_and_key_settings(tmp_path: Path) -> None:
    from adaptive_llm.app import Settings

    assert Settings().replay_ttl_seconds == 86_400
    assert Settings().retention_seconds is None
    assert Settings().migrate_on_startup
    assert Settings().payload_key == Settings().payload_key
    for kwargs in ({"replay_ttl_seconds": 0}, {"retention_seconds": 0}, {"payload_key": b"bad"}):
        with pytest.raises(ValueError):
            Settings(**kwargs)
    with pytest.raises(ValueError, match="^payload_key_required$"):
        Settings(environment="production")
    assert "ssss" not in repr(Settings(payload_key=b"s" * 32))
    local = SQLiteDatabase(tmp_path)
    staging = SQLiteDatabase(tmp_path, "staging")
    assert local.path != staging.path
    local.close()
    staging.close()


def test_started_at_migration_backfill_index_and_window_boundaries(
    tmp_path: Path,
    dataset_seed: "DatasetSeed",
) -> None:
    legacy = tmp_path / "migration-4"
    legacy.mkdir()
    for path in MIGRATIONS.glob("[0-9]*_*.sql"):
        if int(path.name.split("_")[0]) <= 4:
            shutil.copyfile(path, legacy / path.name)
    database = SQLiteDatabase(tmp_path / "legacy-data", migrations=legacy)
    metadata = SQLiteMetadataStore(database)
    original = dataset_seed.app.state.metadata.get("synthetic-a", Interaction, dataset_seed.ids[2])
    instances = [
        original.model_copy(update={"interaction_id": uid(), "started_at": at, "tenant_id": tenant})
        for tenant, at in [
            ("synthetic-a", AT - timedelta(microseconds=1)),
            ("synthetic-a", AT),
            ("synthetic-a", AT + timedelta(microseconds=1)),
            ("synthetic-a", AT + timedelta(seconds=1)),
            ("synthetic-b", AT),
        ]
    ]
    try:
        with database.transaction():
            for item in instances:
                database.connection.execute(
                    "INSERT INTO interactions "
                    "(tenant_id, record_id, interaction_id, data, expires_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        item.tenant_id,
                        item.interaction_id,
                        item.interaction_id,
                        item.model_dump_json(),
                        timestamp(AT + timedelta(hours=1)),
                    ),
                )
        migrate(database.connection)
        assert [
            row[0]
            for row in database.connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ] == [1, 2, 3, 4, 5]
        assert database.connection.execute(
            "SELECT started_at FROM interactions WHERE record_id=?", (instances[1].interaction_id,)
        ).fetchone()[0] == timestamp(AT)
        assert metadata.in_window("synthetic-a", AT, AT + timedelta(seconds=1)) == instances[1:3]
        inserted = original.model_copy(
            update={"interaction_id": uid(), "started_at": AT + timedelta(microseconds=2)}
        )
        with metadata.transaction():
            metadata.put("synthetic-a", inserted, AT + timedelta(hours=1))
            # An unrelated corrupt row must never be loaded or parsed for this source window.
            database.connection.execute(
                "UPDATE interactions SET data=? WHERE record_id=?",
                ("synthetic invalid record", instances[0].interaction_id),
            )
        assert metadata.in_window("synthetic-a", AT, AT + timedelta(seconds=1)) == [
            *instances[1:3],
            inserted,
        ]
        plan = database.connection.execute(
            "EXPLAIN QUERY PLAN SELECT data FROM interactions WHERE tenant_id=? "
            "AND started_at>=? AND started_at<? ORDER BY started_at, record_id",
            ("synthetic-a", timestamp(AT), timestamp(AT + timedelta(seconds=1))),
        ).fetchall()
        assert any("USING INDEX interactions_tenant_started_at" in row[3] for row in plan)
    finally:
        database.close()
