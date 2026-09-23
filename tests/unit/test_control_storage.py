import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from adaptive_llm.storage import StorageError
from adaptive_llm.storage.migrations import CONTROL_MIGRATIONS, MIGRATIONS, migrate
from adaptive_llm.storage.operations import backup_pair, restore_pair
from adaptive_llm.storage.sqlite import SQLiteDatabase, control_database


def names(db: SQLiteDatabase) -> set[str]:
    return {
        r[0] for r in db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_fresh_migrations_are_separate_and_legacy_control_moves(tmp_path: Path) -> None:
    tenant = SQLiteDatabase(tmp_path)
    control = control_database(tmp_path)
    assert "training_jobs" not in names(tenant) and "evaluation_reports" not in names(tenant)
    assert "interactions" not in names(control) and "dataset_manifests" not in names(control)
    assert {"outbox", "evaluation_reports", "model_versions", "training_jobs"} <= names(control)
    tenant.close()
    control.close()
    # Recreate exactly the old control schema (tenant migrations 1-5 plus evaluation migration 6).
    legacy_migrations = tmp_path / "old-migrations"
    legacy_migrations.mkdir()
    for path in MIGRATIONS.glob("000[1-5]_*.sql"):
        shutil.copyfile(path, legacy_migrations / path.name)
    shutil.copyfile(
        CONTROL_MIGRATIONS / "0006_evaluations.sql", legacy_migrations / "0006_evaluations.sql"
    )
    legacy_dir = tmp_path / "legacy" / "evaluations" / "control"
    old = SQLiteDatabase(legacy_dir, migrations=legacy_migrations)
    old.connection.execute(
        "INSERT INTO evaluation_reports VALUES "
        "('synthetic-id','[]','{}','mac',NULL,NULL,'synthetic-at')"
    )
    old.close()
    upgraded = control_database(tmp_path / "legacy")
    assert not legacy_dir.exists()
    assert upgraded.path == tmp_path / "legacy/control/local.sqlite3"
    assert "interactions" not in names(upgraded)
    assert (
        upgraded.connection.execute("SELECT evaluation_id FROM evaluation_reports").fetchone()[0]
        == "synthetic-id"
    )
    assert [r[0] for r in upgraded.connection.execute("SELECT version FROM schema_migrations")] == [
        3,
        6,
        7,
    ]
    upgraded.close()


def test_populated_legacy_tenant_control_tables_fail_without_data_loss(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "legacy.sqlite3", isolation_level=None)
    connection.execute("CREATE TABLE evaluation_reports (evaluation_id TEXT)")
    connection.execute("INSERT INTO evaluation_reports VALUES ('synthetic-preserve')")
    with pytest.raises(StorageError, match="migration_failed"):
        migrate(connection)
    assert (
        connection.execute("SELECT * FROM evaluation_reports").fetchone()[0] == "synthetic-preserve"
    )
    connection.close()


def test_paired_backups_mismatch_atomic_restore_and_repeat(tmp_path: Path) -> None:
    data = tmp_path / "data"
    tenant, control = SQLiteDatabase(data), control_database(data)
    control.connection.execute(
        "INSERT INTO training_jobs VALUES ('synthetic-job', '[]', 'synthetic-v1')"
    )
    first, second = tmp_path / "first", tmp_path / "second"
    backup_pair(tenant, control, first)
    control.connection.execute("UPDATE training_jobs SET data='synthetic-v2'")
    backup_pair(tenant, control, second)
    tenant.close()
    control.close()
    before = {p: p.read_bytes() for p in [data / "local.sqlite3", data / "control/local.sqlite3"]}
    mixed = tmp_path / "mixed"
    shutil.copytree(first, mixed)
    shutil.copyfile(second / "control.sqlite3", mixed / "control.sqlite3")
    # Even with adjusted hashes, a cross-generation pair must fail the embedded version check.
    pair = json.loads((mixed / "pair.json").read_text())
    pair["hashes"]["control"] = hashlib.sha256((mixed / "control.sqlite3").read_bytes()).hexdigest()
    (mixed / "pair.json").write_text(json.dumps(pair))
    with pytest.raises(StorageError, match="restore_pair_failed"):
        restore_pair(mixed, data, "local", force=True)
    assert all(p.read_bytes() == content for p, content in before.items())
    broken = tmp_path / "broken"
    shutil.copytree(first, broken)
    db = sqlite3.connect(broken / "control.sqlite3")
    db.execute("CREATE VIEW unsupported AS SELECT * FROM training_jobs")
    db.close()
    pair = json.loads((broken / "pair.json").read_text())
    pair["hashes"]["control"] = hashlib.sha256(
        (broken / "control.sqlite3").read_bytes()
    ).hexdigest()
    (broken / "pair.json").write_text(json.dumps(pair))
    with pytest.raises(StorageError, match="restore_pair_failed"):
        restore_pair(broken, data, "local", force=True)
    assert all(p.read_bytes() == content for p, content in before.items())
    restore_pair(first, data, "local", force=True)
    tenant, control = SQLiteDatabase(data), control_database(data)
    assert (
        control.connection.execute("SELECT data FROM training_jobs").fetchone()[0] == "synthetic-v1"
    )
    # Restored SQLite schemas contain quoted names; a second backup/restore still works.
    third = tmp_path / "third"
    backup_pair(tenant, control, third)
    tenant.close()
    control.close()
    restore_pair(third, data, "local", force=True)
