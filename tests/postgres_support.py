"""Network-free PostgreSQL fixture: cached image only, bounded startup, fixed skip reason."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from adaptive_llm.contracts import uid

IMAGE = "postgres:17.6"
SKIP = "PostgreSQL requires Docker and cached postgres:17.6 with accessible local networking"


def postgres_unavailable():
    if os.environ.get("REQUIRE_POSTGRES") == "1":
        raise pytest.fail.Exception(SKIP, pytrace=False) from None
    raise pytest.skip.Exception(SKIP) from None


@pytest.fixture(scope="session")
def postgres_server():
    if shutil.which("docker") is None:
        postgres_unavailable()
    name = "adaptive-p1-" + uid()
    started = False
    try:
        subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, check=True, timeout=10
        )
        subprocess.run(
            [
                "docker",
                "run",
                "--pull=never",
                "--rm",
                "-d",
                "--name",
                name,
                "-e",
                "POSTGRES_PASSWORD=synthetic-test-only",
                "-p",
                "127.0.0.1::5432",
                IMAGE,
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
        started = True
        result = subprocess.run(
            ["docker", "inspect", name], capture_output=True, check=True, timeout=5
        )
        port = json.loads(result.stdout)[0]["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]
        url = f"postgresql://postgres:synthetic-test-only@127.0.0.1:{port}/postgres"
        deadline = time.monotonic() + 15
        while True:
            try:
                with psycopg.connect(url, connect_timeout=1):
                    break
            except psycopg.Error:
                if time.monotonic() > deadline:
                    postgres_unavailable()
                time.sleep(0.1)
        yield url, name
    except (OSError, subprocess.SubprocessError):
        postgres_unavailable()
    finally:
        if started:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)


@pytest.fixture
def postgres_urls(postgres_server):
    url, _ = postgres_server
    databases = {}

    def for_path(path: Path) -> str:
        name = "test_" + hashlib.sha256(str(path).encode()).hexdigest()[:24]
        if name not in databases:
            with psycopg.connect(url, autocommit=True) as admin:
                admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            databases[name] = make_conninfo(url, dbname=name)
        return databases[name]

    yield for_path
    with psycopg.connect(url, autocommit=True) as admin:
        for name in databases:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


BACKEND_MODULES = {
    "test_storage",
    "test_outbox",
    "test_registry",
    "test_replay",
    "test_persistence",
    "test_outbox_lifecycle",
    "test_training",
    "test_training_queue",
    "test_persistence_privacy",
    "test_training_privacy",
    "test_evaluation_privacy",
    "test_feedback_datasets",
    "test_evaluations",
    "test_signatures",
    "test_live_routing",
    "test_shadow_routing",
}


def pytest_generate_tests(metafunc):
    sqlite_specific = {
        "test_migrations_fresh_repeat_and_failed_batch_rollback",
        "test_failed_first_migration_leaves_no_schema",
        "test_upgrade_removes_legacy_schema_tenant_without_changing_applied_version",
        "test_storage_cli_and_disabled_migration",
        "test_payload_expiration_and_key_settings",
    }
    if (
        metafunc.module.__name__.split(".")[-1] in BACKEND_MODULES
        and metafunc.function.__name__ not in sqlite_specific
    ):
        metafunc.parametrize("storage_backend", ["sqlite", "postgres"], indirect=True)


@pytest.fixture(autouse=True)
def storage_backend(request, monkeypatch):
    if getattr(request, "param", "sqlite") == "sqlite":
        yield "sqlite"
        return
    urls = request.getfixturevalue("postgres_urls")
    from adaptive_llm.app import Settings
    from adaptive_llm.storage import sqlite
    from adaptive_llm.storage.postgres import PostgresDatabase
    from adaptive_llm.storage.postgres_stores import PostgresOutboxStore

    original_settings = Settings.__post_init__
    original_database = sqlite.SQLiteDatabase

    def settings(instance):
        object.__setattr__(instance, "storage_backend", "postgres")
        object.__setattr__(instance, "database_url", urls(instance.data_dir))
        original_settings(instance)

    def database(path, environment="local", **kwargs):
        if kwargs.get("in_memory") or "migrations" in kwargs:
            # Legacy SQLite migration fixtures intentionally exercise SQLite upgrades.
            return original_database(path, environment, **kwargs)
        return PostgresDatabase(
            urls(path), environment, migrate_on_startup=kwargs.get("migrate_on_startup", True)
        )

    monkeypatch.setattr(Settings, "__post_init__", settings)
    # Patch direct test constructors too; evaluation scratch databases stay in-memory SQLite.
    if hasattr(request.module, "SQLiteDatabase"):
        monkeypatch.setattr(request.module, "SQLiteDatabase", database)
    if hasattr(request.module, "SQLiteOutboxStore"):
        monkeypatch.setattr(request.module, "SQLiteOutboxStore", PostgresOutboxStore)
    yield "postgres"


def stored_bytes(database) -> bytes:
    """Inspect persisted rows on PostgreSQL and the physical file on SQLite."""
    from adaptive_llm.storage.postgres import PostgresDatabase

    if not isinstance(database, PostgresDatabase):
        return database.path.read_bytes()
    connection = database.connect()
    try:
        tables = connection.raw.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=%s", (database.schema,)
        ).fetchall()
        rows = []
        for table in tables:
            rows.extend(
                connection.raw.execute(
                    sql.SQL("SELECT * FROM {}.{}").format(
                        sql.Identifier(database.schema), sql.Identifier(table["tablename"])
                    )
                ).fetchall()
            )
        return repr(rows).encode()
    finally:
        connection.raw.close()
