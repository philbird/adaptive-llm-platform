import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

import pytest
from psycopg.conninfo import conninfo_to_dict

from adaptive_llm.contracts import Event, Started, TrainingJob, TrainingJobSpecification, now, uid
from adaptive_llm.events.outbox import Dispatcher
from adaptive_llm.gateway.identity import Identity, Keyring
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.migrations import CONTROL_MIGRATIONS, MIGRATIONS
from adaptive_llm.storage.postgres import PostgresDatabase
from adaptive_llm.storage.postgres_operations import backup_postgres, restore_postgres
from adaptive_llm.storage.postgres_stores import PostgresModelRegistry, PostgresOutboxStore


@pytest.mark.parametrize("shared_store", [False, True])
def test_two_dispatchers_deliver_1000_events_once_in_order(
    postgres_urls, tmp_path, shared_store, monkeypatch
):
    url = postgres_urls(tmp_path)
    first, second = PostgresDatabase(url), PostgresDatabase(url)
    calls, mutex = [], Lock()

    class Sink:
        def emit(self, event):
            with mutex:
                calls.append(event.event_id)

    # 100 ordered streams, 10 events each; no consumer dedup hides duplicate deliveries.
    events = [
        Event(
            tenant_id="synthetic-a",
            trace_id=uid(),
            event_type="interaction.started.v1",
            data=Started(
                interaction_id=f"synthetic-{i // 10}",
                application_id="synthetic",
                policy_version="synthetic-1",
            ),
        )
        for i in range(1000)
    ]
    a, b = PostgresOutboxStore(first), PostgresOutboxStore(second)
    connections = []
    for database in (first, second):
        original = database.connect

        def connect(original=original):
            connection = original()
            with mutex:
                connections.append(connection)
            return connection

        monkeypatch.setattr(database, "connect", connect)
    try:
        with first.transaction():
            a.enqueue(events, 2000)
        dispatchers = [
            Dispatcher(store, Sink(), InProcessMetrics()) for store in (a, a if shared_store else b)
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sum(pool.map(lambda d: d.dispatch_once(1000), dispatchers)) == 1000
        assert len(connections) == 2
        assert len(calls) == len(set(calls)) == 1000
        positions = {event_id: i for i, event_id in enumerate(calls)}
        for start in range(0, 1000, 10):
            assert [positions[e.event_id] for e in events[start : start + 10]] == sorted(
                positions[e.event_id] for e in events[start : start + 10]
            )
        assert a.stats(now()).pending == 0
        # A failed claim rolls back and releases its lock for another process.
        extra = events[0].model_copy(update={"event_id": uid()})
        with first.transaction():
            a.enqueue([extra], 2000)
        with pytest.raises(RuntimeError):
            with a.claim(now()) as claimed:
                assert claimed.event_id == extra.event_id
                with b.claim(now()) as unavailable:
                    assert unavailable is None
                raise RuntimeError("synthetic-crash")
        assert dispatchers[1].dispatch_once() == 1
    finally:
        first.close()
        second.close()
        assert all(connection.raw.closed for connection in connections)


def test_worker_claims_are_distinct_recoverable_and_allow_progress(postgres_urls, tmp_path):
    url = postgres_urls(tmp_path)
    first, second = PostgresDatabase(url, role="control"), PostgresDatabase(url, role="control")
    actor = Identity(
        "synthetic-a",
        frozenset(),
        "local",
        "synthetic-operator",
        "operator",
        frozenset({"synthetic-a"}),
    )
    try:
        registries = [
            PostgresModelRegistry(
                db, PostgresOutboxStore(db), Keyring(b"synthetic"), None, tmp_path, 100
            )
            for db in (first, second)
        ]
        jobs = [
            TrainingJob(
                specification=TrainingJobSpecification(
                    dataset_id="synthetic", dataset_version=uid()
                ),
                trainer_architecture="synthetic-trainer",
            )
            for _ in range(2)
        ]
        for job in jobs:
            registries[0].enqueue(job, ["synthetic-a"], actor)
        with ExitStack() as stack:
            a, b = [stack.enter_context(r.worker_claim(("synthetic-trainer",))) for r in registries]
            assert len(a) == len(b) == 1
            assert a[0][0].specification.job_id != b[0][0].specification.job_id
            # Claim row is independent of progress/cancellation rows.
            registries[0].save_job(a[0][0].model_copy(update={"state": "running"}), ["synthetic-a"])
        with registries[1].worker_claim(("synthetic-trainer",)) as recovered:
            assert recovered[0][0].specification.job_id == jobs[0].specification.job_id
            assert recovered[0][0].state == "running"
    finally:
        first.close()
        second.close()


def test_postgres_migrations_repeat_and_failed_batch_is_atomic(postgres_urls, tmp_path):
    database = PostgresDatabase(postgres_urls(tmp_path))
    try:
        original = database.connection.execute(
            "SELECT * FROM schema_migrations ORDER BY version"
        ).fetchall()
        database.migrate(MIGRATIONS)
        assert (
            database.connection.execute(
                "SELECT * FROM schema_migrations ORDER BY version"
            ).fetchall()
            == original
        )
        broken = tmp_path / "bad-migrations"
        broken.mkdir()
        (broken / "9998_good.sql").write_text("CREATE TABLE synthetic_partial (value TEXT);")
        (broken / "9999_bad.sql").write_text("SYNTHETIC INVALID SQL;")
        with pytest.raises(StorageError, match="migration_failed"):
            database.migrate(broken)
        assert (
            database.connection.execute(
                "SELECT to_regclass('synthetic_partial') AS name"
            ).fetchone()[0]
            is None
        )
        assert (
            database.connection.execute(
                "SELECT * FROM schema_migrations ORDER BY version"
            ).fetchall()
            == original
        )
    finally:
        database.close()


@pytest.fixture
def container_pg_tools(postgres_server, monkeypatch):
    _, container = postgres_server

    def run(command, database_url, arguments):
        args, data = list(arguments), None
        if command == "pg_restore":
            data = Path(args.pop()).read_bytes()
        name = str(conninfo_to_dict(database_url)["dbname"])
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                f"PGDATABASE={name}",
                "-e",
                "PGUSER=postgres",
                container,
                command,
                *args,
            ],
            input=data,
            capture_output=True,
            timeout=60,
        )
        if result.returncode:
            raise StorageError("postgres_backup_tool_failed")
        return result.stdout

    monkeypatch.setattr("adaptive_llm.storage.postgres_operations.pg_tool", run)


def test_postgres_atomic_pair_backup_restore(postgres_urls, container_pg_tools, tmp_path):
    url = postgres_urls(tmp_path)
    tenant, control = PostgresDatabase(url), PostgresDatabase(url, role="control")
    output = tmp_path / "backup"
    try:
        for db in (tenant, control):
            with db.transaction():
                db.connection.execute("CREATE TABLE synthetic_marker (value TEXT)")
                db.connection.execute(
                    "INSERT INTO synthetic_marker VALUES (?)", ("synthetic-original",)
                )
        backup_postgres(tenant, control, output)
        for db in (tenant, control):
            with db.transaction():
                db.connection.execute("UPDATE synthetic_marker SET value=?", ("synthetic-changed",))
        with pytest.raises(StorageError, match="restore_destination_exists"):
            restore_postgres(output, url, "local")
        restore_postgres(output, url, "local", force=True)
        for db, migrations in ((tenant, MIGRATIONS), (control, CONTROL_MIGRATIONS)):
            assert (
                db.connection.execute("SELECT value FROM synthetic_marker").fetchone()[0]
                == "synthetic-original"
            )
            assert {
                row[0] for row in db.connection.execute("SELECT version FROM schema_migrations")
            } >= {int(p.name.split("_")[0]) for p in migrations.glob("[0-9]*_*.sql")}
        # A corrupt pair is refused before changing either live schema.
        (output / "pair.dump").write_bytes(b"synthetic-corrupt-archive")
        with pytest.raises(StorageError, match="restore_pair_failed"):
            restore_postgres(output, url, "local", force=True)
        assert (
            control.connection.execute("SELECT value FROM synthetic_marker").fetchone()[0]
            == "synthetic-original"
        )
    finally:
        tenant.close()
        control.close()
