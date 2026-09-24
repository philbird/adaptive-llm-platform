import subprocess

import psycopg
import pytest

from adaptive_llm.storage import StorageError
from adaptive_llm.storage.postgres import PostgresConnection, Row
from adaptive_llm.storage.postgres_operations import pg_tool


def test_parameter_adaptation_keeps_bindings_out_of_sql_and_errors():
    queries = []

    class Raw:
        rowcount = 1

        def execute(self, query, parameters):
            queries.append((query, parameters))
            if parameters == ("synthetic-error",):
                raise psycopg.ProgrammingError("SYNTHETIC PRIVATE ERROR")
            return self

        def fetchall(self):
            return [{"data": "synthetic"}]

        def fetchone(self):
            return {"data": "synthetic"}

    connection = PostgresConnection(Raw())
    cursor = connection.execute(
        "SELECT json_extract(j.data, '$.specification.version') FROM jobs j WHERE id=?",
        ("synthetic-binding",),
    )
    assert queries[0] == (
        "SELECT (j.data::jsonb #>> '{specification,version}') FROM jobs j WHERE id=%s",
        ("synthetic-binding",),
    )
    assert cursor.fetchone()[0] == cursor.fetchall()[0]["data"] == "synthetic"
    assert list(cursor) == [Row(data="synthetic")]
    with pytest.raises(StorageError, match="^database_operation_failed$"):
        connection.execute("SELECT ?", ("synthetic-error",))


def test_pg_tool_uses_environment_and_fixed_errors(monkeypatch):
    def run(command, **kwargs):
        assert "synthetic-password" not in repr(command)
        assert kwargs["env"]["PGPASSWORD"] == "synthetic-password"
        assert kwargs["env"]["PGDATABASE"] == "synthetic"
        raise subprocess.CalledProcessError(1, command, stderr=b"SYNTHETIC PRIVATE ERROR")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(StorageError, match="^postgres_backup_tool_failed$"):
        pg_tool("pg_dump", "postgresql://synthetic:synthetic-password@localhost/synthetic", [])


@pytest.mark.parametrize("kind", ["outbox", "worker"])
def test_claims_reuse_connections_per_thread_and_close_all(kind, monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Barrier, RLock
    from unittest.mock import Mock

    from psycopg.pq import TransactionStatus

    from adaptive_llm.contracts import now
    from adaptive_llm.gateway.identity import Keyring
    from adaptive_llm.storage.postgres import PostgresDatabase
    from adaptive_llm.storage.postgres_stores import PostgresModelRegistry, PostgresOutboxStore

    connections = []

    def connect():
        raw = Mock(closed=False)
        raw.info.transaction_status = TransactionStatus.IDLE
        raw.close.side_effect = lambda: setattr(raw, "closed", True)
        raw.execute.return_value.fetchone.return_value = None
        raw.commits, raw.rollbacks = 0, 0

        @contextmanager
        def transaction():
            assert raw.info.transaction_status == TransactionStatus.IDLE
            raw.info.transaction_status = TransactionStatus.INTRANS
            try:
                yield
                raw.commits += 1
            except Exception:
                raw.rollbacks += 1
                raise
            finally:
                raw.info.transaction_status = TransactionStatus.IDLE

        raw.transaction.side_effect = transaction
        connection = PostgresConnection(raw)
        connections.append(connection)
        return connection

    database = object.__new__(PostgresDatabase)
    database.lock = RLock()
    database._claim_connections = []
    database.connection = connect()
    monkeypatch.setattr(database, "connect", connect)
    store = (
        PostgresOutboxStore(database)
        if kind == "outbox"
        else PostgresModelRegistry(database, None, Keyring(b"synthetic"), None, tmp_path, 100)
    )
    claim = (
        (lambda: store.claim(now()))
        if kind == "outbox"
        else (lambda: store.worker_claim(("synthetic",)))
    )
    assert len(connections) == 1  # lazy until first claim
    for _ in range(1000):
        with claim() as value:
            assert not value
    assert len(connections) == 2
    assert connections[1].raw.commits == 1000
    with pytest.raises(ValueError, match="synthetic_failure"), claim():
        raise ValueError("synthetic_failure")
    assert connections[1].raw.rollbacks == 1
    with claim():
        with pytest.raises(StorageError, match="claim_already_active"), claim():
            pass
    # A dropped connection is replaced on the next claim.
    connections[1].raw.close()
    with claim():
        pass
    assert len(connections) == 3
    barrier = Barrier(2)

    def run(_):
        for _ in range(2):
            with claim():
                barrier.wait(timeout=3)
        return store._connections.get()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.map(run, range(2))
    assert a is not b and len(connections) == 5
    assert a.raw.commits == b.raw.commits == 2
    store.close()
    assert all(c.raw.closed for c in connections[1:])
    assert not database.connection.raw.closed
    with pytest.raises(StorageError, match="claim_connections_closed"), claim():
        pass
    # Database ownership also closes pools when callers only close their database.
    other = PostgresOutboxStore(database)
    with other.claim(now()):
        pass
    assert not connections[-1].raw.closed
    database.close()
    assert all(c.raw.closed for c in connections)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("failure", ["docker_absent", "image_absent", "network_absent"])
def test_postgres_fixture_skip_or_required_failure(required, failure, monkeypatch):
    import postgres_support

    monkeypatch.setenv("REQUIRE_POSTGRES", "1" if required else "0")
    monkeypatch.setattr(
        postgres_support.shutil, "which", lambda _: None if failure == "docker_absent" else "docker"
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == "image_absent":
            raise subprocess.CalledProcessError(1, command)
        if command[1] == "inspect":
            return subprocess.CompletedProcess(
                command, 0, b'[{"NetworkSettings":{"Ports":{"5432/tcp":[{"HostPort":"1234"}]}}}]'
            )
        return subprocess.CompletedProcess(command, 0, b"")

    def connect(*args, **kwargs):
        raise psycopg.OperationalError("SYNTHETIC CONNECTION UNAVAILABLE")

    monkeypatch.setattr(postgres_support.subprocess, "run", run)
    monkeypatch.setattr(postgres_support.psycopg, "connect", connect)
    times = iter([0, 16])
    monkeypatch.setattr(postgres_support.time, "monotonic", lambda: next(times))
    expected = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(expected, match=f"^{postgres_support.SKIP}$"):
        next(postgres_support.postgres_server.__wrapped__())
    if failure == "network_absent":
        assert calls[-1][1:3] == ["rm", "-f"]
