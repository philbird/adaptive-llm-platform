"""Snapshot-consistent paired pg_dump and validated, atomic pg_restore."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from adaptive_llm.contracts import Environment, uid
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.migrations import CONTROL_MIGRATIONS, MIGRATIONS
from adaptive_llm.storage.postgres import PostgresDatabase


def pg_tool(command: str, database_url: str, arguments: list[str]) -> bytes:
    # Credentials go through environment, never argv, errors or application logs.
    info = conninfo_to_dict(database_url)
    environment = dict(os.environ)
    for key, value in info.items():
        environment["PG" + key.upper()] = str(value)
    environment["PGDATABASE"] = str(info.get("dbname", "postgres"))
    try:
        return subprocess.run(
            [command, *arguments], env=environment, check=True, capture_output=True, timeout=120
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise StorageError("postgres_backup_tool_failed") from None


def backup_postgres(tenant: PostgresDatabase, control: PostgresDatabase, output: Path) -> None:
    if output.exists():
        raise StorageError("backup_destination_exists")
    if tenant.database_url != control.database_url:
        raise StorageError("backup_pair_database_mismatch")
    staging = output.with_name(f".{uid()}.backup")
    try:
        staging.mkdir(parents=True, mode=0o700)
        raw = tenant.connection.raw
        with tenant.lock, raw.transaction():
            raw.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            snapshot = raw.execute("SELECT pg_export_snapshot() AS snapshot").fetchone()
            assert snapshot is not None
            archive = pg_tool(
                "pg_dump",
                tenant.database_url,
                [
                    "--format=custom",
                    "--no-owner",
                    "--no-privileges",
                    f"--snapshot={snapshot['snapshot']}",
                    f"--schema={tenant.schema}",
                    f"--schema={control.schema}",
                ],
            )
        (staging / "pair.dump").write_bytes(archive)
        (staging / "pair.json").write_text(
            json.dumps(
                {
                    "version": uid(),
                    "backend": "postgres",
                    "schemas": [tenant.schema, control.schema],
                    "sha256": hashlib.sha256(archive).hexdigest(),
                }
            )
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise StorageError("backup_failed") from None


def restore_postgres(
    source: Path, database_url: str, environment: Environment, *, force: bool = False
) -> None:
    """Validate in a disposable database, then replace both schemas in one server transaction."""
    temporary = "restore_" + uid().replace("-", "")
    admin = None
    created = False
    try:
        schemas = [f"adaptive_{environment}_{role}" for role in ("tenant", "control")]
        manifest = json.loads((source / "pair.json").read_text())
        archive = source / "pair.dump"
        if (
            manifest["backend"] != "postgres"
            or manifest["schemas"] != schemas
            or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]
        ):
            raise ValueError
        admin = psycopg.connect(database_url, autocommit=True, connect_timeout=3)
        existing = admin.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname = ANY(%s)", (schemas,)
        ).fetchall()
        if existing and not force:
            raise StorageError("restore_destination_exists")
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(temporary)))
        created = True
        scratch_url = make_conninfo(database_url, dbname=temporary)
        arguments = ["--exit-on-error", "--no-owner", "--no-privileges", "--single-transaction"]
        pg_tool("pg_restore", scratch_url, [*arguments, "--dbname=" + temporary, str(archive)])
        with psycopg.connect(scratch_url, connect_timeout=3) as check:
            actual = check.execute(
                "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' "
                "AND nspname != 'information_schema'"
            ).fetchall()
            if {r[0] for r in actual} != {"public", *schemas}:
                raise ValueError
            for schema, directory in zip(schemas, (MIGRATIONS, CONTROL_MIGRATIONS), strict=True):
                versions = {
                    r[0]
                    for r in check.execute(
                        sql.SQL("SELECT version FROM {}.schema_migrations").format(
                            sql.Identifier(schema)
                        )
                    )
                }
                expected = {int(p.name.split("_")[0]) for p in directory.glob("[0-9]*_*.sql")}
                if not expected <= versions:
                    raise ValueError
        target_name = conninfo_to_dict(database_url).get("dbname", "postgres")
        pg_tool(
            "pg_restore",
            database_url,
            [*arguments, "--clean", "--if-exists", "--dbname=" + str(target_name), str(archive)],
        )
    except StorageError as error:
        if str(error) == "restore_destination_exists":
            raise
        raise StorageError("restore_pair_failed") from None
    except Exception:
        raise StorageError("restore_pair_failed") from None
    finally:
        if admin is not None:
            try:
                if created:
                    admin.execute(
                        sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(temporary))
                    )
            except psycopg.Error:
                raise StorageError("restore_cleanup_failed") from None
            finally:
                admin.close()
