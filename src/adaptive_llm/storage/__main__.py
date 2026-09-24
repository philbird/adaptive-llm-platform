"""Local operational commands. Output is limited to counts, ids and fixed codes."""

import argparse
import os
from pathlib import Path
from typing import cast

from adaptive_llm.contracts import Environment, now
from adaptive_llm.events import EventSink, InMemoryEventSink
from adaptive_llm.events.outbox import Dispatcher
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.crypto import PayloadCipher, load_keyring
from adaptive_llm.storage.database import Database
from adaptive_llm.storage.operations import backup_pair, restore_pair, rotate_key
from adaptive_llm.storage.outbox import SQLiteOutboxStore
from adaptive_llm.storage.postgres import PostgresDatabase
from adaptive_llm.storage.postgres_operations import backup_postgres, restore_postgres
from adaptive_llm.storage.postgres_stores import PostgresOutboxStore
from adaptive_llm.storage.retention import sweep
from adaptive_llm.storage.sqlite import (
    SQLiteDatabase,
    SQLiteMetadataStore,
    SQLitePayloadStore,
    control_database,
)


def main(argv: list[str] | None = None, *, sink: EventSink | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "migrate",
            "retention-sweep",
            "dispatch-once",
            "dead-letters",
            "redeliver",
            "rotate-key",
            "backup",
            "restore",
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    parser.add_argument(
        "--environment", choices=("local", "development", "staging", "production"), default="local"
    )
    parser.add_argument(
        "--storage-backend",
        choices=("sqlite", "postgres"),
        default=os.environ.get("STORAGE_BACKEND", "sqlite"),
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--tenant")
    parser.add_argument("--event-id")
    parser.add_argument("--new-key-version")
    parser.add_argument("--keyring", type=Path, default=os.environ.get("PAYLOAD_KEYRING"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ("retention-sweep", "dead-letters", "rotate-key") and not args.tenant:
        parser.error("command requires --tenant")
    if args.command == "redeliver" and not args.event_id:
        parser.error("redeliver requires --event-id")
    if args.command in ("backup", "restore") and args.out is None:
        parser.error("command requires --out")
    if args.command == "rotate-key" and (not args.new_key_version or args.keyring is None):
        parser.error("rotate-key requires --new-key-version and --keyring")
    try:
        if args.storage_backend == "postgres" and not args.database_url:
            raise StorageError("database_url_required")
        if args.command == "restore":
            if args.storage_backend == "postgres":
                restore_postgres(args.out, args.database_url, args.environment, force=args.force)
            else:
                restore_pair(args.out, args.data_dir, args.environment, force=args.force)
            print("database_restored")
            return
        database: Database = (
            PostgresDatabase(args.database_url, args.environment)
            if args.storage_backend == "postgres"
            else SQLiteDatabase(args.data_dir, cast(Environment, args.environment))
        )
        try:
            outbox = (
                PostgresOutboxStore(database)
                if isinstance(database, PostgresDatabase)
                else SQLiteOutboxStore(database)
            )
            if args.command == "retention-sweep":
                count = sweep(
                    SQLiteMetadataStore(database), SQLitePayloadStore(database), args.tenant
                )
                print(f"expired_interactions={count}")
            elif args.command == "dispatch-once":
                dispatcher = Dispatcher(
                    outbox, sink if sink is not None else InMemoryEventSink(), InProcessMetrics()
                )
                count = dispatcher.dispatch_once()
                stats = outbox.stats(now())
                print(f"processed={count} pending={stats.pending} dead={stats.dead}")
            elif args.command == "dead-letters":
                for event_id, event_type, code in outbox.dead_letters(args.tenant):
                    print(f"{event_id} {event_type} {code}")
            elif args.command == "redeliver":
                print(f"redelivered={int(outbox.redeliver(args.event_id, now()))}")
            elif args.command == "rotate-key":
                keys, current = load_keyring(args.keyring)
                if current != args.new_key_version:
                    raise StorageError("rotation_requires_current_key")
                count = rotate_key(
                    database, args.tenant, PayloadCipher(keys, current), batch_size=args.batch_size
                )
                print(f"rotated_payloads={count}")
            elif args.command == "backup":
                control: Database = (
                    PostgresDatabase(args.database_url, args.environment, role="control")
                    if args.storage_backend == "postgres"
                    else control_database(args.data_dir, cast(Environment, args.environment))
                )
                try:
                    if isinstance(database, PostgresDatabase) and isinstance(
                        control, PostgresDatabase
                    ):
                        backup_postgres(database, control, args.out)
                    else:
                        assert isinstance(database, SQLiteDatabase) and isinstance(
                            control, SQLiteDatabase
                        )
                        backup_pair(database, control, args.out)
                finally:
                    control.close()
                print("backup_complete")
            else:
                control = (
                    PostgresDatabase(args.database_url, args.environment, role="control")
                    if args.storage_backend == "postgres"
                    else control_database(args.data_dir, cast(Environment, args.environment))
                )
                control.close()
                print("migrations_applied")
        finally:
            database.close()
    except StorageError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
