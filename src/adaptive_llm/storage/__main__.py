"""Local migration and tenant-scoped retention commands."""

import argparse
from pathlib import Path
from typing import cast

from adaptive_llm.contracts import Environment
from adaptive_llm.storage.retention import sweep
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate", "retention-sweep"))
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    parser.add_argument(
        "--environment", choices=("local", "development", "staging", "production"), default="local"
    )
    parser.add_argument("--tenant")
    args = parser.parse_args(argv)
    if args.command == "retention-sweep" and args.tenant is None:
        parser.error("retention-sweep requires --tenant")
    database = SQLiteDatabase(
        args.data_dir,
        cast(Environment, args.environment),
        migrate_on_startup=args.command == "migrate",
    )
    try:
        if args.command == "retention-sweep":
            count = sweep(SQLiteMetadataStore(database), SQLitePayloadStore(database), args.tenant)
            print(f"expired_interactions={count}")
        else:
            print("migrations_applied")
    finally:
        database.close()


if __name__ == "__main__":
    main()
