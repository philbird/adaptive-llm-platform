"""SQL unit-of-work boundary shared by tenant and control repositories."""

import sqlite3
from contextlib import AbstractContextManager
from threading import RLock
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from adaptive_llm.storage.postgres import PostgresConnection


class Database(Protocol):
    @property
    def lock(self) -> RLock: ...
    @property
    def connection(self) -> "sqlite3.Connection | PostgresConnection": ...
    def transaction(self, *, read_only: bool = False) -> AbstractContextManager[None]: ...
    def writable(self, tenant_id: str, interaction_id: str) -> None: ...
    def close(self) -> None: ...
