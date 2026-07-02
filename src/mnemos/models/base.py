"""Base ORM + fabrique d'engine async SQLite avec sqlite-vec chargé.

Chaque connexion charge l'extension sqlite-vec (tables vec0) et active
WAL + synchronous=NORMAL (§9.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def make_async_engine(db_path: Path | str) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        # dbapi_conn est l'adaptateur sync de SQLAlchemy autour d'aiosqlite ;
        # run_async donne accès à la connexion driver pour charger l'extension.
        dbapi_conn.run_async(lambda c: c.enable_load_extension(True))
        dbapi_conn.run_async(lambda c: c.load_extension(sqlite_vec.loadable_path()))
        dbapi_conn.run_async(lambda c: c.enable_load_extension(False))
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine
