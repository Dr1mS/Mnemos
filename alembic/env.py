"""Alembic env multi-database (§18 Phase 0).

Deux bases indépendantes (episodic / semantic), chacune avec sa propre
version table (alembic_version_<db>). Sélection par `-x db=episodic` ;
sans -x db, les deux sont migrées séquentiellement.

Les tables virtuelles vec0 (episodes_vec, facts_vec) sont INVISIBLES à
l'autogenerate → leurs migrations sont écrites à la main (cf. spec §18),
et include_object les exclut de la comparaison pour éviter les DROP
intempestifs.
"""

from __future__ import annotations

from typing import Any

import sqlite_vec  # type: ignore[import-untyped]
from alembic import context
from sqlalchemy import create_engine, event, pool

from mnemos.config import get_settings

settings = get_settings()

DATABASES = {
    "episodic": f"sqlite:///{settings.EPISODIC_DB}",
    "semantic": f"sqlite:///{settings.SEMANTIC_DB}",
}

# Metadata par base — branchées en Phase 2 (episodic) et Phase 4 (semantic)
# sur mnemos.models.*. None = pas d'autogenerate possible, migrations manuelles.
TARGET_METADATA: dict[str, Any] = {
    "episodic": None,
    "semantic": None,
}

# Tables vec0 : jamais gérées par autogenerate.
VEC0_TABLES = {"episodes_vec", "facts_vec"}


def _selected_dbs() -> list[str]:
    x_args = context.get_x_argument(as_dictionary=True)
    if "db" in x_args:
        name = x_args["db"]
        if name not in DATABASES:
            raise SystemExit(f"-x db={name} inconnu (choix : {', '.join(DATABASES)})")
        return [name]
    return list(DATABASES)


def _include_object(obj: Any, name: str, type_: str, reflected: bool, compare_to: Any) -> bool:
    return not (type_ == "table" and name in VEC0_TABLES)


def run_migrations_offline() -> None:
    for db_name in _selected_dbs():
        context.configure(
            url=DATABASES[db_name],
            target_metadata=TARGET_METADATA[db_name],
            version_table=f"alembic_version_{db_name}",
            literal_binds=True,
            include_object=_include_object,
            dialect_opts={"paramstyle": "named"},
        )
        with context.begin_transaction():
            context.run_migrations(db_name=db_name)


def _load_sqlite_vec(dbapi_conn: Any, _record: Any) -> None:
    """Les migrations créent des tables vec0 → extension requise (sync sqlite3)."""
    dbapi_conn.enable_load_extension(True)
    sqlite_vec.load(dbapi_conn)
    dbapi_conn.enable_load_extension(False)


def run_migrations_online() -> None:
    for db_name in _selected_dbs():
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        engine = create_engine(DATABASES[db_name], poolclass=pool.NullPool)
        event.listens_for(engine, "connect")(_load_sqlite_vec)
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=TARGET_METADATA[db_name],
                version_table=f"alembic_version_{db_name}",
                include_object=_include_object,
            )
            with context.begin_transaction():
                context.run_migrations(db_name=db_name)
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
