"""Utilitaires de mesure neutre de l'empreinte de stockage (pages SQLite et fichiers)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageStats:
    """Mesure neutre de l'empreinte disque et de la rétention."""

    active_bytes: int  # (page_count - freelist_count) * page_size pour SQLite, ou taille fichiers actifs
    freelist_bytes: int  # freelist_count * page_size pour SQLite
    raw_disk_bytes: int  # Taille totale physique des fichiers sur disque
    active_items_count: int  # Nombre d'enregistrements/lignes actifs
    archived_items_count: int  # Nombre d'enregistrements/lignes archivés
    notes: str = ""  # Précisions techniques (ex: réservation vec0)
    details: dict[str, Any] = field(default_factory=dict)


def get_sqlite_storage_stats(
    db_paths: list[Path],
    active_count_queries: list[tuple[Path, str]],
    archived_count_queries: list[tuple[Path, str]] | None = None,
    archive_files: list[Path] | None = None,
    notes: str = "",
    details: dict[str, Any] | None = None,
) -> StorageStats:
    """Calcule l'empreinte SQLite active en excluant les pages libres (freelist)."""
    total_active_bytes = 0
    total_freelist_bytes = 0
    total_raw_disk_bytes = 0
    active_items = 0
    archived_items = 0

    for db_path in db_paths:
        if not db_path.exists():
            continue
        total_raw_disk_bytes += db_path.stat().st_size
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                page_count = conn.execute("PRAGMA page_count").fetchone()[0]
                freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                active_pages = max(0, page_count - freelist_count)
                total_active_bytes += active_pages * page_size
                total_freelist_bytes += freelist_count * page_size
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            total_active_bytes += db_path.stat().st_size

    for db_path, query in active_count_queries:
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                count = conn.execute(query).fetchone()[0]
                active_items += int(count)
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            pass

    if archived_count_queries:
        for db_path, query in archived_count_queries:
            if not db_path.exists():
                continue
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    count = conn.execute(query).fetchone()[0]
                    archived_items += int(count)
                finally:
                    conn.close()
            except Exception:  # noqa: BLE001
                pass

    if archive_files:
        for af in archive_files:
            if af.exists():
                total_raw_disk_bytes += af.stat().st_size
                try:
                    with af.open("r", encoding="utf-8") as f:
                        archived_items += sum(1 for line in f if line.strip())
                except Exception:  # noqa: BLE001
                    pass

    return StorageStats(
        active_bytes=total_active_bytes,
        freelist_bytes=total_freelist_bytes,
        raw_disk_bytes=total_raw_disk_bytes,
        active_items_count=active_items,
        archived_items_count=archived_items,
        notes=notes,
        details=details or {},
    )
