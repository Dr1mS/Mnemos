"""Procedural store (§12) — filesystem, lecture seule pour le MVP.

Pas de DB. Un dossier par skill (skill.py + meta.json) + _registry.json
comme index global. Enregistrement MANUEL uniquement — l'auto-amélioration
Voyager-style est explicitement hors scope (Phase 4+ du produit, §12).

data/procedural/
├── send_email_with_attachment/
│   ├── skill.py
│   └── meta.json     # {desc, signature, success_rate, last_used_at, version}
└── _registry.json    # index global
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mnemos.clock import Clock
from mnemos.logging import get_logger

logger = get_logger(__name__)

REGISTRY_FILE = "_registry.json"


@dataclass(frozen=True)
class SkillMeta:
    name: str
    desc: str
    signature: str
    success_rate: float = 1.0
    last_used_at: int | None = None
    version: int = 1
    uses: int = 0


@dataclass(frozen=True)
class Skill:
    meta: SkillMeta
    code: str


class ProceduralStore:
    def __init__(self, root: Path, clock: Clock) -> None:
        self._root = root
        self._clock = clock
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Registry ─────────────────────────────────────────────────────────────

    def _registry_path(self) -> Path:
        return self._root / REGISTRY_FILE

    def _load_registry(self) -> dict[str, dict[str, object]]:
        path = self._registry_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("procedural_registry_corrupt", error=str(exc))
            return {}

    def _save_registry(self, registry: dict[str, dict[str, object]]) -> None:
        self._registry_path().write_text(
            json.dumps(registry, indent=2, ensure_ascii=False)
        )

    # ── API (§12) ─────────────────────────────────────────────────────────────

    def list_skills(self) -> list[SkillMeta]:
        return [SkillMeta(**meta) for meta in self._load_registry().values()]  # type: ignore[arg-type]

    def get_skill(self, name: str) -> Skill | None:
        registry = self._load_registry()
        if name not in registry:
            return None
        code_path = self._root / name / "skill.py"
        if not code_path.exists():
            logger.error("procedural_skill_code_missing", name=name)
            return None
        return Skill(
            meta=SkillMeta(**registry[name]),  # type: ignore[arg-type]
            code=code_path.read_text(),
        )

    def register_skill(self, name: str, code: str, meta: SkillMeta) -> None:
        """Enregistrement manuel (CLI/admin). Écrase la version précédente."""
        if not name.replace("_", "").isalnum():
            raise ValueError(f"nom de skill invalide : {name!r}")
        skill_dir = self._root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.py").write_text(code)
        (skill_dir / "meta.json").write_text(
            json.dumps(asdict(meta), indent=2, ensure_ascii=False)
        )
        registry = self._load_registry()
        registry[name] = asdict(meta)
        self._save_registry(registry)
        logger.info("skill_registered", name=name, version=meta.version)

    def update_stats(self, name: str, success: bool) -> None:
        registry = self._load_registry()
        if name not in registry:
            logger.warning("skill_stats_unknown", name=name)
            return
        meta = SkillMeta(**registry[name])  # type: ignore[arg-type]
        # Moyenne incrémentale : rate' = (rate*uses + success) / (uses+1)
        new_rate = round(
            (meta.success_rate * meta.uses + (1.0 if success else 0.0)) / (meta.uses + 1), 4
        )
        registry[name] = asdict(
            replace(
                meta,
                success_rate=new_rate,
                uses=meta.uses + 1,
                last_used_at=self._clock.now_ms(),
            )
        )
        self._save_registry(registry)

    def search(self, query: str, k: int = 5) -> list[SkillMeta]:
        """Match naïf par mots-clés sur nom + desc — best-effort du router."""
        tokens = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for meta in self.list_skills():
            haystack = f"{meta.name} {meta.desc}".lower()
            hits = sum(1 for t in tokens if t in haystack)
            if hits:
                scored.append((hits, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored[:k]]
