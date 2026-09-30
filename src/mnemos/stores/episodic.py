"""Episodic store (§9) — write, recherche hybride, decay, archivage.

Recherche (§9.2) : KNN dense top-50 (vec0, cosine) → filtres Python
(session, fenêtre, archived, salience) → re-rank hybride
`0.7*dense + 0.3*sparse + 0.1*récence`, poids normalisés à somme 1 → top-k.

Deux points du re-rank, mesurés le 29/09/2026 : la récence se calcule à
l'horloge murale et reste donc inerte sur un corpus rejoué — la rendre
active a été mesuré et rejeté (voir `search`) ; et la composante lexicale
est un **recouvrement** — la part des jetons de la requête présents dans
l'épisode — et non plus une similarité de Hamming, qui pénalisait la longueur
et se laissait départager par des bits de date. Voir
`embeddings.sparse.query_coverage`.

Décroissance (§9.2) : elapsed depuis COALESCE(last_decayed_at, created_at),
JAMAIS depuis created_at seul (double-comptage → décroissance quadratique).
Temps via Clock injectable (§6).

Multi-tenant (Lot 1 / P1) : chaque méthode prend un `tenant` (défaut
DEFAULT_TENANT). write pose le tenant ; search/list/decay/archive/counts
filtrent dessus. episodes_vec n'a pas de colonne tenant → le tenant est
filtré côté Python sur l'épisode joint (KNN cross-tenant réduit, mais le
résultat retourné est strictement du tenant demandé). Les hooks par
episode_id (mark_consolidated, set_entity_refs, update_salience) restent
sûrs : un id ULID est unique tous tenants confondus.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import CursorResult, bindparam, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from ulid import ULID

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.embeddings.sparse import query_coverage, sparse_encode
from mnemos.logging import get_logger
from mnemos.models.episodic import Episode, EpisodeSparse, ProcessedRequest
from mnemos.tagger.salience import SalienceScores
from mnemos.tenancy import DEFAULT_TENANT


class DuplicateRequest(Exception):
    """Ce `request_id` a déjà été écrit pour ce tenant (rejeu de la plateforme).

    L'appelant doit répondre un succès sans réécrire : le contrat AML exige que
    les rejeux ne créent pas de doublon."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"request_id déjà traité : {request_id}")
        self.request_id = request_id

logger = get_logger(__name__)

KNN_CANDIDATES = 50
DAY_MS = 86_400_000

# Pondérations du score hybride (§8.2, documentées au README). Elles sommaient
# à 1,1, ce qui laissait le score dépasser 1 et le rendait incomparable au score
# des faits (`1 - cosine` ∈ [0,1]) au moment de la fusion dans /search. La
# division par leur somme est un changement d'échelle uniforme : elle ne modifie
# aucun classement entre épisodes, elle rend seulement l'échelle lisible.
_W_DENSE, _W_SPARSE, _W_RECENCY = 0.7, 0.3, 0.1
_W_SUM = _W_DENSE + _W_SPARSE + _W_RECENCY
DENSE_WEIGHT = _W_DENSE / _W_SUM
SPARSE_WEIGHT = _W_SPARSE / _W_SUM
RECENCY_WEIGHT = _W_RECENCY / _W_SUM
RECENCY_HALF_LIFE_DAYS = 30.0

# Oubli (axe 3) : échos antérieurs de l'assistant effacés avec une consigne.
# Valeurs mesurées puis validées sur des personas jamais vues — voir
# `EpisodicStore._echos_a_oublier`. Ne pas les retoucher sans remesurer avec
# bench/bench_forget_targeting.py : à 0,60 les premiers dégâts apparaissent.
FORGET_ECHO_KNN = 50
FORGET_ECHO_MIN_COSINE = 0.65
FORGET_ECHO_MAX = 5


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosinus explicite. Aucun client d'embedding ne garantit des vecteurs
    normés : un produit scalaire nu donnerait des valeurs incomparables au
    cosinus que sqlite-vec calcule pour les épisodes déjà indexés, et le seuil
    calibré sur l'un ne vaudrait rien sur l'autre — silencieusement."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


@dataclass(frozen=True)
class ForgetRequest:
    """Une consigne d'oubli portée par le lot écrit (voir `write_batch`)."""

    target: str    # ce qu'il faut oublier, dans les mots de l'utilisateur
    position: int  # nombre d'épisodes du lot, après filtrage, qui précèdent la consigne
    echo_min_cosine: float = FORGET_ECHO_MIN_COSINE
    echo_max: int = FORGET_ECHO_MAX


@dataclass(frozen=True)
class ScoredEpisode:
    episode: Episode
    score: float
    dense_sim: float
    sparse_sim: float
    recency: float


@dataclass(frozen=True)
class DecayReport:
    scanned: int
    dry_run: bool
    now_ms: int


@dataclass(frozen=True)
class ArchiveReport:
    decayed_consolidated: int  # règle 1 §9.2
    expired_unconsolidated: int  # règle 2 §9.2
    dry_run: bool


@dataclass(frozen=True)
class BatchEpisodeItem:
    content: str
    role: str
    session_id: str | None = None
    salience_scores: SalienceScores | None = None
    tenant: str = DEFAULT_TENANT
    created_at: int | None = None


@dataclass(frozen=True)
class ArchiveDumpReport:
    dumped: int
    path: str | None


class EpisodicStore:
    def __init__(
        self,
        engine: AsyncEngine,
        embedder: DenseEmbedder,
        clock: Clock,
        settings: Settings,
    ) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._embedder = embedder
        self._clock = clock
        self._settings = settings

    # ── Write path (partie synchrone du §13.3) ───────────────────────────────

    async def write(
        self,
        content: str,
        role: str,
        session_id: str | None = None,
        salience_scores: SalienceScores | None = None,
        tenant: str = DEFAULT_TENANT,
        created_at: int | None = None,
    ) -> Episode:
        items = [
            BatchEpisodeItem(
                content=content,
                role=role,
                session_id=session_id,
                salience_scores=salience_scores,
                tenant=tenant,
                created_at=created_at,
            )
        ]
        episodes = await self.write_batch(items)
        episode = episodes[0]
        logger.info("episode_written", episode_id=episode.id, session_id=session_id, role=role)
        return episode

    async def write_batch(
        self,
        items: list[BatchEpisodeItem],
        request_id: str | None = None,
        forget: list[ForgetRequest] | None = None,
        tenant: str | None = None,
    ) -> list[Episode]:
        """Écrit un lot d'épisodes de manière synchrone et atomique (§13.3).

        Effectue un batch embedding unique via le DenseEmbedder (un seul appel
        HTTP groupé) et insère tous les épisodes, bits épars et vecteurs au sein
        d'une seule transaction SQLite (session.begin()).

        `request_id` rend l'écriture idempotente : il est inscrit dans la même
        transaction, et un rejeu lève `DuplicateRequest` au lieu de dupliquer.

        `forget` : consignes d'oubli portées par ce lot (axe 3). La consigne
        elle-même est écrite normalement ; ses échos ANTÉRIEURS par l'assistant
        sont effacés dans la MÊME transaction — physiquement, pas marqués :
        oublier doit oublier. Les échos situés plus tôt dans le lot lui-même ne
        sont simplement pas insérés.
        """
        forget = forget or []
        if not items and not forget:
            return []
        tenant_ = items[0].tenant if items else (tenant or DEFAULT_TENANT)

        # Les cibles d'oubli voyagent dans le même appel d'embedding que le lot.
        texts = [it.content for it in items]
        vecteurs = await self._embedder.embed_batch(texts + [f.target for f in forget])
        dense_vectors, cibles = vecteurs[: len(texts)], vecteurs[len(texts):]
        exclus, a_effacer = await self._echos_a_oublier(
            tenant_, items, dense_vectors, forget, cibles
        )

        records: list[tuple[Episode, EpisodeSparse, dict[str, Any]]] = []
        now_default = self._clock.now_ms()

        for j, (it, dense) in enumerate(zip(items, dense_vectors, strict=True)):
            if j in exclus:
                continue
            now = it.created_at if it.created_at is not None else now_default
            sparse = sparse_encode(it.content, now)
            ep = Episode(
                id=str(ULID()),
                tenant=it.tenant,
                created_at=now,
                session_id=it.session_id,
                role=it.role,
                content=it.content,
                **(
                    {
                        "salience": it.salience_scores["combined"],
                        "surprise": it.salience_scores["surprise"],
                        "arousal": it.salience_scores["arousal"],
                        "self_ref": it.salience_scores["self_ref"],
                        "recurrence": it.salience_scores["recurrence"],
                    }
                    if it.salience_scores is not None
                    else {}
                ),
            )
            sp = EpisodeSparse(episode_id=ep.id, sparse_bits=sparse)
            vec_param = {
                "id": ep.id,
                "tenant": it.tenant,
                "emb": sqlite_vec.serialize_float32(dense),
            }
            records.append((ep, sp, vec_param))

        try:
            async with self._sessions() as session, session.begin():
                session.add_all([r[0] for r in records])
                session.add_all([r[1] for r in records])
                for _, _, vec_param in records:
                    await session.execute(
                        text(
                            "INSERT INTO episodes_vec(episode_id, tenant, embedding) "
                            "VALUES (:id, :tenant, :emb)"
                        ),
                        vec_param,
                    )
                if a_effacer:
                    # Le vecteur d'abord : un épisode effacé mais encore indexé
                    # occuperait une place dans le KNN et ferait fuiter son contenu.
                    params = {"ids": list(a_effacer), "tenant": tenant_}
                    await session.execute(
                        text("DELETE FROM episodes_vec WHERE episode_id IN :ids")
                        .bindparams(bindparam("ids", expanding=True)), params)
                    await session.execute(
                        text("DELETE FROM episodes_sparse WHERE episode_id IN :ids")
                        .bindparams(bindparam("ids", expanding=True)), params)
                    await session.execute(
                        text("DELETE FROM episodes WHERE id IN :ids AND tenant = :tenant")
                        .bindparams(bindparam("ids", expanding=True)), params)
                if request_id is not None:
                    # Dans la MÊME transaction que les épisodes : soit les deux
                    # atterrissent, soit aucun. Un registre écrit après coup
                    # laisserait une fenêtre où un rejeu dupliquerait.
                    session.add(
                        ProcessedRequest(
                            tenant=tenant_,
                            request_id=request_id,
                            created_at=now_default,
                            episode_count=len(records),
                        )
                    )
        except IntegrityError as exc:
            # Course entre deux rejeux du même request_id : le perdant voit la
            # violation de clé primaire. L'écriture gagnante a eu lieu, donc
            # c'est un succès du point de vue de l'appelant.
            if request_id is None:
                raise
            logger.info("write_batch_rejeu_ignore", tenant=tenant_, request_id=request_id)
            raise DuplicateRequest(request_id) from exc

        episodes = [r[0] for r in records]
        if exclus or a_effacer:
            # Des comptes, jamais de contenu : c'est précisément ce qu'on oublie.
            logger.info("forget_echos_effaces", tenant=tenant_,
                        existants=len(a_effacer), dans_le_lot=len(exclus))
        logger.info("episodes_batch_written", count=len(episodes), tenant=tenant_)
        return episodes

    async def _echos_a_oublier(
        self,
        tenant: str,
        items: list[BatchEpisodeItem],
        vecteurs_lot: list[list[float]],
        forget: list[ForgetRequest],
        cibles: list[list[float]],
    ) -> tuple[set[int], set[str]]:
        """Échos antérieurs de l'assistant à oublier, pour chaque consigne.

        Règle mesurée et validée le 29/09/2026 (bench/bench_forget_targeting.py,
        calibration 47 personas puis validation 53 personas jamais vues) :
        parmi les 50 plus proches voisins de la cible, les messages ASSISTANT
        antérieurs à la consigne, 5 au plus, de cosinus ≥ 0,65. En validation,
        la préférence oubliée remonte dans le top 10 pour 16 % des questions au
        lieu de 33 %, sans qu'aucun message-preuve d'une autre question soit
        touché ; à 0,60 les premiers dégâts apparaissent, dans les deux moitiés.

        Seul l'assistant : relus à la main, les échos de l'assistant appliquent
        la préférence (« Since you enjoy visiting aquariums… ») ; les messages
        de l'utilisateur au même seuil sont des QUESTIONS sur le sujet (« Can
        you suggest some good books? »), qu'il ne faut pas effacer. Et les
        messages postérieurs à la consigne ne sont jamais visés : une
        re-déclaration de l'utilisateur doit survivre.

        Rend (positions du lot à ne pas insérer, épisodes existants à effacer)."""
        exclus: set[int] = set()
        a_effacer: set[str] = set()
        for demande, cible in zip(forget, cibles, strict=True):
            candidats: list[tuple[float, int | str]] = [
                (cos, eid) for eid, cos in await self._voisins_assistant(tenant, cible)
            ]
            candidats += [
                (_cosine(cible, vecteurs_lot[j]), j)
                for j in range(min(demande.position, len(items)))
                if items[j].role == "assistant"
            ]
            candidats.sort(key=lambda c: c[0], reverse=True)
            for cos, ref in candidats[: demande.echo_max]:
                if cos < demande.echo_min_cosine:
                    break
                if isinstance(ref, int):
                    exclus.add(ref)
                else:
                    a_effacer.add(ref)
        return exclus, a_effacer

    async def _voisins_assistant(self, tenant: str, cible: list[float]) -> list[tuple[str, float]]:
        """Messages ASSISTANT existants parmi les voisins de la cible : (id, cosinus).

        Tous les épisodes déjà stockés sont antérieurs au lot en cours, donc
        antérieurs à ses consignes."""
        async with self._sessions() as session:
            knn = await session.execute(
                text("SELECT episode_id, distance FROM episodes_vec "
                     "WHERE embedding MATCH :emb AND k = :k AND tenant = :tenant"),
                {"emb": sqlite_vec.serialize_float32(cible), "k": FORGET_ECHO_KNN,
                 "tenant": tenant},
            )
            distances = {row[0]: float(row[1]) for row in knn}
            if not distances:
                return []
            roles = (await session.execute(
                select(Episode.id, Episode.role).where(
                    Episode.id.in_(distances), Episode.tenant == tenant,
                    Episode.archived == 0)
            )).all()
        return [(eid, 1.0 - distances[eid]) for eid, role in roles if role == "assistant"]

    async def update_salience(self, episode_id: str, scores: SalienceScores) -> None:
        """Mise à jour asynchrone post-scoring (§13.3) — hors write path."""
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(
                    salience=scores["combined"],
                    surprise=scores["surprise"],
                    arousal=scores["arousal"],
                    self_ref=scores["self_ref"],
                    recurrence=scores["recurrence"],
                )
            )

    # ── Read path ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        time_window: tuple[datetime, datetime] | None = None,
        min_salience: float = 0.0,
        tenant: str = DEFAULT_TENANT,
    ) -> list[ScoredEpisode]:
        now = self._clock.now_ms()
        dense = await self._embedder.embed(query)
        query_sparse = sparse_encode(query, now)

        knn_candidates = max(k * 2, KNN_CANDIDATES)
        async with self._sessions() as session:
            knn = await session.execute(
                text(
                    "SELECT episode_id, distance FROM episodes_vec "
                    "WHERE embedding MATCH :emb AND k = :k AND tenant = :tenant"
                ),
                {
                    "emb": sqlite_vec.serialize_float32(dense),
                    "k": knn_candidates,
                    "tenant": tenant,
                },
            )
            distances = {row[0]: float(row[1]) for row in knn}
            if not distances:
                return []
            episodes = (
                (
                    await session.execute(
                        select(Episode, EpisodeSparse.sparse_bits)
                        .join(EpisodeSparse, EpisodeSparse.episode_id == Episode.id)
                        .where(Episode.id.in_(distances))
                    )
                )
                .tuples()
                .all()
            )

        # Filtres Python (§9.2 étape 3). episodes_vec (vec0) est partitionné par
        # tenant à la source. Les filtres suivants éliminent archivés, salience
        # sous le seuil, sessions et fenêtres temporelles.
        retenus: list[tuple[Episode, bytes]] = []
        for episode, sparse_bits in episodes:
            if episode.tenant != tenant:
                continue
            if episode.archived:
                continue
            if episode.salience < min_salience:
                continue
            if session_id is not None and episode.session_id != session_id:
                continue
            if time_window is not None:
                start_ms = int(time_window[0].timestamp() * 1000)
                end_ms = int(time_window[1].timestamp() * 1000)
                if not start_ms <= episode.created_at <= end_ms:
                    continue
            retenus.append((episode, sparse_bits))

        if not retenus:
            return []

        # Récence : l'âge se mesure à l'horloge murale, pas au corpus.
        #
        # Conséquence assumée : sur un corpus rejoué (données datées de 2023
        # interrogées en 2026), `2 ** (-1162/30)` vaut 2e-12 et le terme est
        # INERTE. C'est le cas des évaluations AML.
        #
        # L'alternative évidente — rapporter l'âge au plus récent des candidats
        # pour le rendre actif — a été mesurée le 29/09/2026 et REJETÉE
        # (bench/bench_rerank_variants.py, bench/bench_locomo_qa.py) : elle
        # améliore un jeu fabriqué de bascule de valeur (le plus ancien devant
        # l'actif 11/13 → 8/13) mais recule la meilleure preuve LoCoMo sur 104
        # questions sur 150 (hit@10 0,607 → 0,373). Un poids réduit (0,005 à
        # 0,04) ou une demi-vie allongée (180, 365 j) n'améliorent D1 dans aucun
        # cas sans coûter sur LoCoMo. Sur une conversation de plusieurs mois,
        # une récence vivante écrase la similarité. Ne pas la réactiver sans
        # remesurer sur ces deux jeux.
        scored: list[ScoredEpisode] = []
        for episode, sparse_bits in retenus:
            dense_sim = 1.0 - distances[episode.id]  # distance cosine → similarité
            sparse_sim = query_coverage(query_sparse, sparse_bits)
            age_days = max(0.0, (now - episode.created_at) / DAY_MS)
            recency = 2.0 ** (-age_days / RECENCY_HALF_LIFE_DAYS)
            score = (
                DENSE_WEIGHT * dense_sim
                + SPARSE_WEIGHT * sparse_sim
                + RECENCY_WEIGHT * recency
            )
            scored.append(ScoredEpisode(episode, score, dense_sim, sparse_sim, recency))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def get_by_id(self, episode_id: str) -> Episode | None:
        async with self._sessions() as session:
            return await session.get(Episode, episode_id)

    async def has_request(self, tenant: str, request_id: str) -> bool:
        """Ce `request_id` a-t-il déjà été écrit pour ce tenant ?

        Vérifié *avant* le travail coûteux : sans ce raccourci, un rejeu
        recalculerait tous les embeddings du lot avant de buter sur la clé
        primaire. La contrainte reste le garde-fou en cas de course."""
        async with self._sessions() as session:
            found = await session.execute(
                select(ProcessedRequest.request_id).where(
                    ProcessedRequest.tenant == tenant,
                    ProcessedRequest.request_id == request_id,
                )
            )
            return found.first() is not None

    async def ping(self) -> str | None:
        """Sonde DB pour /health (§Santé) : exécute une vraie requête (pas un
        simple test d'existence de fichier — un .db corrompu ou verrouillé
        existe mais ne répond pas). None si OK, sinon le message d'erreur."""
        try:
            async with self._sessions() as session:
                await session.execute(text("SELECT 1 FROM episodes LIMIT 1"))
        except Exception as exc:  # noqa: BLE001 — diagnostic, jamais de crash
            return f"episodic DB inaccessible : {exc}"
        return None

    async def list_recent(
        self, session_id: str | None = None, n: int = 5, tenant: str = DEFAULT_TENANT
    ) -> list[Episode]:
        """Derniers épisodes (ordre chronologique) — historique du scoring §13.2."""
        stmt = (
            select(Episode)
            .where(Episode.tenant == tenant, Episode.archived == 0)
            .order_by(Episode.created_at.desc())
            .limit(n)
        )
        if session_id is not None:
            stmt = stmt.where(Episode.session_id == session_id)
        async with self._sessions() as session:
            rows = list((await session.execute(stmt)).scalars())
        return list(reversed(rows))

    # ── Consolidation hooks (§15) ─────────────────────────────────────────────

    async def pending_counts(self, tenant: str | None = None) -> dict[str, int]:
        """Ce qui attend l'IA locale (observabilité) : épisodes jamais scorés,
        candidats mûrs pour consolidation, saillants mais trop récents.
        tenant=None → tous tenants confondus (vue worker/globale)."""
        now = self._clock.now_ms()
        cutoff = now - int(self._settings.CONSOLIDATION_DELAY_HOURS * 3_600_000)
        threshold = self._settings.SALIENCE_THRESHOLD_CONSOLIDATE
        tenant_clause = "" if tenant is None else " AND tenant = :tenant"
        params: dict[str, object] = {"thr": threshold, "cutoff": cutoff}
        if tenant is not None:
            params["tenant"] = tenant
        async with self._sessions() as session:
            row = await session.execute(
                text(
                    f"""
                    SELECT
                      SUM(CASE WHEN surprise IS NULL THEN 1 ELSE 0 END),
                      SUM(CASE WHEN surprise IS NOT NULL AND consolidated_at IS NULL
                               AND salience > :thr AND created_at <= :cutoff
                               THEN 1 ELSE 0 END),
                      SUM(CASE WHEN surprise IS NOT NULL AND consolidated_at IS NULL
                               AND salience > :thr AND created_at > :cutoff
                               THEN 1 ELSE 0 END)
                    FROM episodes WHERE archived = 0{tenant_clause}
                    """  # noqa: S608 — tenant_clause est un littéral fixe, pas de l'input
                ),
                params,
            )
            unscored, ready, too_recent = row.one()
        return {
            "unscored": int(unscored or 0),
            "consolidation_ready": int(ready or 0),
            "consolidation_waiting": int(too_recent or 0),
        }

    async def list_unscored(self, limit: int = 50, tenant: str | None = None) -> list[Episode]:
        """Épisodes jamais passés au scoring de salience (surprise IS NULL,
        §5.1) — jobs perdus quand le process meurt avant que la queue draine.
        Rattrapés par le worker (§15.1). tenant=None → tous tenants."""
        stmt = (
            select(Episode)
            .where(Episode.surprise.is_(None), Episode.archived == 0)
            .order_by(Episode.created_at)
            .limit(limit)
        )
        if tenant is not None:
            stmt = stmt.where(Episode.tenant == tenant)
        async with self._sessions() as session:
            rows = await session.execute(stmt)
            return list(rows.scalars())

    async def list_pending_consolidation(
        self,
        min_salience: float,
        min_age_hours: float,
        limit: int,
        tenant: str | None = None,
    ) -> list[Episode]:
        """Candidats à consolidation. tenant=None → tous tenants (le worker lit
        episode.tenant par épisode pour écrire les faits dans le bon tenant)."""
        cutoff = self._clock.now_ms() - int(min_age_hours * 3_600_000)
        stmt = (
            select(Episode)
            .where(
                Episode.consolidated_at.is_(None),
                Episode.extraction_attempted_at.is_(None),
                Episode.archived == 0,
                Episode.salience > min_salience,
                Episode.created_at <= cutoff,
            )
            .order_by(Episode.salience.desc())
            .limit(limit)
        )
        if tenant is not None:
            stmt = stmt.where(Episode.tenant == tenant)
        async with self._sessions() as session:
            rows = await session.execute(stmt)
            return list(rows.scalars())

    async def record_extraction_attempt(self, episode_id: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(extraction_attempted_at=self._clock.now_ms())
            )

    async def mark_consolidated(self, episode_id: str, extraction_failed: bool = False) -> None:
        now = self._clock.now_ms()
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(
                    consolidated_at=now,
                    extraction_attempted_at=now,
                    extraction_failed=1 if extraction_failed else 0,
                )
            )

    async def set_entity_refs(self, episode_id: str, entity_names: list[str]) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(entity_refs=json.dumps(entity_names, ensure_ascii=False))
            )

    # ── Lifecycle (§9.2) ──────────────────────────────────────────────────────

    async def apply_decay(self, dry_run: bool = False) -> DecayReport:
        now = self._clock.now_ms()
        rate = self._settings.DECAY_RATE_DAILY
        async with self._sessions() as session, session.begin():
            if dry_run:
                count = len(
                    (
                        await session.execute(select(Episode.id).where(Episode.archived == 0))
                    ).all()
                )
                return DecayReport(scanned=count, dry_run=True, now_ms=now)
            result = await session.execute(
                text(
                    """
                    UPDATE episodes
                    SET decay_state = MAX(
                          0.0,
                          decay_state
                          - :rate
                            * ((:now - COALESCE(last_decayed_at, created_at)) / 86400000.0)
                            * (2 - salience)
                        ),
                        last_decayed_at = :now
                    WHERE archived = 0
                    """
                ),
                {"rate": rate, "now": now},
            )
            scanned = cast("CursorResult[Any]", result).rowcount or 0
        logger.info("decay_applied", scanned=scanned, now_ms=now)
        return DecayReport(scanned=scanned, dry_run=False, now_ms=now)

    async def archive_old(self, dry_run: bool = False) -> ArchiveReport:
        now = self._clock.now_ms()
        retention_cutoff = now - self._settings.EPISODIC_RETENTION_DAYS * DAY_MS
        rule1 = (
            Episode.archived == 0,
            Episode.decay_state < 0.1,
            Episode.consolidated_at.is_not(None),
            Episode.salience < self._settings.SALIENCE_THRESHOLD_CONSOLIDATE,
        )
        rule2 = (
            Episode.archived == 0,
            Episode.created_at < retention_cutoff,
            Episode.salience < self._settings.SALIENCE_THRESHOLD_CONSOLIDATE,
            Episode.consolidated_at.is_(None),
        )
        async with self._sessions() as session, session.begin():
            if dry_run:
                n1 = len((await session.execute(select(Episode.id).where(*rule1))).all())
                n2 = len((await session.execute(select(Episode.id).where(*rule2))).all())
                return ArchiveReport(n1, n2, dry_run=True)
            r1 = await session.execute(update(Episode).where(*rule1).values(archived=1))
            r2 = await session.execute(update(Episode).where(*rule2).values(archived=1))
        report = ArchiveReport(
            cast("CursorResult[Any]", r1).rowcount or 0,
            cast("CursorResult[Any]", r2).rowcount or 0,
            dry_run=False,
        )
        logger.info(
            "archive_applied",
            decayed_consolidated=report.decayed_consolidated,
            expired_unconsolidated=report.expired_unconsolidated,
        )
        return report

    async def dump_archived(self) -> ArchiveDumpReport:
        """Dump JSONL des épisodes archivés vers data/archive/YYYY-MM.jsonl
        puis DELETE (§9.2, worker mensuel). Deletes explicites sur les trois
        tables : le FK CASCADE de episodes_sparse ne fire pas (PRAGMA
        foreign_keys OFF par défaut) et episodes_vec (vec0) n'a pas de FK.
        """
        now_dt = self._clock.now_dt()
        archive_dir = self._settings.DATA_DIR / "archive"
        async with self._sessions() as session, session.begin():
            rows = list(
                (await session.execute(select(Episode).where(Episode.archived == 1))).scalars()
            )
            if not rows:
                return ArchiveDumpReport(dumped=0, path=None)
            archive_dir.mkdir(parents=True, exist_ok=True)
            path = archive_dir / f"{now_dt.year}-{now_dt.month:02d}.jsonl"
            with path.open("a") as fh:
                for ep in rows:
                    record = {
                        k: v for k, v in vars(ep).items() if not k.startswith("_")
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            ids = [ep.id for ep in rows]
            for stmt in (
                "DELETE FROM episodes_sparse WHERE episode_id IN (SELECT id FROM episodes WHERE archived = 1)",  # noqa: E501
                "DELETE FROM episodes_vec WHERE episode_id IN (SELECT id FROM episodes WHERE archived = 1)",  # noqa: E501
                "DELETE FROM episodes WHERE archived = 1",
            ):
                await session.execute(text(stmt))
        logger.info("archive_dumped", count=len(ids), path=str(path))
        return ArchiveDumpReport(dumped=len(ids), path=str(path))
