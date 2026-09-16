"""Scénario 4 : Simulation Longue Durée (Decay Biologique, Oubli & Archivage JSONL).

Mesure :
1. Décroissance différentielle de la saillance (low: 0.2, med: 0.5, high: 0.9) sur 365j.
2. Déclenchement de l'archivage automatique vers JSONL (règle 90j / seuil < 0.1).
3. Préservation des souvenirs à forte charge cognitive.
4. Stabilisation de l'empreinte disque au cours du temps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select

from benchmarks.config import BenchConfig
from benchmarks.harness import BenchmarkHarness
from mnemos.clock import FixedClock
from mnemos.models.episodic import Episode


@dataclass
class DecaySnapshot:
    day: int
    low_salience_avg_decay: float
    med_salience_avg_decay: float
    high_salience_avg_decay: float
    active_episodes: int
    archived_episodes: int


@dataclass
class DecayReport:
    snapshots: list[DecaySnapshot] = field(default_factory=list)
    dumped_jsonl_count: int = 0
    high_salience_survival_rate: float = 0.0
    low_salience_survival_rate: float = 0.0


async def run_decay_benchmark(config: BenchConfig) -> DecayReport:
    start_ms = 1_750_000_000_000
    clock = FixedClock(start_ms=start_ms)

    harness = BenchmarkHarness(config)
    await harness.setup(reset=True, custom_clock=clock)
    assert harness.episodic_store is not None
    store = harness.episodic_store

    report = DecayReport()

    # Ingestion initiale de 3 cohortes à Day 0
    low_ids: list[str] = []
    for i in range(25):
        ep = await store.write(f"Météo du jour #{i} : nuageux avec quelques éclaircies", "user")
        await store.update_salience(
            ep.id,
            {"surprise": 0.1, "arousal": 0.1, "self_ref": 0.1, "recurrence": 0.0, "combined": 0.2}
        )
        low_ids.append(ep.id)

    med_ids: list[str] = []
    for i in range(25):
        ep = await store.write(f"Réunion d'équipe #{i} sur les sprints bi-mensuels", "user")
        await store.update_salience(
            ep.id,
            {"surprise": 0.4, "arousal": 0.3, "self_ref": 0.5, "recurrence": 0.0, "combined": 0.5}
        )
        med_ids.append(ep.id)

    high_ids: list[str] = []
    for i in range(25):
        ep = await store.write(f"Événement de vie capital #{i} : projet Mnemos validé", "user")
        await store.update_salience(
            ep.id,
            {"surprise": 0.9, "arousal": 0.9, "self_ref": 0.9, "recurrence": 0.2, "combined": 0.9}
        )
        high_ids.append(ep.id)

    days_milestones = [0, 15, 30, 60, 90, 180, 365]
    prev_day = 0

    for day in days_milestones:
        delta_days = day - prev_day
        if delta_days > 0:
            clock.advance(delta_days * 86_400_000)
            prev_day = day

        if day > 0:
            await store.apply_decay()
            await store.archive_old()

        async with store._sessions() as session:
            rows = (await session.execute(select(Episode))).scalars().all()
            lookup = {e.id: e for e in rows}

        def compute_avg_decay(ids: list[str], mapping: dict[str, Episode] = lookup) -> float:
            values = [mapping[eid].decay_state for eid in ids if eid in mapping]
            return sum(values) / len(values) if values else 0.0

        active_count = sum(1 for e in lookup.values() if e.archived == 0)
        archived_count = sum(1 for e in lookup.values() if e.archived == 1)

        report.snapshots.append(
            DecaySnapshot(
                day=day,
                low_salience_avg_decay=compute_avg_decay(low_ids),
                med_salience_avg_decay=compute_avg_decay(med_ids),
                high_salience_avg_decay=compute_avg_decay(high_ids),
                active_episodes=active_count,
                archived_episodes=archived_count,
            )
        )

    dump_report = await store.dump_archived()
    report.dumped_jsonl_count = dump_report.dumped

    async with store._sessions() as session:
        final_active = (
            await session.execute(select(Episode.id).where(Episode.archived == 0))
        ).scalars().all()
        active_set = set(final_active)

    report.high_salience_survival_rate = (
        sum(1 for eid in high_ids if eid in active_set) / len(high_ids)
    ) * 100.0
    report.low_salience_survival_rate = (
        sum(1 for eid in low_ids if eid in active_set) / len(low_ids)
    ) * 100.0

    await harness.teardown()
    return report
