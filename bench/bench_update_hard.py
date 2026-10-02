"""Catégorie D1 — mise à jour de valeur, version dure.

`bench_knowledge_update.py` donne 14/14 sur la capacité que l'évaluation externe
note **29,25**. L'écart ne vient pas de la façon de mesurer — les deux jugent une
vraie réponse produite à partir du contexte rendu — mais de la difficulté des cas.
Voir `HARD_UPDATE_INSTANCES` dans `bench/datasets.py` pour les béquilles retirées.

**Ce bench est un instrument, pas un score.** Un taux global ne dit pas où ça casse,
et c'est précisément ce qui nous a fait optimiser à l'aveugle. On enregistre donc le
**rang de chaque énoncé** dans le contexte rendu, ce qui classe chaque échec :

  * énoncé actif en tête du contexte et réponse fausse → défaut de **réponse** ;
  * énoncé actif absent ou enfoui → défaut de **classement** (axe 2) ;
  * énoncé périmé mieux classé que l'actif → défaut de **récence** (axes 2.1/2.6).

À `top_k=100` sur une trentaine d'épisodes, tout remonte : le taux seul ne mesure
donc *que* la réponse. Les rangs, eux, restent lisibles et exposent le classement.
Ne pas lire un taux élevé comme « les cas sont trop faciles » sans avoir lu les rangs.

Trois contextes comparés :
  O — le transcript complet, chronologique, sans mémoire : la **référence du
      répondeur**. Il ne mesure pas Mnemos ; il dit ce qu'un échec de A ou B
      peut légitimement signifier, et rend le bench valide même quand la VRAM
      de cette machine impose un modèle de réponse plus petit. Ce n'est PAS une
      borne supérieure stricte : mesuré le 29/09, A=8/13 contre O=7/13, parce
      qu'un contexte court et trié par pertinence aide un 3B plus qu'un vidage
      chronologique de 24 tours. Un résultat au-dessus de O n'est pas un bogue ;
  A — la recherche épisodique seule (`store.search`) ;
  B — la route AML `/search` complète (faits + épisodes fusionnés).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import drain_consolidation, setup_bench_app
from bench.datasets import HARD_UPDATE_INSTANCES, HardUpdateInstance
from bench.eval_utils import (
    check_active_fact_answer,
    mcnemar_test,
    mentions_term,
    normalize_text,
    wilson_score_interval,
)
from bench.remote_llm import RemoteChat
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

ANSWER_PROMPT = """Using only the memories below, answer the question.
The memories are listed {order}, with their date.

Memories:
{context}

Question: {question}
{hint}"""

ORDRES = {"O": "in chronological order", "A": "most relevant first",
          "B": "most relevant first"}

DAY_MS = 86_400_000

# Garde-fou : certains modèles rendent leur réflexion dans le corps de la
# réponse même avec think=False. On la retire quand elle est balisée ; quand
# elle ne l'est pas, c'est le contexte O qui s'effondre et prévient.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Du remplissage neutre, jamais du même champ sémantique que l'attribut testé :
# il donne du corps au contexte sans être un distracteur déguisé. Sans lui, une
# dizaine d'épisodes suffiraient à saturer le top_k et les rangs ne voudraient
# plus rien dire.
NEUTRAL_FILLER = [
    "Il a plu toute la semaine, je n'ai pas pu sortir le vélo.",
    "J'ai fini la série que tu m'avais conseillée, la fin est décevante.",
    "Le four a rendu l'âme hier soir, il va falloir en racheter un.",
    "J'ai croisé un ancien camarade de promo au marché ce matin.",
    "Je me suis remis à la lecture le soir, ça me vide la tête.",
    "La voiture passe au contrôle technique le mois prochain.",
]


def build_messages(inst: HardUpdateInstance, base_ms: int) -> list[dict[str, Any]]:
    """Trois valeurs successives, espacées de semaines, en terrain encombré.

    Trois différences volontaires avec le bench d'origine :

    * l'assistant n'accuse jamais réception **de la valeur** — sinon le signal
      est doublé et la tâche devient triviale ;
    * un seul distracteur par tour, dans l'ordre du jeu de données : celui qui
      cite l'ancienne valeur au passé arrive **en dernier**, là où il piège la
      réponse finale, et non juste après l'énoncé qui la pose ;
    * les valeurs sont séparées de ~3 semaines. Un mois entre la première et la
      dernière : à 6 jours d'écart, le bonus de récence (0,1) et les dates
      elles-mêmes ne portaient aucun signal ordonnable.
    """
    msgs: list[dict[str, Any]] = []
    t = base_ms
    filler = list(NEUTRAL_FILLER)

    def add(content: str, role: str, delta_days: float) -> None:
        nonlocal t
        t += int(delta_days * DAY_MS)
        msgs.append({"role": role, "content": content, "timestamp": t})

    valeurs = (inst.value_0, inst.value_1, inst.value_2)
    for rang, (gabarit, valeur) in enumerate(zip(inst.statements, valeurs, strict=True)):
        add(gabarit.format(valeur), "user", 3.0)
        # Accusé de réception neutre : il ne répète jamais la valeur.
        add("C'est noté.", "assistant", 0.001)
        if rang < len(inst.distractors):
            add(inst.distractors[rang], "user", 4.0)
            add("D'accord.", "assistant", 0.001)
        for _ in range(2):
            if filler:
                add(filler.pop(0), "user", 4.0)
                add("D'accord.", "assistant", 0.001)
    return msgs


def _iso_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date().isoformat()


def _context(items: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}. [{date}] {content}" for i, (date, content) in enumerate(items, 1))


def _ranks(ctx: list[tuple[str, str]], inst: HardUpdateInstance) -> list[int | None]:
    """Rang (1-based) de chacun des trois énoncés dans le contexte rendu.

    C'est la mesure qui rend les échecs interprétables : sans elle, un taux de
    40 % est aussi muet que le 14/14 qu'on remplace."""
    contenus = [normalize_text(c) for _, c in ctx]
    rangs: list[int | None] = []
    for gabarit, valeur in zip(inst.statements, (inst.value_0, inst.value_1, inst.value_2),
                              strict=True):
        cible = normalize_text(gabarit.format(valeur))
        rangs.append(next((i for i, c in enumerate(contenus, 1) if c == cible), None))
    return rangs


# Les modèles que Mnemos charge lui-même. Le préambule VRAM ne décharge QUE
# ceux-là : la machine est partagée avec un autre projet qui se sert d'Ollama,
# et décharger son modèle en plein travail serait une panne qu'on aurait créée.
MODELES_MNEMOS = {"bge-m3:latest", "bge-m3", "qwen2.5:3b", "qwen3.5:9b",
                  "qwen3:4b", "qwen2.5:7b-instruct-q4_K_M"}

MARGE_VRAM_MIB = 900  # cache KV + graphe de calcul à num_ctx=8192


def vram_libre_mib() -> int | None:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


async def preparer_vram(settings: Settings, keep: set[str], modele_reponse: str) -> dict[str, Any]:
    """Libère la VRAM de Mnemos, puis **vérifie** qu'il reste de quoi mesurer.

    Deux runs du 29/09 sont morts sur des 500 d'Ollama avec **718 Mo de VRAM
    libres** : `OLLAMA_KEEP_ALIVE=-1` gardait un 9B chargé pour rien. Ce bench
    veut simultanément un répondeur, un 3B extracteur et des embeddings ; sans
    ce préambule il se remet exactement dans la configuration qui a échoué.

    Deux règles : on ne décharge que nos propres modèles (`MODELES_MNEMOS`), et
    si ça ne suffit pas on **refuse de démarrer** en nommant ce qui occupe la
    carte. Un bench qui meurt à la moitié coûte plus cher qu'un bench qui ne
    démarre pas. (Si EMBED_BACKEND=llamacpp, llama-server tient sa propre VRAM,
    invisible d'Ollama : ce déchargement ne la libère pas.)"""
    rapport: dict[str, Any] = {"dechargés": [], "étrangers": [], "deja_charge": False}
    tailles: dict[str, int] = {}
    try:
        async with httpx.AsyncClient(base_url=settings.OLLAMA_HOST, timeout=60) as c:
            for m in (await c.get("/api/tags")).json().get("models", []):
                tailles[m["name"]] = int(m.get("size", 0)) // (1024 * 1024)
            for m in (await c.get("/api/ps")).json().get("models", []):
                nom = m.get("model") or m.get("name") or ""
                if not nom or nom in keep:
                    rapport["deja_charge"] |= nom == modele_reponse
                    continue
                if nom not in MODELES_MNEMOS:
                    rapport["étrangers"].append(
                        f"{nom} ({int(m.get('size_vram', 0)) // (1024 * 1024)} Mio,"
                        f" expire {m.get('expires_at', '?')[:19]})"
                    )
                    continue
                # keep_alive=0 sans prompt : décharge sans rien générer.
                await c.post("/api/generate", json={"model": nom, "keep_alive": 0})
                rapport["dechargés"].append(nom)
    except httpx.HTTPError as exc:
        rapport["erreur"] = str(exc)

    libre = vram_libre_mib()
    requis = 0 if rapport["deja_charge"] else tailles.get(modele_reponse, 0) + MARGE_VRAM_MIB
    rapport["libre_mib"] = libre
    rapport["requis_mib"] = requis
    rapport["suffisant"] = libre is None or libre >= requis
    return rapport


async def run_instance(
    inst: HardUpdateInstance,
    llm: Any,
    answer_model: str,
    top_k: int,
    user_id: str,
    llm_model: str,
    embed_backend: str,
) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_upd_"))
    try:
        app, _, settings = await setup_bench_app(
            tmp, "ollama", llm_model=llm_model, salience_workers=2,
            embed_backend=embed_backend,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t", timeout=600
            ) as client:
                messages = build_messages(inst, 1_690_000_000_000)
                for i in range(0, len(messages), 20):
                    resp = await client.post("/add", json={
                        "request_id": f"{inst.id}:chunk-{i // 20}",
                        "user_id": user_id,
                        "session_id": f"{inst.id}:session",
                        "messages": messages[i:i + 20],
                    })
                    resp.raise_for_status()
                await app.state.queue.join()
                cons = await drain_consolidation(app)

                episodes = await app.state.store.search(inst.probe, k=top_k, tenant=user_id)
                ctx_a = [(_iso_day(e.episode.created_at), e.episode.content) for e in episodes]
                items = (await client.post("/search", json={
                    "query": inst.probe, "user_id": user_id, "top_k": top_k,
                })).json()["data"]
                ctx_b = [(str(it.get("created_at", ""))[:10], it["content"]) for it in items]
                faits = [it["content"] for it in items if it["id"].startswith("fact_")]

        # Contexte O — le plafond : tous les tours utilisateur, dans l'ordre, sans
        # mémoire du tout. Il ne mesure pas Mnemos, il mesure le répondeur. Sans
        # lui, un échec est indécidable — mémoire fautive ou répondeur trop faible ?
        # Avec lui, le bench reste valide même en changeant de modèle de réponse,
        # ce que la VRAM de cette machine impose périodiquement.
        ctx_o = [(_iso_day(m["timestamp"]), m["content"])
                 for m in messages if m["role"] == "user"]

        # Les leurres comptent comme des valeurs à ne pas présenter comme
        # actuelles, au même titre que les valeurs périmées : citer l'employeur
        # du frère en réponse à « où est-ce que je travaille » est une faute.
        interdits = [inst.value_0, inst.value_1, *inst.decoys]
        norm_faits = [normalize_text(f) for f in faits]
        row: dict[str, Any] = {
            "instance": inst.id,
            "attribut": inst.attribute,
            "valeurs": [inst.value_0, inst.value_1, inst.value_2],
            "leurres": inst.decoys,
            "messages": len(messages),
            "facts_inserted": cons["facts_inserted"],
            # Un échec d'extraction change les faits, donc B : il doit se voir.
            "extraction_failures": cons["extraction_failures"],
            # « Les faits ne changent aucune réponse » est la thèse qu'on re-teste :
            # sans ces trois champs, une discordance nulle resterait inexplicable.
            "facts_in_context": faits,
            "active_in_facts": any(mentions_term(f, normalize_text(inst.value_2))
                                   for f in norm_faits),
            "stale_in_facts": any(mentions_term(f, normalize_text(v))
                                  for f in norm_faits
                                  for v in (inst.value_0, inst.value_1)),
            "ranks_A": _ranks(ctx_a, inst),
            "ranks_B": _ranks(ctx_b, inst),
            "ctx_size_A": len(ctx_a),
            "ctx_size_B": len(ctx_b),
        }
        for label, ctx in (("O", ctx_o), ("A", ctx_a), ("B", ctx_b)):
            raw = await llm.generate(
                ANSWER_PROMPT.format(
                    order=ORDRES[label], context=_context(ctx),
                    question=inst.probe, hint=inst.answer_hint,
                ),
                answer_model,
                options={"temperature": 0.0, "num_ctx": 8192, "num_predict": 64},
            )
            answer = _THINK_RE.sub("", raw).strip()
            ok, reason = check_active_fact_answer(answer, inst.value_2, interdits)
            row[f"answer_{label}"] = answer
            row[f"correct_{label}"] = ok
            row[f"reason_{label}"] = reason
        row["embed_backend_effectif"] = settings.EMBED_BACKEND
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _diagnostic(row: dict[str, Any], label: str) -> str:
    """Classe l'échec à partir des rangs — la raison d'être de ce bench."""
    if row[f"correct_{label}"]:
        return "ok"
    # Le contrôle oracle passe avant tout : si le répondeur échoue déjà sur la
    # conversation entière et ordonnée, l'instance ne dit rien de la mémoire.
    if not row["correct_O"]:
        return "hors plafond: le répondeur échoue déjà sur le transcript complet"
    r0, r1, r2 = row[f"ranks_{label}"]
    if r2 is None:
        return "classement: énoncé actif absent du contexte"
    autres = [r for r in (r0, r1) if r is not None]
    if autres and min(autres) < r2:
        return f"récence: un énoncé périmé mieux classé (rang {min(autres)} < {r2})"
    if r2 <= 3:
        return f"réponse: énoncé actif au rang {r2}, réponse fausse quand même"
    return f"classement: énoncé actif enfoui au rang {r2}"


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    distant = args.answer_backend != "ollama"
    # Répondeur distant : seul l'extracteur occupe la VRAM locale.
    modele_local = args.llm_model if distant else args.answer_model
    garde = {modele_local, args.llm_model}
    if args.embed_backend == "ollama":
        garde.add(settings.EMBED_MODEL)
    vram = await preparer_vram(settings, garde, modele_local)
    print(f"VRAM — déchargé : {vram['dechargés'] or 'rien'}"
          f" | libre {vram['libre_mib']} Mio | requis ~{vram['requis_mib']} Mio")
    if vram["étrangers"]:
        print(f"       modèles d'un autre projet, laissés en place : {vram['étrangers']}")
    if not vram["suffisant"]:
        print("\nARRÊT : pas assez de VRAM pour charger le répondeur sans risquer les")
        print("500 d'Ollama qui ont tué deux runs le 29/09. Choisir un --answer-model")
        print("plus petit, ou attendre que les modèles ci-dessus expirent.")
        return {"aborted": "vram", "vram": vram}
    print(f"Backend d'embeddings demandé : {args.embed_backend}\n")

    llm = OllamaClient(settings)
    repondeur: Any = RemoteChat(args.answer_backend) if distant else llm
    # Vol d'essai : un appel par modèle avant toute mesure. Le 29/09, un run a
    # démarré sur un runner Ollama en mauvais état : 500 dès la première
    # saillance, retombée sur les scores par défaut, donc des faits différents
    # et un B incomparable — sans que rien ne l'arrête. Un bench qui mesure un
    # système dégradé est pire qu'un bench qui ne démarre pas.
    essais = [(repondeur, args.answer_model), (llm, args.llm_model)]
    for client, modele in dict.fromkeys(essais):
        try:
            await client.generate("Réponds OK.", modele, options={"num_predict": 4})
        except Exception as exc:  # noqa: BLE001 — tout échec invalide la mesure
            await llm.aclose()
            if distant:
                await repondeur.aclose()
            print(f"\nARRÊT : {modele} ne répond pas ({exc}).")
            print("Redémarrer Ollama ou attendre, puis relancer.")
            return {"aborted": "ollama", "modele": modele, "erreur": str(exc)}
    instances = HARD_UPDATE_INSTANCES[: args.instances]
    records: list[dict[str, Any]] = []
    try:
        for n, inst in enumerate(instances, 1):
            row = await run_instance(
                inst, repondeur, args.answer_model, args.top_k, args.user_id,
                args.llm_model, args.embed_backend,
            )
            records.append(row)
            print(
                f"  [{n}/{len(instances)}] {row['instance']:18s}"
                f" | O={'OK ' if row['correct_O'] else 'NON'}"
                f" | A={'OK ' if row['correct_A'] else 'NON'} rangs={row['ranks_A']}"
                f" | B={'OK ' if row['correct_B'] else 'NON'} rangs={row['ranks_B']}"
                f" | {row['answer_B'][:24]!r}"
            )
    finally:
        await llm.aclose()
        if distant:
            await repondeur.aclose()

    o = sum(r["correct_O"] for r in records)
    a = sum(r["correct_A"] for r in records)
    b = sum(r["correct_B"] for r in records)
    n = len(records)
    p, verdict = mcnemar_test([r["correct_B"] for r in records],
                              [r["correct_A"] for r in records])
    for r in records:
        r["diagnostic_A"] = _diagnostic(r, "A")
        r["diagnostic_B"] = _diagnostic(r, "B")
    causes = {}
    for r in records:
        cle = r["diagnostic_B"].split(":")[0]
        causes[cle] = causes.get(cle, 0) + 1

    result = {
        "config": {"instances": n, "top_k": args.top_k, "user_id": args.user_id,
                   "answer_model": args.answer_model, "answer_backend": args.answer_backend,
                   "extraction_model": args.llm_model,
                   "embed_backend": records[0]["embed_backend_effectif"] if records else None},
        "vram": vram,
        "accuracy": {"O_plafond_repondeur": o / n if n else 0.0,
                     "A_episodique_seul": a / n if n else 0.0,
                     "B_search_aml": b / n if n else 0.0},
        "wilson_A": wilson_score_interval(a, n),
        "wilson_B": wilson_score_interval(b, n),
        "only_A_correct": sum(r["correct_A"] and not r["correct_B"] for r in records),
        "only_B_correct": sum(r["correct_B"] and not r["correct_A"] for r in records),
        "mcnemar_p": p,
        "mcnemar_verdict": verdict,
        "causes_B": causes,
        "facts_inserted_total": sum(r["facts_inserted"] for r in records),
        "extraction_failures_total": sum(r["extraction_failures"] for r in records),
        "instances_avec_fait_actif": sum(r["active_in_facts"] for r in records),
        "instances_avec_fait_perime": sum(r["stale_in_facts"] for r in records),
        "par_attribut": {
            attr: {
                "n": sum(1 for r in records if r["attribut"] == attr),
                "O": sum(r["correct_O"] for r in records if r["attribut"] == attr),
                "A": sum(r["correct_A"] for r in records if r["attribut"] == attr),
                "B": sum(r["correct_B"] for r in records if r["attribut"] == attr),
            }
            for attr in sorted({r["attribut"] for r in records})
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nO — plafond répondeur (transcript complet) : {o}/{n}")
    print(f"A — épisodique seul : {a}/{n}")
    print(f"B — /search AML     : {b}/{n}")
    print(f"discordances : seul A juste {result['only_A_correct']}"
          f" | seul B juste {result['only_B_correct']} | McNemar p={p:.3f}")
    print(f"faits insérés : {result['facts_inserted_total']}"
          f" | fait actif présent dans {result['instances_avec_fait_actif']}/{n}"
          f" | fait périmé présent dans {result['instances_avec_fait_perime']}/{n}")
    print(f"causes d'échec (B) : {causes}")
    if result["extraction_failures_total"]:
        print(f"\nATTENTION : {result['extraction_failures_total']} échec(s) d'extraction —"
              " les faits, donc B, ne sont pas ceux d'un système sain.")
    print("\nLe taux seul ne mesure que la réponse : à top_k=100 tout le contexte")
    print("remonte. Lire les rangs avant de conclure quoi que ce soit — et lire O")
    print("d'abord : un échec que O partage ne dit rien de la mémoire. (O n'est")
    print("pas une borne stricte : un contexte trié peut battre le transcript.)")
    print(f"Rapport : {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 dur : mise à jour de valeur en terrain ambigu")
    parser.add_argument("--instances", type=int, default=len(HARD_UPDATE_INSTANCES))
    parser.add_argument("--top-k", type=int, default=100)
    # PAS un modèle à raisonnement. Le premier passage tournait sur qwen3:4b :
    # `think=False` n'a pas supprimé sa réflexion, elle est sortie dans le corps
    # de la réponse et num_predict l'a coupée avant toute conclusion. Résultat,
    # O=0/13 — le contrôle oracle a invalidé la mesure au lieu de laisser croire
    # que la mémoire échouait. qwen2.5 n'a pas de mode réflexion, et le 3b est
    # déjà résident pour l'extraction : zéro VRAM supplémentaire.
    parser.add_argument("--answer-model", default="qwen2.5:3b")
    parser.add_argument("--answer-backend", choices=("ollama", "nvidia", "mistral"), default="ollama",
                        help="ollama = local ; nvidia / mistral = API distante (données fabriquées)")
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    # La production sert ses embeddings par llama-server depuis le 22/09. Suivre
    # l'environnement en silence, c'est mesurer une pile qu'on n'exploite plus.
    parser.add_argument("--embed-backend", choices=("llamacpp", "ollama"), default="llamacpp")
    parser.add_argument("--user-id", default="update_hard")
    parser.add_argument("--output", type=Path,
                        default=Path("bench/results/gpu/update_hard.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    if sys.platform == "win32":
        # 02/10 : une coupure réseau (WinError 64) a cassé le canal de réveil de la
        # boucle Proactor ; elle ne le réarme pas après une erreur, et le bench est
        # resté figé 30 min sans rien dire. La boucle Selector garde ce canal
        # inscrit en permanence.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
