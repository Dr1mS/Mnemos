#!/usr/bin/env python3
"""POC bench — teste les modèles candidats sur les charges réelles de Mnemos.

Pas de dépendances : stdlib uniquement (urllib). Ollama doit tourner en local.

Trois charges, calquées sur MNEMOS_SPEC.md :
  1. embedding  — latence bge-m3 (write path synchrone, cible < 500 ms)
  2. salience   — §13.2 : JSON 4 floats, latence + JSON valide + plausibilité
  3. extraction — §15.2 : faits + entités FR/EN, latence + JSON valide
                  + rappel gold / pièges (faits interdits) / silence sur bruit

Corpus v2 : 10 épisodes de base + 12 cas piégeux (négation, hypothétique,
coréférence, tiers, incertitude, sarcasme, temps passé, fait enfoui, mixte).

Usage :
  python3 poc/bench_models.py                # tout
  python3 poc/bench_models.py --task salience
  python3 poc/bench_models.py --task extraction --models qwen3:4b

Suivi en direct : les logs sont horodatés et flushés → `tail -f` sur la
sortie redirigée fonctionne.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request

RAW_DUMP = pathlib.Path(__file__).parent / "raw_extractions.jsonl"

OLLAMA = "http://localhost:11434"
T_START = time.perf_counter()

# ── Matrice de modèles ────────────────────────────────────────────────────────
EMBED_MODELS = ["bge-m3"]
# qwen3:1.7b éliminé au round 1 (plausibilité salience 3/10) — réactivable
# via --models qwen3:1.7b.
SALIENCE_MODELS = ["qwen2.5:3b", "qwen3:4b"]
EXTRACTION_MODELS = ["qwen2.5:7b-instruct-q4_K_M", "qwen3:4b"]

# Les modèles qwen3 ont un mode thinking qui casse le JSON structuré sous
# Ollama (issue #10929) et explose la latence CPU → think=false obligatoire.
def is_thinking_family(model: str) -> bool:
    return model.startswith("qwen3")


# ── Logging temps réel ────────────────────────────────────────────────────────
def log(msg: str) -> None:
    elapsed = time.perf_counter() - T_START
    print(f"[{int(elapsed // 60):02d}:{elapsed % 60:04.1f}] {msg}", flush=True)


# ── Corpus de test (persona "Alice", FR/EN) ──────────────────────────────────
# Champs par épisode :
#   content          : le message
#   gold             : faits attendus [(subject, predicate, object)] — rappel
#   forbidden        : faits PIÈGES à ne PAS extraire — précision
#   expect_silence   : True si l'extraction doit être strictement vide
#   salience_expect  : "high_self_ref" | "low" | "skip" (affiché, non compté)
EPISODES = [
    # ── Base (round 1) ──
    {
        "content": "Salut ! Moi c'est Alice, je bosse chez Datalyse comme data engineer.",
        "gold": [("alice", "works_at", "datalyse"), ("alice", "is_a", "data engineer")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "J'habite à Lyon depuis 3 ans, dans le quartier de la Croix-Rousse.",
        "gold": [("alice", "lives_in", "lyon")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "Je préfère le thé au café, surtout le thé vert japonais.",
        "gold": [("alice", "prefers", "thé")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "Big news! I quit Datalyse last week, I'm joining Nexora as a senior ML engineer.",
        "gold": [("alice", "works_at", "nexora")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "Du coup je déménage à Paris le mois prochain, Nexora est dans le 11e.",
        "gold": [("alice", "lives_in", "paris")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "Je déteste les open spaces, impossible de me concentrer.",
        "gold": [("alice", "dislikes", "open space")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "Tu peux me rappeler quelle heure il est à Tokyo ?",
        "gold": [],
        "expect_silence": True,
        "salience_expect": "low",
    },
    {
        "content": "My cat is named Miso, she's a 2-year-old Siamese.",
        "gold": [("alice", "owns", "miso")],
        "salience_expect": "high_self_ref",
    },
    {
        "content": "ok merci",
        "gold": [],
        "expect_silence": True,
        "salience_expect": "low",
    },
    {
        "content": "J'aimerais apprendre Rust cette année, c'est mon objectif principal.",
        "gold": [("alice", "has_goal", "rust")],
        "salience_expect": "high_self_ref",
    },
    # ── Cas piégeux (round 2) ──
    {
        # Rétractation : le piège est d'extraire la préférence NIÉE.
        "content": "Finalement je ne bois plus de thé, je suis passée au maté.",
        "gold": [("alice", "prefers", "maté")],
        "forbidden": [("alice", "prefers", "thé")],
        "salience_expect": "high_self_ref",
    },
    {
        # Hypothétique : rien à extraire (règle IGNORE hypotheticals).
        "content": "Si je gagnais au loto, j'achèterais une villa à Nice.",
        "gold": [],
        "forbidden": [("alice", "owns", "villa"), ("alice", "lives_in", "nice")],
        "expect_silence": True,
        "salience_expect": "skip",
    },
    {
        # Question avec fait tentant : ne PAS extraire works_at.
        "content": "Tu penses que je devrais accepter l'offre de Quantic Labs ?",
        "gold": [],
        "forbidden": [("alice", "works_at", "quantic")],
        "expect_silence": True,
        "salience_expect": "skip",
    },
    {
        # Tiers explicite : le fait porte sur Tom, PAS sur l'utilisatrice.
        "content": "Mon frère Tom travaille chez Airbus à Toulouse.",
        "gold": [("tom", "works_at", "airbus")],
        "forbidden": [("alice", "works_at", "airbus")],
        "salience_expect": "skip",
    },
    {
        # Coréférence non résolvable (épisode isolé) : "il" = ??? → ne PAS
        # attribuer la moto à l'utilisatrice.
        "content": "Il vient de s'acheter une moto, il en rêvait depuis des années.",
        "gold": [],
        "forbidden": [("alice", "owns", "moto")],
        "salience_expect": "low",
    },
    {
        # Incertitude : règle IGNORE expressions of uncertainty.
        "content": "I'm thinking about maybe learning guitar someday, not sure though.",
        "gold": [],
        "forbidden": [("alice", "has_goal", "guitar")],
        "expect_silence": True,
        "salience_expect": "skip",
    },
    {
        # Sarcasme : ne PAS extraire prefers(lundi).
        "content": "Super, encore un lundi... j'adore les lundis 🙄",
        "gold": [],
        "forbidden": [("alice", "prefers", "lundi")],
        "expect_silence": True,
        "salience_expect": "skip",
    },
    {
        # Émotionnel mais pas auto-révélateur : arousal haut, self_ref bas.
        "content": "Incroyable, l'équipe de France a gagné 5-0 hier soir !",
        "gold": [],
        "expect_silence": True,
        "salience_expect": "low",
    },
    {
        # Temps passé : works_at(TechCorp) n'est PLUS vrai → l'extraire comme
        # fait courant serait un bug de versioning pour Mnemos.
        "content": "Avant je bossais chez TechCorp, c'était l'enfer.",
        "gold": [],
        "forbidden": [("alice", "works_at", "techcorp")],
        "salience_expect": "high_self_ref",
    },
    {
        # Fait enfoui dans du bavardage.
        "content": ("Bref, journée interminable, trois réunions d'affilée et le "
                     "métro en panne au retour... Ah et au fait, je me suis "
                     "inscrite à un cours de poterie le jeudi soir !"),
        "gold": [("alice", "has_attribute", "poterie")],
        "salience_expect": "high_self_ref",
    },
    {
        # Mixte FR/EN dans la même phrase.
        "content": "BTW j'ai commencé à apprendre le japonais, my sensei says I'm making progress!",
        "gold": [("alice", "knows_about", "japonais")],
        "salience_expect": "high_self_ref",
    },
    {
        # Deux faits dans un seul message.
        "content": ("Petit update : j'ai adopté un deuxième chat, Yuzu, et j'ai "
                     "enfin décroché ma certification AWS !"),
        "gold": [("alice", "owns", "yuzu"), ("alice", "has_skill", "aws")],
        "salience_expect": "high_self_ref",
    },
]

# ── Prompts (copiés du spec, §13.2 et §15.2) ─────────────────────────────────
SALIENCE_PROMPT = """You score the salience of a single message for memory consolidation.
Return JSON with four floats in [0,1]:

- surprise: how unexpected/novel is this content vs typical conversation
- arousal: emotional intensity (positive or negative, both score high)
- self_ref: how much the user reveals about themselves (preferences, identity, life facts)
- recurrence: 0 if this topic is new in the recent history, higher if it repeats

Recent history (last 5 turns):
{recent_history}

Current message:
{content}

Output ONLY JSON: {{"surprise": 0.X, "arousal": 0.X, "self_ref": 0.X, "recurrence": 0.X}}"""

EXTRACTION_PROMPT = """Extract structured facts and named entities from this conversation episode.
Output JSON object: {{"facts": [...], "entities": [...]}}.
Each fact: {{subject, predicate, object, confidence}}.
Each entity: {{name, entity_type, aliases}} where entity_type is one of
             person|org|place|concept|product, and aliases lists other
             surface forms used in the episode (may be empty).

Allowed predicates: works_at, lives_in, prefers, dislikes, owns,
                    is_a, has_attribute, knows_about, has_goal, has_skill

Rules:
- subject is the user unless explicitly about another entity
- only extract facts that are statements of preference, identity, or factual claims
- IGNORE questions, hypotheticals, jokes, expressions of uncertainty
- entities: only entities actually mentioned; use the most complete surface
  form as name
- if nothing extractable, return {{"facts": [], "entities": []}}
- use canonical English for predicates even if input is French

Episode (role=user, timestamp=2026-07-02T10:00:00Z):
{content}

Output ONLY JSON."""

# Prompt v2 — fixes issus du bench round 2 :
#   1. temps passé → ne pas extraire comme fait courant (piège TechCorp)
#   2. pronom sans référent nommé → ne rien extraire (piège moto)
#   3. objects dans la langue d'origine (artefact de traduction poterie/japonais)
#   + renforcement hypothétiques/conditionnels/sarcasme (piège loto)
EXTRACTION_PROMPT_V2 = """Extract structured facts and named entities from this conversation episode.
Output JSON object: {{"facts": [...], "entities": [...]}}.
Each fact: {{subject, predicate, object, confidence}}.
Each entity: {{name, entity_type, aliases}} where entity_type is one of
             person|org|place|concept|product, and aliases lists other
             surface forms used in the episode (may be empty).

Allowed predicates: works_at, lives_in, prefers, dislikes, owns,
                    is_a, has_attribute, knows_about, has_goal, has_skill

Rules:
- subject is the user unless explicitly about another entity
- if a statement is about a pronoun ("il", "elle", "he", "she", "they")
  whose referent is NOT named in this episode, extract NOTHING about it
- only extract facts that are statements of preference, identity, or factual claims
- only extract facts that are CURRENTLY true: IGNORE statements about the
  past that are no longer true ("I used to work at X", "avant je bossais
  chez X", "j'habitais à Y")
- IGNORE questions, hypotheticals, conditionals ("si je gagnais...",
  "if I won..."), wishes, jokes, sarcasm, and expressions of uncertainty
  ("maybe", "peut-être", "not sure")
- use canonical English for predicates, but KEEP subject and object in the
  original language of the episode (do not translate them)
- entities: only entities actually mentioned; use the most complete surface
  form as name
- if nothing extractable, return {{"facts": [], "entities": []}}

Episode (role=user, timestamp=2026-07-02T10:00:00Z):
{content}

Output ONLY JSON."""

# Prompt v3 — corrige la régression v2 : la règle de langue d'origine avait
# débordé sur le subject (le modèle sortait "Je" / "J'habite à Lyon" comme
# sujet → inutilisable pour la résolution d'alias). Le subject redevient
# canonique ("user" ou nom d'entité), seul l'object garde sa langue.
# + confidence explicitement float (v2 sortait "high").
EXTRACTION_PROMPT_V3 = """Extract structured facts and named entities from this conversation episode.
Output JSON object: {{"facts": [...], "entities": [...]}}.
Each fact: {{subject, predicate, object, confidence}}.
Each entity: {{name, entity_type, aliases}} where entity_type is one of
             person|org|place|concept|product, and aliases lists other
             surface forms used in the episode (may be empty).

Allowed predicates: works_at, lives_in, prefers, dislikes, owns,
                    is_a, has_attribute, knows_about, has_goal, has_skill

Rules:
- subject must be EXACTLY "user" when the fact is about the user speaking
  (statements with "je", "I", "my", "mon"); when the fact is explicitly
  about another named entity, use that entity's name as subject
- if a statement is about a pronoun ("il", "elle", "he", "she", "they")
  whose referent is NOT named in this episode, extract NOTHING about it
- only extract facts that are statements of preference, identity, or factual claims
- only extract facts that are CURRENTLY true: IGNORE statements about the
  past that are no longer true ("I used to work at X", "avant je bossais
  chez X", "j'habitais à Y")
- IGNORE questions, hypotheticals, conditionals ("si je gagnais...",
  "if I won..."), wishes, jokes, sarcasm, and expressions of uncertainty
  ("maybe", "peut-être", "not sure")
- use canonical English for predicates, but keep the object in the original
  language of the episode (do not translate it)
- confidence is a number between 0.0 and 1.0
- entities: only entities actually mentioned; use the most complete surface
  form as name
- if nothing extractable, return {{"facts": [], "entities": []}}

Episode (role=user, timestamp=2026-07-02T10:00:00Z):
{content}

Output ONLY JSON."""

# Prompt v4 — v3 sur-supprimait : la règle "passé" tuait le passé composé
# établissant un état courant ("j'ai adopté un chat"), la règle "wishes"
# tuait has_goal ("j'aimerais apprendre Rust"). Un 4B suit mal 10 règles
# abstraites → on remplace les règles ambiguës par des exemples de bascule.
EXTRACTION_PROMPT_V4 = """Extract structured facts and named entities from this conversation episode.
Output JSON object: {{"facts": [...], "entities": [...]}}.
Each fact: {{subject, predicate, object, confidence}}.
Each entity: {{name, entity_type, aliases}} where entity_type is one of
             person|org|place|concept|product, and aliases lists other
             surface forms used in the episode (may be empty).

Allowed predicates: works_at, lives_in, prefers, dislikes, owns,
                    is_a, has_attribute, knows_about, has_goal, has_skill

Rules:
- subject is EXACTLY "user" when the fact is about the user speaking; when
  the sentence explicitly names another person/entity as the actor, that
  entity is the subject
- if the actor is a pronoun whose referent is NOT named in this episode,
  extract NOTHING about it
- extract facts that are CURRENTLY true. A past event that established a
  current state IS a current fact. A state explicitly ended is NOT.
- personal goals and desires to learn/do something ARE facts (has_goal)
- IGNORE questions, unrealistic hypotheticals/conditionals, jokes, sarcasm,
  and statements the speaker is unsure about
- use canonical English for predicates, but keep the object in the original
  language of the episode (do not translate it)
- confidence is a number between 0.0 and 1.0
- entities: only entities actually mentioned; use the most complete surface
  form as name
- if nothing extractable, return {{"facts": [], "entities": []}}

Examples:
- "Avant je bossais chez TechCorp." → facts: []  (state ended, no longer true)
- "J'ai adopté un chat, Yuzu." → {{"subject": "user", "predicate": "owns", "object": "Yuzu", "confidence": 0.9}}  (past event, current state)
- "J'aimerais apprendre Rust." → {{"subject": "user", "predicate": "has_goal", "object": "Rust", "confidence": 0.9}}
- "Mon frère Tom travaille chez Airbus." → {{"subject": "Tom", "predicate": "works_at", "object": "Airbus", "confidence": 0.9}}
- "Je ne bois plus de thé, je suis passée au maté." → {{"subject": "user", "predicate": "prefers", "object": "maté", "confidence": 0.9}}  (only the NEW preference)
- "Si je gagnais au loto, j'achèterais une villa." → facts: []  (hypothetical)

Episode (role=user, timestamp=2026-07-02T10:00:00Z):
{content}

Output ONLY JSON."""

EXTRACTION_PROMPTS = {
    "spec": EXTRACTION_PROMPT,
    "v2": EXTRACTION_PROMPT_V2,
    "v3": EXTRACTION_PROMPT_V3,
    "v4": EXTRACTION_PROMPT_V4,
}

PREDICATES = {
    "works_at", "lives_in", "prefers", "dislikes", "owns",
    "is_a", "has_attribute", "knows_about", "has_goal", "has_skill",
}

# Prédicats interchangeables pour le matching gold — l'extracteur peut
# légitimement hésiter entre catégories voisines.
PREDICATE_EQUIV = {
    "owns": {"owns", "has_attribute"},
    "is_a": {"is_a", "has_attribute", "has_skill"},
    "has_goal": {"has_goal", "knows_about"},
    "has_attribute": {"has_attribute", "prefers", "has_goal", "knows_about"},
    "knows_about": {"knows_about", "has_skill", "has_goal"},
    "has_skill": {"has_skill", "has_attribute", "knows_about"},
}


# ── Client Ollama minimal ─────────────────────────────────────────────────────
def ollama_post(path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def embed(model: str, text: str) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    out = ollama_post("/api/embed", {"model": model, "input": text})
    dt = time.perf_counter() - t0
    return out["embeddings"][0], dt


def generate_json(model: str, prompt: str) -> tuple[str, float, dict]:
    """Retourne (texte brut, latence s, métriques ollama)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    if is_thinking_family(model):
        payload["think"] = False
    t0 = time.perf_counter()
    out = ollama_post("/api/generate", payload)
    dt = time.perf_counter() - t0
    metrics = {
        "eval_count": out.get("eval_count", 0),
        "eval_duration_s": out.get("eval_duration", 0) / 1e9,
        "load_duration_s": out.get("load_duration", 0) / 1e9,
    }
    return out.get("response", ""), dt, metrics


def unload_all() -> None:
    """Décharge les modèles résidents pour isoler chaque mesure (RAM 16 GB)."""
    try:
        req = urllib.request.Request(f"{OLLAMA}/api/ps")
        with urllib.request.urlopen(req, timeout=10) as resp:
            running = json.loads(resp.read()).get("models", [])
        for m in running:
            ollama_post("/api/generate", {"model": m["name"], "keep_alive": 0})
    except Exception:
        pass


# ── Charge 1 : embedding ─────────────────────────────────────────────────────
def bench_embedding(models: list[str]) -> list[dict]:
    results = []
    texts = [e["content"] for e in EPISODES]
    for mi, model in enumerate(models, 1):
        log(f"embedding [{mi}/{len(models)}] {model} — chargement…")
        unload_all()
        embed(model, "warmup")  # charge le modèle, non compté
        lats, dim = [], 0
        for i, t in enumerate(texts, 1):
            vec, dt = embed(model, t)
            dim = len(vec)
            lats.append(dt)
            log(f"  ep {i:2d}/{len(texts)}  {dt*1000:6.0f} ms  {t[:48]!r}")
        row = {
            "model": model,
            "dim": dim,
            "p50_ms": statistics.median(lats) * 1000,
            "max_ms": max(lats) * 1000,
            "mean_ms": statistics.mean(lats) * 1000,
        }
        log(f"  → dim={dim}  p50={row['p50_ms']:.0f}ms  max={row['max_ms']:.0f}ms")
        results.append(row)
    return results


# ── Charge 2 : salience ──────────────────────────────────────────────────────
def parse_salience(raw: str) -> dict | None:
    try:
        d = json.loads(raw)
        keys = {"surprise", "arousal", "self_ref", "recurrence"}
        if not keys <= set(d):
            return None
        for k in keys:
            v = float(d[k])
            if not 0.0 <= v <= 1.0:
                return None
            d[k] = v
        return d
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def bench_salience(models: list[str]) -> list[dict]:
    results = []
    for mi, model in enumerate(models, 1):
        log(f"salience [{mi}/{len(models)}] {model} — chargement…")
        unload_all()
        generate_json(model, "Return JSON: {\"ok\": true}")  # warm-up
        lats, valid = [], 0
        total_high, n_high, total_low, n_low = 0, 0, 0, 0
        history: list[str] = []
        for i, ep in enumerate(EPISODES, 1):
            prompt = SALIENCE_PROMPT.format(
                recent_history="\n".join(history[-5:]) or "(empty)",
                content=ep["content"],
            )
            raw, dt, _ = generate_json(model, prompt)
            lats.append(dt)
            scores = parse_salience(raw)
            valid += scores is not None
            verdict = "✗ JSON invalide"
            if scores is not None:
                combined = max(
                    0.4 * scores["surprise"] + 0.3 * scores["self_ref"]
                    + 0.2 * scores["arousal"] + 0.1 * scores["recurrence"],
                    scores["self_ref"],
                )
                expect = ep["salience_expect"]
                if expect == "high_self_ref":
                    n_high += 1
                    hit = scores["self_ref"] >= 0.5
                    total_high += hit
                    verdict = f"self_ref={scores['self_ref']:.2f} {'✓' if hit else '✗ attendu ≥0.5'}"
                elif expect == "low":
                    n_low += 1
                    hit = combined < 0.6  # sous le seuil de consolidation
                    total_low += hit
                    verdict = f"combined={combined:.2f} {'✓' if hit else '✗ attendu <0.6'}"
                else:  # skip : affiché, non compté
                    verdict = f"combined={combined:.2f} self_ref={scores['self_ref']:.2f} (skip)"
            log(f"  ep {i:2d}/{len(EPISODES)}  {dt:5.1f}s  {verdict:38s}  {ep['content'][:38]!r}")
            history.append(ep["content"])
        row = {
            "model": model,
            "json_valid": f"{valid}/{len(EPISODES)}",
            "high_ok": f"{total_high}/{n_high}",
            "low_ok": f"{total_low}/{n_low}",
            "score": f"{total_high + total_low}/{n_high + n_low}",
            "p50_s": statistics.median(lats),
            "max_s": max(lats),
        }
        log(f"  → JSON {row['json_valid']}  high {row['high_ok']}  low {row['low_ok']}  "
            f"p50={row['p50_s']:.1f}s  max={row['max_s']:.1f}s")
        results.append(row)
    return results


# ── Charge 3 : extraction ────────────────────────────────────────────────────
def _subject_matches(gold_subject: str, extracted_subject: str) -> bool:
    s = extracted_subject.lower()
    if gold_subject in s:
        return True
    # "user"/"utilisateur" n'est un alias valide que pour Alice elle-même.
    return gold_subject == "alice" and ("user" in s or "utilisat" in s)


def match_fact(gold: tuple[str, str, str], extracted: list[dict]) -> bool:
    gs, gp, go = gold
    preds_ok = PREDICATE_EQUIV.get(gp, {gp})
    for f in extracted:
        try:
            s = str(f["subject"]).lower()
            p = str(f["predicate"]).lower()
            o = str(f["object"]).lower()
        except (KeyError, TypeError):
            continue
        if not _subject_matches(gs, s):
            continue
        if p not in preds_ok:
            continue
        if go in o or o in go:
            return True
    return False


def bench_extraction(models: list[str], prompt_version: str = "spec") -> list[dict]:
    results = []
    extraction_prompt = EXTRACTION_PROMPTS[prompt_version]
    n_gold = sum(len(e["gold"]) for e in EPISODES)
    n_traps = sum(len(e.get("forbidden", [])) for e in EPISODES)
    n_silence = sum(1 for e in EPISODES if e.get("expect_silence"))
    for mi, model in enumerate(models, 1):
        log(f"extraction [{mi}/{len(models)}] {model} (prompt {prompt_version}) — chargement…")
        unload_all()
        generate_json(model, "Return JSON: {\"ok\": true}")  # warm-up
        lats, valid = [], 0
        recall_hits, trap_hits, silence_ok = 0, 0, 0
        false_facts, bad_predicates = 0, 0
        tok_speeds = []
        for i, ep in enumerate(EPISODES, 1):
            raw, dt, m = generate_json(model, extraction_prompt.format(content=ep["content"]))
            lats.append(dt)
            with RAW_DUMP.open("a") as fh:  # audit : sorties brutes rejouables
                fh.write(json.dumps({
                    "model": model, "prompt": prompt_version,
                    "episode": ep["content"], "raw": raw,
                }, ensure_ascii=False) + "\n")
            if m["eval_duration_s"] > 0:
                tok_speeds.append(m["eval_count"] / m["eval_duration_s"])
            try:
                d = json.loads(raw)
                facts = d.get("facts", [])
                assert isinstance(facts, list)
                ok = True
            except (json.JSONDecodeError, AssertionError, AttributeError):
                facts, ok = [], False
            valid += ok
            hits = sum(match_fact(g, facts) for g in ep["gold"])
            recall_hits += hits
            traps = sum(match_fact(t, facts) for t in ep.get("forbidden", []))
            trap_hits += traps
            bad_predicates += sum(
                1 for f in facts
                if isinstance(f, dict) and str(f.get("predicate", "")).lower() not in PREDICATES
            )
            parts = []
            if ep["gold"]:
                parts.append(f"{hits}/{len(ep['gold'])} gold")
                extra = max(0, len(facts) - len(ep["gold"]) - 1)  # tolérance +1
                false_facts += extra
                if extra:
                    parts.append(f"+{extra} extra")
            if ep.get("expect_silence"):
                clean = ok and len(facts) == 0
                silence_ok += clean
                parts.append("silence ✓" if clean else f"✗ {len(facts)} fait(s) au lieu de 0")
            if traps:
                parts.append(f"⚠ {traps} PIÈGE(S)")
            if not ok:
                parts = ["✗ JSON invalide"]
            verdict = "  ".join(parts) or "rien attendu, rien noté"
            log(f"  ep {i:2d}/{len(EPISODES)}  {dt:5.1f}s  {verdict:34s}  {ep['content'][:38]!r}")
        row = {
            "model": model,
            "json_valid": f"{valid}/{len(EPISODES)}",
            "recall": f"{recall_hits}/{n_gold}",
            "traps": f"{trap_hits}/{n_traps}",
            "silence": f"{silence_ok}/{n_silence}",
            "extra": false_facts,
            "bad_pred": bad_predicates,
            "p50_s": statistics.median(lats),
            "max_s": max(lats),
            "tok_s": statistics.mean(tok_speeds) if tok_speeds else 0,
        }
        log(f"  → JSON {row['json_valid']}  rappel {row['recall']}  "
            f"pièges {row['traps']} (0 = parfait)  silence {row['silence']}  "
            f"extra={false_facts}  pred-hors-vocab={bad_predicates}  "
            f"p50={row['p50_s']:.1f}s  {row['tok_s']:.1f} tok/s")
        results.append(row)
    return results


# ── Rapport final ─────────────────────────────────────────────────────────────
def print_table(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}", flush=True)
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols), flush=True)
    for r in rows:
        print("  ".join(_fmt(r[c]).ljust(widths[c]) for c in cols), flush=True)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["embedding", "salience", "extraction", "all"],
                    default="all")
    ap.add_argument("--models", nargs="*", help="restreint aux modèles listés")
    ap.add_argument("--extraction-prompt", choices=list(EXTRACTION_PROMPTS),
                    default="spec", help="version du prompt d'extraction")
    args = ap.parse_args()

    def keep(models: list[str]) -> list[str]:
        return [m for m in models if not args.models or m in args.models]

    try:
        urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=5)
    except urllib.error.URLError:
        print("Ollama ne répond pas sur localhost:11434", file=sys.stderr)
        return 1

    n_gold = sum(len(e["gold"]) for e in EPISODES)
    n_traps = sum(len(e.get("forbidden", [])) for e in EPISODES)
    log(f"corpus : {len(EPISODES)} épisodes, {n_gold} gold facts, {n_traps} pièges")

    report = {}
    if args.task in ("embedding", "all"):
        report["EMBEDDING (write path synchrone — cible < 500 ms)"] = \
            bench_embedding(keep(EMBED_MODELS))
    if args.task in ("salience", "all"):
        report["SALIENCE (async, JSON 4 floats)"] = bench_salience(keep(SALIENCE_MODELS))
    if args.task in ("extraction", "all"):
        report[f"EXTRACTION (worker consolidation, prompt {args.extraction_prompt})"] = \
            bench_extraction(keep(EXTRACTION_MODELS), args.extraction_prompt)

    for title, rows in report.items():
        print_table(title, rows)
    log(f"terminé — durée totale {(time.perf_counter() - T_START) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
