"""Multi-tenant (Lot 1) — tenant par défaut + sujet canonique par tenant.

Mnemos est passé de single-tenant (une seule mémoire, le sujet `user`) à
multi-tenant : chaque store porte une dimension `tenant` (TEXT, indexée) et
toute requête filtre dessus. L'isolation est stricte — aucun chemin de code
ne lit ou n'écrit hors du tenant courant.

Deux notions distinctes, à ne pas confondre :

- `tenant` : la CLOISON de données. C'est la clé d'isolation présente dans
  chaque table (episodes.tenant, facts.tenant, entities PK composite). Le
  défaut historique est `user` — c'est le tenant de la mémoire personnelle,
  choisi pour que les 47 faits existants (tous `subject='user'`) et les
  clients existants (MCP claude.ai, Claude Code) continuent SANS changement.

- `canonical_subject` : le SUJET des faits produits par l'extracteur pour ce
  tenant (P2). Pour la mémoire personnelle, tenant `user` → sujet `user`
  (l'extracteur parle de « l'utilisateur »). Pour un tenant applicatif comme
  `atelios`, le sujet canonique est `atelios` ; pour un NPC Tomodochi, son
  nom. Le mapping est configurable ci-dessous, fallback = nom du tenant.

Les deux coïncident pour le tenant personnel (`user`/`user`), d'où la
confusion facile — mais ils divergent dès qu'un tenant applicatif entre en
jeu, et le code les traite séparément.
"""

from __future__ import annotations

# Tenant par défaut : la mémoire personnelle historique. Toute écriture/lecture
# sans tenant explicite retombe ici → non-régression totale des clients
# existants (MCP, Claude Code, faits `subject='user'` déjà en base).
DEFAULT_TENANT = "user"

# Mapping explicite tenant → sujet canonique des faits extraits (P2). Un tenant
# absent de cette table utilise son propre nom comme sujet (fallback).
_CANONICAL_SUBJECT: dict[str, str] = {
    "user": "user",  # mémoire personnelle : l'extracteur parle de « user »
}


def canonical_subject(tenant: str) -> str:
    """Sujet canonique des faits extraits pour ce tenant (§P2).

    Fallback = nom du tenant lui-même, de sorte qu'un nouveau tenant
    applicatif produise des faits `subject=<tenant>` sans configuration.
    """
    return _CANONICAL_SUBJECT.get(tenant, tenant)
