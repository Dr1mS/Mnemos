"""Résolution du tenant d'un appel MCP (argument `tenant` + TENANT_PREFIX)."""

from __future__ import annotations

import pytest

from mnemos.tenancy import resolve_tenant


def test_sans_tenant_demande_reste_sur_le_tenant_de_l_instance() -> None:
    assert resolve_tenant(None, "user") == "user"
    assert resolve_tenant("", "zygos", "zygos-") == "zygos"
    assert resolve_tenant("zygos", "zygos", "zygos-") == "zygos"


def test_instance_sans_prefixe_refuse_tout_autre_tenant() -> None:
    with pytest.raises(ValueError):
        resolve_tenant("zygos-lucioles", "user")


def test_prefixe_autorise_les_tenants_du_projet() -> None:
    assert resolve_tenant("zygos-lucioles", "zygos", "zygos-") == "zygos-lucioles"


def test_instance_applicative_ne_peut_pas_atteindre_la_memoire_personnelle() -> None:
    with pytest.raises(ValueError):
        resolve_tenant("user", "zygos", "zygos-")
    with pytest.raises(ValueError):
        resolve_tenant("zygos-", "zygos", "zygos-")
    with pytest.raises(ValueError):
        resolve_tenant("autre-lucioles", "zygos", "zygos-")


def test_tenant_trop_long_refuse() -> None:
    with pytest.raises(ValueError):
        resolve_tenant("zygos-" + "x" * 200, "zygos", "zygos-")
