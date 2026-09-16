"""Tests unitaires pour le serveur MCP (§MCP).

Vérifie l'importation de mnemos.mcp_server et l'enregistrement
des 5 outils exposés à Claude (memory_write, memory_query,
memory_facts, memory_forget, memory_consolidate).
"""

from __future__ import annotations

from mnemos.mcp_server import mcp


async def test_mcp_server_tools_registered() -> None:
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    expected = {
        "memory_write",
        "memory_query",
        "memory_facts",
        "memory_forget",
        "memory_consolidate",
    }
    assert tool_names == expected


async def test_mcp_server_instructions_present() -> None:
    assert mcp.instructions is not None
    assert "Persistent local memory" in mcp.instructions
