"""Registre des backends de mémoire pour le banc d'essai comparatif neutre."""

from __future__ import annotations

from bench.backends.base import BaseMemoryBackend, RecallResult
from bench.backends.control import EmptyControlMemory, OracleControlMemory
from bench.backends.generative_agents import GenerativeAgentsLikeMemory
from bench.backends.graphiti_temporal import GraphitiLikeMemory
from bench.backends.mem0_dynamic import Mem0LikeMemory
from bench.backends.mnemos_backend import MnemosBackend
from bench.backends.naive_vector import NaiveVectorMemory
from bench.backends.obsidian import ObsidianMarkdownMemory
from bench.backends.sliding_window import SlidingWindowMemory
from bench.backends.summary_buffer import SummaryBufferMemory

# Alias de transition
Mem0DynamicMemory = Mem0LikeMemory
GenerativeAgentsMemory = GenerativeAgentsLikeMemory
GraphitiTemporalMemory = GraphitiLikeMemory

COMPARATIVE_BACKENDS = [
    SlidingWindowMemory,
    ObsidianMarkdownMemory,
    SummaryBufferMemory,
    NaiveVectorMemory,
    Mem0LikeMemory,
    GenerativeAgentsLikeMemory,
    GraphitiLikeMemory,
    MnemosBackend,
]

CONTROL_BACKENDS = [
    EmptyControlMemory,
    OracleControlMemory,
]

ALL_BACKENDS = COMPARATIVE_BACKENDS

__all__ = [
    "ALL_BACKENDS",
    "BaseMemoryBackend",
    "COMPARATIVE_BACKENDS",
    "CONTROL_BACKENDS",
    "EmptyControlMemory",
    "GenerativeAgentsLikeMemory",
    "GenerativeAgentsMemory",
    "GraphitiLikeMemory",
    "GraphitiTemporalMemory",
    "Mem0DynamicMemory",
    "Mem0LikeMemory",
    "MnemosBackend",
    "NaiveVectorMemory",
    "ObsidianMarkdownMemory",
    "OracleControlMemory",
    "RecallResult",
    "SlidingWindowMemory",
    "SummaryBufferMemory",
]
