"""Schémas Pydantic pour l'adaptateur officiel Agent Memory Leaderboard (AML / Challenge Cycle 2).

Conforme au contrat officiel :
- POST /add (requête et réponse)
- POST /search (requête et réponse)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AMLMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., min_length=1, description="The message role is user or assistant.")
    content: str | list[dict[str, Any]] = Field(
        ...,
        description=(
            "A string for Textual and Coding, or an ordered ContentPart[] array for Multimodal."
        ),
    )
    timestamp: int | None = Field(
        default=None,
        description=(
            "Sent when the source has a timestamp, in Unix milliseconds. "
            "Chunking does not change its value or message order."
        ),
    )


class AMLAddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifier for this logical write. "
            "Retries keep the same value; echo it in the response."
        ),
    )
    messages: list[AMLMessage] = Field(
        ...,
        min_length=1,
        description="Messages in source order. Store and process them in this order.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Memory isolation scope; later Search requests use the same value.",
    )
    session_id: str = Field(
        ..., min_length=1, description="Identifier for the source conversation or session."
    )


class AMLAddResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        ...,
        description=(
            "Must be true after the messages are durably stored and immediately searchable."
        ),
    )
    request_id: str = Field(..., description="Exact request_id received in the Add request.")
    user_id: str = Field(..., description="Exact user_id received in the Add request.")
    session_id: str = Field(..., description="Exact session_id received in the Add request.")


class AMLSearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str | list[dict[str, Any]] = Field(
        ...,
        description=(
            "The original benchmark question: a string for Textual and Coding, "
            "or ordered ContentPart[] for Multimodal."
        ),
    )
    options: list[str] | None = Field(
        default=None,
        description=(
            "Sent at the top level for multiple-choice questions, including Streaming; "
            "absent for open questions. Never contains the gold answer."
        ),
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="The same memory isolation scope supplied in the corresponding Add request.",
    )
    top_k: int = Field(
        ...,
        ge=1,
        description="Maximum result count; formal external evaluations use 100.",
    )


class AMLMemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Identifiant stable du souvenir")
    content: str = Field(..., min_length=1, description="Texte de la preuve de mémoire")
    score: float | None = Field(default=None, description="Score de pertinence numérique")
    created_at: str | None = Field(default=None, description="Horodatage de persistance ou source")


class AMLSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: list[AMLMemoryItem] = Field(
        default_factory=list, description="Mémoires ordonnées par pertinence"
    )
