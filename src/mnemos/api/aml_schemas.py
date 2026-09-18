"""Schémas Pydantic pour l'adaptateur officiel Agent Memory Leaderboard (AML / Challenge Cycle 2).

Conforme au contrat officiel :
- POST /add (requête et réponse)
- POST /search (requête et réponse)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AMLMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., min_length=1, description="user ou assistant")
    content: str = Field(..., min_length=1, description="Contenu textuel du souvenir")
    timestamp: int | None = Field(default=None, description="Timestamp Unix en millisecondes")


class AMLAddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(..., min_length=1, description="Identifiant logique de l'écriture")
    messages: list[AMLMessage] = Field(
        ..., min_length=1, description="Messages dans l'ordre source"
    )
    user_id: str = Field(..., min_length=1, description="Périmètre d'isolation mémoire")
    session_id: str = Field(..., min_length=1, description="Identifiant de la session source")


class AMLAddResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = Field(default=True, description="True après persistance synchrone")
    request_id: str = Field(..., description="request_id exactement répercuté")
    user_id: str = Field(..., description="user_id exactement répercuté")
    session_id: str = Field(..., description="session_id exactement répercuté")


class AMLSearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="Question originale du benchmark")
    options: list[str] | None = Field(default=None, description="Options de QCM le cas échéant")
    user_id: str = Field(..., min_length=1, description="Périmètre d'isolation mémoire")
    top_k: int = Field(default=100, ge=1, description="Nombre maximum de résultats demandés")


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
