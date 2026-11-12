"""
Decisão de revisão humana sobre um alerta.

Todo alerta nasce em `pending_medical_review`. A decisão é sempre de um
profissional identificado; o sistema apenas registra.

Responsável: Paola
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

ReviewDecisionValue = Literal["confirmed", "dismissed"]

REVIEW_STATUS_PENDING = "pending_medical_review"
REVIEW_STATUS_BY_DECISION: dict[str, str] = {
    "confirmed": "reviewed_confirmed",
    "dismissed": "reviewed_dismissed",
}


class ReviewDecision(BaseModel):
    """Decisão informada por um profissional para um alerta específico."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(
        pattern=r"^ALT-[0-9a-f]{12}$",
        description="ID do alerta revisado (gerado pelo integrador).",
    )
    reviewer_id: str = Field(
        min_length=2,
        max_length=60,
        description="Identificação do revisor (ex.: CRM fictício ou matrícula).",
    )
    reviewer_name: str | None = Field(default=None, max_length=120)
    decision: ReviewDecisionValue = Field(
        description="confirmed = achado procede; dismissed = achado descartado."
    )
    decided_at: AwareDatetime = Field(
        description="Horário da decisão, ISO 8601 com fuso."
    )
    notes: str | None = Field(default=None, max_length=1000)
