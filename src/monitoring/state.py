from __future__ import annotations

from typing import Any, TypedDict


class RawInput(TypedDict):
    """Uma entrada bruta: o texto/objeto JSON recebido e de onde veio."""

    origin: str  # ex.: data/examples/P0001_audio.json
    content: Any  # str (JSON bruto), dict (um evento) ou list (lote)


class MonitoringState(TypedDict, total=False):
    """
    Estado trafegado entre os nós do LangGraph da Fase 4.
    Responsável: Paola
    """

    run_id: str
    generated_at: str
    window_minutes: int

    raw_inputs: list[RawInput]
    review_decisions: list[Any]

    accepted_events: list[dict[str, Any]]
    rejected_events: list[dict[str, Any]]
    duplicate_events: list[dict[str, Any]]
    events_received: int

    groups: list[dict[str, Any]]
    review_results: list[dict[str, Any]]

    report: dict[str, Any]
    status: str
