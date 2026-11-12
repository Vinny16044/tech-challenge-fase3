"""
Regras determinísticas de alerta — REGRAS DO PROTÓTIPO (versão prototipo-1).

Estas regras servem para demonstrar a integração. Elas NÃO foram validadas
clinicamente e não substituem protocolos institucionais. O resultado é um
achado para avaliação profissional; todo alerta nasce pendente de revisão.

Ordem de aplicação (só eventos com anomaly_detected = true contam):

  R0 SEM_ANOMALIA          nenhum evento anômalo na janela -> sem alerta.
  R1 PRIORIDADE_MODULO     prioridade inicial = maior prioridade sugerida
                           pelos módulos entre os eventos anômalos.
  R2 EVENTO_SENSIVEL       fall_detected ou call_for_help anômalo -> no
                           mínimo "high".
  R3 CORROBORACAO_2        anomalias em 2+ modalidades diferentes -> sobe
                           um nível (limite "critical").
  R4 CORROBORACAO_3        anomalias nas 3 modalidades -> "critical".
  R5 AVALIACAO_PARCIAL     informativa: lista modalidades ausentes na
                           janela; não altera a prioridade.

Responsável: Paola
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.contracts import MODULES, PRIORITY_ORDER, REVIEW_STATUS_PENDING

RULES_VERSION = "prototipo-1"

SENSITIVE_EVENT_TYPES: frozenset[str] = frozenset({"fall_detected", "call_for_help"})

DISCLAIMER = (
    "Achados automatizados para avaliação profissional. Não constituem "
    "diagnóstico nem indicação de conduta; a decisão é do profissional de saúde."
)


def _rank(priority: str) -> int:
    return PRIORITY_ORDER.index(priority)


def _max_priority(priorities: list[str]) -> str:
    return max(priorities, key=_rank)


def build_alert_id(patient_id: str, window_start: str, event_ids: list[str]) -> str:
    """ID determinístico: mesmo paciente, janela e eventos => mesmo alerta."""
    key = "|".join([patient_id, window_start, *sorted(event_ids)])
    return "ALT-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def evaluate_group(group: dict[str, Any]) -> dict[str, Any] | None:
    """
    Aplica as regras a um grupo (paciente + janela).

    `group["events"]` é a lista de eventos consolidados (dicts do relatório).
    Retorna o alerta (dict) ou None quando R0 se aplica.
    """
    anomalous = [event for event in group["events"] if event["anomaly_detected"]]
    if not anomalous:
        return None

    reasons: list[dict[str, Any]] = []
    anomalous_ids = [event["event_id"] for event in anomalous]

    # R1 — prioridade sugerida pelos módulos.
    priority = _max_priority([event["priority"] for event in anomalous])
    top_ids = [event["event_id"] for event in anomalous if event["priority"] == priority]
    reasons.append(
        {
            "rule_id": "R1",
            "rule": "PRIORIDADE_MODULO",
            "description": (
                f"Maior prioridade sugerida pelos módulos entre os eventos "
                f"anômalos: {priority}."
            ),
            "priority_before": None,
            "priority_after": priority,
            "event_ids": top_ids,
        }
    )

    # R2 — tipos de evento sensíveis exigem no mínimo "high".
    sensitive = [
        event for event in anomalous if event["event_type"] in SENSITIVE_EVENT_TYPES
    ]
    if sensitive:
        before = priority
        priority = _max_priority([priority, "high"])
        reasons.append(
            {
                "rule_id": "R2",
                "rule": "EVENTO_SENSIVEL",
                "description": (
                    "Evento sensível detectado ("
                    + ", ".join(sorted({event["event_type"] for event in sensitive}))
                    + "): prioridade mínima high."
                ),
                "priority_before": before,
                "priority_after": priority,
                "event_ids": [event["event_id"] for event in sensitive],
            }
        )

    # R3/R4 — corroboração entre modalidades.
    anomalous_modules = sorted(
        {event["module"] for event in anomalous}, key=MODULES.index
    )
    if len(anomalous_modules) >= 2:
        before = priority
        priority = PRIORITY_ORDER[min(_rank(priority) + 1, len(PRIORITY_ORDER) - 1)]
        reasons.append(
            {
                "rule_id": "R3",
                "rule": "CORROBORACAO_2",
                "description": (
                    "Anomalias em 2 ou mais modalidades na mesma janela ("
                    + ", ".join(anomalous_modules)
                    + "): prioridade sobe um nível."
                ),
                "priority_before": before,
                "priority_after": priority,
                "event_ids": anomalous_ids,
            }
        )
    if len(anomalous_modules) == len(MODULES):
        before = priority
        priority = "critical"
        reasons.append(
            {
                "rule_id": "R4",
                "rule": "CORROBORACAO_3",
                "description": (
                    "Anomalias nas 3 modalidades (áudio, vídeo e dados clínicos) "
                    "na mesma janela: prioridade critical."
                ),
                "priority_before": before,
                "priority_after": priority,
                "event_ids": anomalous_ids,
            }
        )

    # R5 — informativa: avaliação parcial.
    missing = group.get("sources_missing", [])
    if missing:
        reasons.append(
            {
                "rule_id": "R5",
                "rule": "AVALIACAO_PARCIAL",
                "description": (
                    "Avaliação parcial: sem eventos de "
                    + ", ".join(missing)
                    + " nesta janela. Prioridade não alterada."
                ),
                "priority_before": priority,
                "priority_after": priority,
                "event_ids": [],
            }
        )

    evidence = [
        {
            "event_id": event["event_id"],
            "module": event["module"],
            "event_type": event["event_type"],
            **item,
        }
        for event in anomalous
        for item in event["evidence"]
    ]

    return {
        "alert_id": build_alert_id(group["patient_id"], group["window_start"], anomalous_ids),
        "patient_id": group["patient_id"],
        "window_start": group["window_start"],
        "window_end": group["window_end"],
        "priority": priority,
        "rules_version": RULES_VERSION,
        "priority_reasons": reasons,
        "anomalous_modalities": anomalous_modules,
        "event_ids": anomalous_ids,
        "evidence": evidence,
        "review": {
            "status": REVIEW_STATUS_PENDING,
            "reviewer_id": None,
            "reviewer_name": None,
            "decision": None,
            "decided_at": None,
            "notes": None,
        },
        "disclaimer": DISCLAIMER,
    }
