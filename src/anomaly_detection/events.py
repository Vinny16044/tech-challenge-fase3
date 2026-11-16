"""
Montagem de eventos no contrato da Fase 4 (`src.contracts.MonitoringEventV1`).

Usado pelos módulos de anomalias clínicas e de vídeo. Todo evento é validado
pelo modelo Pydantic antes de ser devolvido, então um erro de contrato aparece
no módulo que gerou o achado, e não só na integração.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.contracts import MonitoringEventV1

CONTRACT_VERSION = "1.1"
MODULE_VERSION = "1.0.0"

MAX_FINDING = 500
MAX_EVIDENCE = 20
MAX_DESCRIPTION = 300
MAX_REFERENCE = 40
MAX_METRIC = 60
MAX_UNIT = 20


@dataclass(frozen=True)
class EventContext:
    """
    Como os identificadores e horários da origem viram os do evento.

    `patient_alias` troca o ID de origem por outro (ex.: paciente do MIMIC-IV
    apresentado como `P0001` no teste conjunto). `time_shift` desloca os
    horários: o MIMIC-IV já publica datas deslocadas para o futuro, e o teste
    conjunto precisa que as três modalidades caiam na mesma janela.
    """

    patient_alias: dict[str, str] = field(default_factory=dict)
    time_shift: timedelta = timedelta(0)
    tz: timezone = timezone.utc

    def patient_id(self, source_id: str | int) -> str:
        key = str(source_id)
        if key in self.patient_alias:
            return self.patient_alias[key]
        if key.isdigit():
            return f"P{int(key):04d}"  # o contrato pede ao menos 4 dígitos
        return key if key.startswith("P") else f"P{key}"

    def timestamp(self, moment: datetime) -> datetime:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return (moment + self.time_shift).astimezone(self.tz)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def safe_id(text: str) -> str:
    """Reduz um identificador local aos caracteres aceitos em `event_id`."""
    return re.sub(r"[^A-Za-z0-9._:-]+", "-", str(text)).strip("-")


def measurement(
    *,
    uri: str,
    metric: str,
    value: float,
    unit: str,
    reference_range: str | None = None,
    description: str | None = None,
    confidence: float | None = None,
    kind: str = "clinical_measurement",
) -> dict[str, Any]:
    """Evidência clínica (`clinical_measurement`, `lab_result` ou `prescription`)."""
    item: dict[str, Any] = {
        "type": kind,
        "uri": uri,
        "metric": _clip(metric, MAX_METRIC),
        "value": round(float(value), 4),
        "unit": _clip(unit, MAX_UNIT),
    }
    if reference_range:
        item["reference_range"] = _clip(reference_range, MAX_REFERENCE)
    if description:
        item["description"] = _clip(description, MAX_DESCRIPTION)
    if confidence is not None:
        item["confidence"] = round(min(max(float(confidence), 0.0), 1.0), 3)
    return item


def build_event(
    *,
    module: str,
    patient_id: str,
    local_id: str,
    timestamp: datetime,
    event_type: str,
    finding: str,
    anomaly_detected: bool,
    priority: str,
    evidence: list[dict[str, Any]],
    system: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Monta e valida um evento. O mesmo achado gera sempre o mesmo `event_id`."""
    if timestamp.tzinfo is None:
        raise ValueError("o horário do evento precisa de fuso")
    source: dict[str, Any] = {"system": system, "version": MODULE_VERSION}
    if model:
        source["model"] = _clip(model, 80)
    event = {
        "schema_version": CONTRACT_VERSION,
        "event_id": f"{module}-{patient_id}-{safe_id(local_id)}"[:124],
        "patient_id": patient_id,
        "module": module,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "event_type": event_type,
        "finding": _clip(finding, MAX_FINDING),
        "anomaly_detected": bool(anomaly_detected),
        "priority": priority if anomaly_detected else "low",
        "evidence": evidence[:MAX_EVIDENCE],
        "source": source,
    }
    MonitoringEventV1.model_validate(event)
    return event


def save_json(payload: Any, path: Path) -> Path:
    """Grava JSON em UTF-8, criando a pasta se preciso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
