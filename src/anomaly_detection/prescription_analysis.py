"""
Detecção de alterações inesperadas na evolução das prescrições.

Quatro detectores sobre a tabela de prescrições (hosp/prescriptions do
MIMIC-IV), todos causais: cada pedido é comparado só com o que veio antes.

- `dose_change`      mudança abrupta de dose do mesmo medicamento, via e
                     unidade na mesma internação (razão >= 3x ou <= 1/3).
- `atypical_dose`    dose muito acima do padrão da população para o mesmo
                     medicamento, via e unidade (z robusto em escala log).
- `vasoactive_start` início de droga vasoativa (ou de um segundo agente):
                     escalada de suporte que indica mudança na evolução.
- `order_burst`      volume de novos pedidos em 6 h muito acima do padrão da
                     própria internação (revisão ampla do tratamento).

São sinais para revisão: uma mudança de dose pode ser titulação planejada. O
módulo não avalia indicação, interação nem dose por peso.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection.events import EventContext, build_event, measurement

SYSTEM = "fase4-prescription-anomaly"
MODEL_NAME = "regras de razão de dose + z robusto (mediana/MAD)"

# Diluentes, soluções de manutenção e itens de preparo: não entram na análise.
_BASE_PATTERN = re.compile(
    r"sodium chloride|dextrose|sterile water|lactated ringer|\bbag\b|flush|syringe|\bvial\b|"
    r"iso-osmotic|soln|d5w|\bns\b|heparin flush|potassium chl.*replacement",
    re.IGNORECASE,
)
_VASOACTIVE_PATTERN = re.compile(
    r"\b(norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine|dobutamine|milrinone)\b",
    re.IGNORECASE,
)
# Medicamentos de alta vigilância (lista resumida, inspirada no ISMP).
_HIGH_ALERT_PATTERN = re.compile(
    r"\b(heparin|warfarin|enoxaparin|insulin|morphine|hydromorphone|fentanyl|oxycodone|methadone|"
    r"potassium chloride|digoxin|amiodarone|propofol|midazolam|norepinephrine|epinephrine|"
    r"vasopressin|phenylephrine|dopamine|dobutamine|milrinone)\b",
    re.IGNORECASE,
)
# Titulados por protocolo (escala móvel, ajuste por exame): variação é rotina.
_TITRATED_PATTERN = re.compile(r"\b(insulin|warfarin|vancomycin)\b", re.IGNORECASE)

PRIORITIES = ("low", "medium", "high", "critical")


@dataclass(frozen=True)
class PrescriptionConfig:
    change_ratio: float = 3.0  # mudança de dose que conta como abrupta
    strong_ratio: float = 5.0  # mudança que sobe a prioridade
    titrated_ratio: float = 5.0  # limiar para medicamentos titulados por protocolo
    change_max_gap_days: float = 7.0
    population_min: int = 20  # pedidos mínimos para ter um padrão da população
    population_z: float = 3.5
    population_min_ratio: float = 5.0  # e pelo menos 5x a mediana
    burst_hours: int = 6
    burst_skip_hours: int = 24  # a admissão concentra pedidos: não conta
    burst_min_history: int = 4
    burst_min_orders: int = 10
    burst_z: float = 4.0


@dataclass
class PrescriptionFinding:
    kind: str
    subject_id: int
    hadm_id: int
    time: datetime
    drug: str
    route: str
    unit: str
    value: float
    reference: str
    priority: str
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)


def parse_dose(raw: Any) -> float:
    """`"25,000"` -> 25000; faixa `"1-2"` -> maior valor; texto livre -> NaN."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return float("nan")
    text = str(raw).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        parts = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
        return float(parts.group(2)) if parts else float("nan")


def _route_group(route: Any) -> str:
    text = str(route).upper()
    if "DRIP" in text:
        return "IV DRIP"
    if text.startswith("IV"):
        return "IV"
    if text.startswith("PO") or "NG" in text:
        return "PO"
    return text


def prepare(prescriptions: pd.DataFrame) -> pd.DataFrame:
    """Normaliza os pedidos e mantém só medicamentos com dose numérica positiva."""
    frame = prescriptions.copy()
    frame = frame[frame["drug_type"].fillna("MAIN") == "MAIN"]
    frame = frame[~frame["drug"].str.contains(_BASE_PATTERN, na=False)]
    frame["dose"] = frame["dose_val_rx"].map(parse_dose)
    frame["unit"] = frame["dose_unit_rx"].fillna("").astype(str).str.strip().str.lower()
    frame["route_group"] = frame["route"].map(_route_group)
    frame["drug_key"] = frame["drug"].str.strip().str.lower()
    frame["per_day"] = pd.to_numeric(frame.get("doses_per_24_hrs"), errors="coerce")
    frame = frame.dropna(subset=["hadm_id"])
    frame["hadm_id"] = frame["hadm_id"].astype(int)
    return frame.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)


def _raise(priority: str, levels: int = 1) -> str:
    return PRIORITIES[min(PRIORITIES.index(priority) + levels, len(PRIORITIES) - 1)]


def _number(value: float) -> str:
    return f"{value:g}"


def detect_dose_changes(orders: pd.DataFrame, config: PrescriptionConfig) -> list[PrescriptionFinding]:
    """Compara cada dose com a anterior do mesmo medicamento, via e unidade."""
    findings = []
    dosed = orders[orders["dose"] > 0]
    keys = ["hadm_id", "drug_key", "route_group", "unit"]
    for (hadm_id, _, route, unit), group in dosed.groupby(keys, sort=False):
        # Pedidos no mesmo horário (ex.: ataque + manutenção) contam como um só.
        steps = group.groupby("starttime").agg(
            dose=("dose", "max"), per_day=("per_day", "max"), drug=("drug", "first"),
            subject_id=("subject_id", "first"),
        )
        if len(steps) < 2:
            continue
        times = list(steps.index)
        max_gap = pd.Timedelta(days=config.change_max_gap_days)
        for position in range(1, len(steps)):
            row, before, time = steps.iloc[position], steps.iloc[position - 1], times[position]
            if time - times[position - 1] > max_gap:
                continue
            # Dose diária quando os dois pedidos informam a frequência.
            daily = row["per_day"] > 0 and before["per_day"] > 0
            new = row["dose"] * (row["per_day"] if daily else 1)
            old = before["dose"] * (before["per_day"] if daily else 1)
            ratio = new / old
            titrated = bool(_TITRATED_PATTERN.search(row["drug"]))
            limit = config.titrated_ratio if titrated else config.change_ratio
            if 1 / limit < ratio < limit:
                continue
            high_alert = bool(_HIGH_ALERT_PATTERN.search(row["drug"]))
            priority = "medium" if high_alert else "low"
            strong = ratio >= config.strong_ratio or ratio <= 1 / config.strong_ratio
            if strong and not titrated:
                priority = _raise(priority)
            basis = "dose diária" if daily else "dose por administração"
            direction = "Aumento abrupto" if ratio > 1 else "Redução abrupta"
            findings.append(
                PrescriptionFinding(
                    kind="dose_change",
                    subject_id=int(row["subject_id"]),
                    hadm_id=int(hadm_id),
                    time=time.to_pydatetime(),
                    drug=row["drug"],
                    route=route,
                    unit=unit,
                    value=float(new),
                    reference=f"anterior: {_number(old)} {unit}",
                    priority=priority,
                    detail=(
                        f"{direction} de {basis} de {row['drug']} ({route}): "
                        f"{_number(old)} para {_number(new)} {unit} ({ratio:.1f}x)."
                    ),
                    extra={"ratio": round(float(ratio), 3), "high_alert": high_alert, "daily": bool(daily)},
                )
            )
    return findings


def population_reference(orders: pd.DataFrame, config: PrescriptionConfig) -> pd.DataFrame:
    """Mediana e dispersão robusta do log da dose por medicamento, via e unidade."""
    dosed = orders[orders["dose"] > 0].assign(log_dose=lambda f: np.log(f["dose"]))
    stats = dosed.groupby(["drug_key", "route_group", "unit"])["log_dose"].agg(
        n="count",
        median="median",
        mad=lambda values: float(np.median(np.abs(values - np.median(values)))),
    )
    return stats[stats["n"] >= config.population_min]


def detect_atypical_doses(orders: pd.DataFrame, config: PrescriptionConfig) -> list[PrescriptionFinding]:
    """Dose muito acima do que a população recebe do mesmo medicamento."""
    reference = population_reference(orders, config)
    if reference.empty:
        return []
    dosed = orders[orders["dose"] > 0].join(reference, on=["drug_key", "route_group", "unit"], how="inner")
    sigma = np.maximum(1.4826 * dosed["mad"], 0.15)  # piso: 15% de variação
    dosed = dosed.assign(z=(np.log(dosed["dose"]) - dosed["median"]) / sigma)
    typical = np.exp(dosed["median"])
    flagged = dosed[
        (dosed["z"] >= config.population_z) & (dosed["dose"] >= config.population_min_ratio * typical)
    ]
    findings = []
    for row in flagged.itertuples(index=False):
        median = math.exp(row.median)
        high_alert = bool(_HIGH_ALERT_PATTERN.search(row.drug))
        findings.append(
            PrescriptionFinding(
                kind="atypical_dose",
                subject_id=int(row.subject_id),
                hadm_id=int(row.hadm_id),
                time=row.starttime.to_pydatetime(),
                drug=row.drug,
                route=row.route_group,
                unit=row.unit,
                value=float(row.dose),
                reference=f"mediana pop.: {_number(round(median, 2))} {row.unit}",
                priority="high" if high_alert else "medium",
                detail=(
                    f"Dose de {row.drug} ({row.route_group}) de {_number(row.dose)} {row.unit}, "
                    f"{row.dose / median:.1f}x a mediana de {int(row.n)} pedidos do mesmo medicamento "
                    f"({_number(round(median, 2))} {row.unit})."
                ),
                extra={"z": round(float(row.z), 2), "population_n": int(row.n), "high_alert": high_alert},
            )
        )
    return findings


def detect_vasoactive_starts(orders: pd.DataFrame) -> list[PrescriptionFinding]:
    """Primeiro pedido de cada droga vasoativa na internação."""
    findings = []
    vaso = orders[
        orders["drug"].str.contains(_VASOACTIVE_PATTERN, na=False)
        & orders["route_group"].isin(["IV", "IV DRIP"])
    ]
    for hadm_id, group in vaso.groupby("hadm_id", sort=False):
        seen: list[str] = []
        for row in group.sort_values("starttime").itertuples(index=False):
            agent = _VASOACTIVE_PATTERN.search(row.drug).group(1).lower()
            if agent in seen:
                continue
            seen.append(agent)
            second = len(seen) > 1
            dose = row.dose if row.dose > 0 else 0.0
            findings.append(
                PrescriptionFinding(
                    kind="vasoactive_start",
                    subject_id=int(row.subject_id),
                    hadm_id=int(hadm_id),
                    time=row.starttime.to_pydatetime(),
                    drug=row.drug,
                    route=row.route_group,
                    unit=row.unit or "dose",
                    value=float(dose),
                    reference="sem vasoativo anterior" if not second else f"já em uso: {', '.join(seen[:-1])}"[:40],
                    priority="critical" if second else "high",
                    detail=(
                        f"Associação de novo vasoativo ({row.drug}) a esquema que já incluía "
                        f"{', '.join(seen[:-1])}."
                        if second
                        else f"Início de droga vasoativa ({row.drug}, {row.route_group}) nesta internação."
                    ),
                    extra={"agents": list(seen)},
                )
            )
    return findings


def detect_order_bursts(orders: pd.DataFrame, config: PrescriptionConfig) -> list[PrescriptionFinding]:
    """Blocos de 6 h com muito mais pedidos novos que o padrão da internação."""
    findings = []
    width = pd.Timedelta(hours=config.burst_hours)
    for hadm_id, group in orders.groupby("hadm_id", sort=False):
        start = group["starttime"].min()
        slot = ((group["starttime"] - start) // width).astype(int)
        counts = slot.value_counts().reindex(range(int(slot.max()) + 1), fill_value=0).sort_index()
        skip = config.burst_skip_hours // config.burst_hours
        for index in range(skip + config.burst_min_history, len(counts)):
            count = int(counts.iloc[index])
            if count < config.burst_min_orders:
                continue
            history = counts.iloc[skip:index].to_numpy()
            median = float(np.median(history))
            sigma = max(1.4826 * float(np.median(np.abs(history - median))), 1.0)
            z = (count - median) / sigma
            if z < config.burst_z:
                continue
            block = group[slot == index]
            drugs = block["drug"].value_counts().head(5).index.tolist()
            findings.append(
                PrescriptionFinding(
                    kind="order_burst",
                    subject_id=int(block["subject_id"].iloc[0]),
                    hadm_id=int(hadm_id),
                    time=block["starttime"].min().to_pydatetime(),
                    drug="novos pedidos de medicamento",
                    route="-",
                    unit="pedidos/6h",
                    value=float(count),
                    reference=f"mediana da internação: {_number(median)}",
                    priority="medium" if z >= 2 * config.burst_z else "low",
                    detail=(
                        f"{count} novos pedidos de medicamento em {config.burst_hours} h, contra mediana de "
                        f"{_number(median)} nos blocos anteriores da internação. Principais: "
                        f"{', '.join(drugs)}."
                    ),
                    extra={"z": round(z, 2), "drugs": drugs},
                )
            )
    return findings


def analyze_prescriptions(
    prescriptions: pd.DataFrame, config: PrescriptionConfig | None = None
) -> list[PrescriptionFinding]:
    """Roda os quatro detectores e devolve os achados em ordem cronológica."""
    config = config or PrescriptionConfig()
    orders = prepare(prescriptions)
    findings = [
        *detect_dose_changes(orders, config),
        *detect_atypical_doses(orders, config),
        *detect_vasoactive_starts(orders),
        *detect_order_bursts(orders, config),
    ]
    # Pedidos idênticos repetidos no mesmo horário viram um achado só.
    unique: dict[tuple, PrescriptionFinding] = {}
    for finding in findings:
        key = (finding.kind, finding.hadm_id, finding.time, finding.drug.lower(), finding.route)
        if key not in unique or finding.value > unique[key].value:
            unique[key] = finding
    return sorted(unique.values(), key=lambda f: (f.hadm_id, f.time, f.kind, f.drug))


def to_event(finding: PrescriptionFinding, context: EventContext | None = None) -> dict[str, Any]:
    """Converte um achado em evento `prescription_change` do contrato da Fase 4."""
    context = context or EventContext()
    stamp = finding.time.strftime("%Y%m%dT%H%M%S")
    drug_id = re.sub(r"[^a-z0-9]+", "-", finding.drug.lower()).strip("-")[:30]
    uri = f"clinical://mimic-iv-demo/prescriptions/hadm-{finding.hadm_id}/{stamp}/{drug_id}"
    evidence = [
        measurement(
            kind="prescription",
            uri=uri,
            metric=finding.drug,
            value=finding.value,
            unit=finding.unit or "dose",
            reference_range=finding.reference,
            description=f"Via {finding.route}. Detector: {finding.kind}.",
        )
    ]
    return build_event(
        module="clinical",
        patient_id=context.patient_id(finding.subject_id),
        local_id=f"rx-{finding.hadm_id}-{stamp}-{finding.kind}-{drug_id}-{finding.route}",
        timestamp=context.timestamp(finding.time),
        event_type="prescription_change",
        finding=finding.detail + " Alteração sinalizada para revisão; pode ser ajuste planejado.",
        anomaly_detected=True,
        priority=finding.priority,
        evidence=evidence,
        system=SYSTEM,
        model=MODEL_NAME,
    )
