"""
Padrões de movimentação do paciente durante a internação.

Duas fontes, dois detectores:

1. Atividade corporal medida pelo módulo de vídeo (`detect_activity_patterns`).
   Recebe a série do índice de atividade (deslocamento médio das articulações,
   em alturas de corpo por segundo) e a postura de cada instante, e aponta:
   - `prolonged_immobility`: paciente deitado e praticamente sem movimento por
     mais tempo que o limite — curto fora do leito, longo no leito;
   - `agitation`: movimento intenso e sustentado sem deslocamento pelo quarto.

2. Movimentações entre unidades (`analyze_transfers`, hosp/transfers do
   MIMIC-IV): retorno à UTI pouco depois da alta da UTI e trocas de unidade
   muito frequentes.

Os limites são de protótipo e configuráveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from src.anomaly_detection.events import EventContext, build_event, measurement

SYSTEM = "fase4-movement-patterns"

# --------------------------------------------------------------------------
# 1. Atividade corporal (a partir do vídeo)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityConfig:
    still_threshold: float = 0.1  # abaixo disso o corpo está parado (alturas de corpo/s)
    floor_immobility_seconds: float = 2.0  # deitado fora do leito
    bed_immobility_seconds: float = 2 * 3600.0  # deitado no leito (risco de lesão por pressão)
    agitation_threshold: float = 0.45  # atividade dos membros que conta como intensa
    agitation_seconds: float = 4.0
    agitation_max_travel: float = 0.35  # deslocamento do corpo acima disso é caminhada
    agitation_smooth_seconds: float = 0.6  # média móvel: movimento de vaivém oscila a cada quadro
    bridge_seconds: float = 0.4  # falhas curtas não interrompem um trecho


@dataclass
class MovementFinding:
    kind: str  # prolonged_immobility | agitation
    start_s: float
    end_s: float
    mean_activity: float
    location: str  # floor | support | unknown
    priority: str
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def runs(mask: Sequence[bool], times: Sequence[float], bridge: float = 0.0) -> list[tuple[int, int]]:
    """
    Trechos contínuos em que `mask` é verdadeiro, como pares (início, fim) de índices.

    Interrupções mais curtas que `bridge` segundos são ignoradas.
    """
    mask = np.asarray(mask, dtype=bool)
    times = np.asarray(times, dtype=float)
    spans: list[list[int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if not value and start is not None:
            spans.append([start, index - 1])
            start = None
    if start is not None:
        spans.append([start, len(mask) - 1])

    merged: list[list[int]] = []
    for span in spans:
        if merged and times[span[0]] - times[merged[-1][1]] <= bridge:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [(a, b) for a, b in merged]


def detect_activity_patterns(
    times: Sequence[float],
    activity: Sequence[float],
    lying: Sequence[bool],
    on_support: Sequence[bool] | None = None,
    travel: Sequence[float] | None = None,
    config: ActivityConfig | None = None,
) -> list[MovementFinding]:
    """
    Aponta imobilidade prolongada e agitação numa série de atividade.

    `times` em segundos; `activity` é o índice de atividade dos membros;
    `lying` marca postura deitada; `on_support` marca sobreposição com leito,
    sofá ou cadeira; `travel` é a velocidade de deslocamento do corpo.
    """
    config = config or ActivityConfig()
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)
    lying = np.asarray(lying, dtype=bool)
    if len(times) < 2:
        return []
    on_support = np.zeros(len(times), bool) if on_support is None else np.asarray(on_support, bool)
    travel = np.zeros(len(times)) if travel is None else np.asarray(travel, dtype=float)
    known = ~np.isnan(activity)
    findings: list[MovementFinding] = []

    # Imobilidade: deitado e parado.
    still = known & lying & (activity < config.still_threshold)
    for start, end in runs(still, times, config.bridge_seconds):
        duration = times[end] - times[start]
        supported = float(np.mean(on_support[start : end + 1])) >= 0.5
        limit = config.bed_immobility_seconds if supported else config.floor_immobility_seconds
        if duration < limit:
            continue
        mean = float(np.nanmean(activity[start : end + 1]))
        reaches_end = end >= len(times) - 2
        if supported:
            priority = "medium"
            detail = f"Paciente deitado no leito sem mudança de posição por {duration / 60:.0f} min."
        else:
            priority = "high"
            detail = (
                f"Paciente deitado fora do leito e praticamente imóvel por {duration:.1f} s"
                + (" até o fim do registro." if reaches_end else ".")
            )
        findings.append(
            MovementFinding(
                kind="prolonged_immobility",
                start_s=float(times[start]),
                end_s=float(times[end]),
                mean_activity=mean,
                location="support" if supported else "floor",
                priority=priority,
                detail=detail,
                extra={"until_end": bool(reaches_end)},
            )
        )

    # Agitação: atividade alta e sustentada, sem deslocamento pelo ambiente.
    # O movimento de vaivém zera a cada inversão, então a série é suavizada.
    smooth = np.full(len(times), np.nan)
    half = config.agitation_smooth_seconds / 2
    for index in range(len(times)):
        window = known & (np.abs(times - times[index]) <= half)
        if window.any():
            smooth[index] = float(activity[window].mean())
    valid = activity[known]
    median = float(np.median(valid)) if len(valid) else 0.0
    intense = known & (smooth >= config.agitation_threshold) & (travel < config.agitation_max_travel)
    for start, end in runs(intense, times, config.bridge_seconds):
        duration = times[end] - times[start]
        if duration < config.agitation_seconds:
            continue
        mean = float(np.nanmean(activity[start : end + 1]))
        findings.append(
            MovementFinding(
                kind="agitation",
                start_s=float(times[start]),
                end_s=float(times[end]),
                mean_activity=mean,
                location="support" if float(np.mean(on_support[start : end + 1])) >= 0.5 else "unknown",
                priority="medium",
                detail=(
                    f"Movimentação intensa e contínua por {duration:.1f} s sem deslocamento pelo "
                    f"ambiente (atividade média {mean:.2f}, mediana do registro {median:.2f})."
                ),
                extra={"record_median": round(median, 3)},
            )
        )
    return sorted(findings, key=lambda f: f.start_s)


# --------------------------------------------------------------------------
# 2. Movimentações entre unidades (dados clínicos)
# --------------------------------------------------------------------------

ICU_PATTERN = r"Intensive Care|ICU|CCU"
_NOT_A_UNIT = ("Emergency Department", "Discharge Lounge", "Emergency Department Observation")


@dataclass(frozen=True)
class TransferConfig:
    icu_return_hours: float = 72.0  # retorno à UTI dentro deste prazo
    min_outside_hours: float = 1.0  # passagens mais curtas são artefato de registro
    frequent_count: int = 3  # trocas de unidade...
    frequent_hours: float = 24.0  # ...neste intervalo


@dataclass
class TransferFinding:
    kind: str  # icu_return | frequent_transfers
    subject_id: int
    hadm_id: int
    time: datetime
    value: float
    unit: str
    reference: str
    priority: str
    detail: str


def analyze_transfers(transfers, config: TransferConfig | None = None) -> list[TransferFinding]:
    """Retornos precoces à UTI e trocas de unidade muito frequentes, por internação."""
    config = config or TransferConfig()
    frame = transfers.dropna(subset=["hadm_id", "careunit"]).copy()
    frame = frame[~frame["careunit"].isin(_NOT_A_UNIT)]
    frame["icu"] = frame["careunit"].str.contains(ICU_PATTERN, regex=True)
    findings: list[TransferFinding] = []

    for hadm_id, group in frame.groupby("hadm_id", sort=False):
        group = group.sort_values("intime")
        # Mudanças de leito na mesma unidade não contam como troca de unidade.
        stays = group[group["careunit"] != group["careunit"].shift()].reset_index(drop=True)
        subject = int(stays["subject_id"].iloc[0])

        left_icu_at = None
        for position, row in enumerate(stays.itertuples(index=False)):
            if position == 0:
                continue
            previous = stays.iloc[position - 1]
            if previous["icu"] and not row.icu:
                left_icu_at = row.intime
            elif row.icu and not previous["icu"] and left_icu_at is not None:
                hours = (row.intime - left_icu_at).total_seconds() / 3600
                if config.min_outside_hours <= hours <= config.icu_return_hours:
                    findings.append(
                        TransferFinding(
                            kind="icu_return",
                            subject_id=subject,
                            hadm_id=int(hadm_id),
                            time=row.intime.to_pydatetime(),
                            value=round(hours, 1),
                            unit="h",
                            reference=f"> {config.icu_return_hours:g} h",
                            priority="high",
                            detail=(
                                f"Retorno à terapia intensiva ({row.careunit}) {hours:.0f} h depois da saída "
                                f"para {previous['careunit']}."
                            ),
                        )
                    )
                left_icu_at = None

        times = stays["intime"].tolist()
        window = np.timedelta64(int(config.frequent_hours * 3600), "s")
        reported_until = None
        for end in range(config.frequent_count, len(times)):
            start = end - config.frequent_count
            if times[end] - times[start] > window:
                continue
            if reported_until is not None and times[start] <= reported_until:
                continue
            reported_until = times[end]
            units = stays["careunit"].iloc[start : end + 1].tolist()
            findings.append(
                TransferFinding(
                    kind="frequent_transfers",
                    subject_id=subject,
                    hadm_id=int(hadm_id),
                    time=times[end].to_pydatetime(),
                    value=float(config.frequent_count),
                    unit="trocas/24h",
                    reference=f"< {config.frequent_count}",
                    priority="low",
                    detail=(
                        f"{config.frequent_count} trocas de unidade em menos de "
                        f"{config.frequent_hours:g} h: {' > '.join(units)}."
                    ),
                )
            )
    return sorted(findings, key=lambda f: (f.hadm_id, f.time))


def transfer_to_event(finding: TransferFinding, context: EventContext | None = None) -> dict[str, Any]:
    """Converte um achado de movimentação em evento `patient_transfer`."""
    context = context or EventContext()
    stamp = finding.time.strftime("%Y%m%dT%H%M%S")
    metric = "hours_since_icu_discharge" if finding.kind == "icu_return" else "unit_transfers_24h"
    return build_event(
        module="clinical",
        patient_id=context.patient_id(finding.subject_id),
        local_id=f"tr-{finding.hadm_id}-{stamp}-{finding.kind}",
        timestamp=context.timestamp(finding.time),
        event_type="patient_transfer",
        finding=finding.detail + " Padrão de movimentação sinalizado para revisão.",
        anomaly_detected=True,
        priority=finding.priority,
        evidence=[
            measurement(
                uri=f"clinical://mimic-iv-demo/transfers/hadm-{finding.hadm_id}/{stamp}",
                metric=metric,
                value=finding.value,
                unit=finding.unit,
                reference_range=finding.reference,
                description=f"Detector: {finding.kind}.",
            )
        ],
        system=SYSTEM,
        model="regras sobre a sequência de unidades",
    )
