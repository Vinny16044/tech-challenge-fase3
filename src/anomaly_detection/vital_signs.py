"""
Detecção de anomalias em séries temporais de sinais vitais.

Quatro camadas independentes avaliam cada observação, na ordem de chegada:

1. Faixas de referência — pontuação por sinal inspirada no NEWS2 (FR, SpO2,
   PAS, FC e temperatura). É um escore PARCIAL: não usa nível de consciência
   nem oxigênio suplementar.
2. Desvio do padrão individual — escore z robusto (mediana e MAD) de cada
   sinal contra as últimas observações do próprio paciente.
3. Padrão multivariado — Isolation Forest treinado na população (FC, FR,
   SpO2, PAS e índice de choque FC/PAS).
4. Tendência — piora rápida do escore em poucas observações.

O monitor é incremental (`VitalSignsMonitor.update`): guarda só o estado
recente de cada paciente e decide na hora, como num fluxo em tempo real. Um
episódio anômalo gera evento no início, quando a prioridade sobe e, se
persistir, a cada `realert_minutes`; as demais observações do episódio são
suprimidas para não inundar a equipe.

Os limites são de protótipo: não foram validados clinicamente.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection.events import EventContext, build_event, measurement

SYSTEM = "fase4-clinical-anomaly"
MODEL_NAME = "faixas NEWS2 parcial + z robusto + IsolationForest"

SCORED_SIGNALS: tuple[str, ...] = ("hr", "rr", "spo2", "sbp", "temp_c")
MODEL_SIGNALS: tuple[str, ...] = ("hr", "rr", "spo2", "sbp")
MODEL_FEATURES: tuple[str, ...] = (*MODEL_SIGNALS, "shock_index")

LABELS = {"hr": "FC", "rr": "FR", "spo2": "SpO2", "sbp": "PAS", "temp_c": "Temp"}
METRICS = {
    "hr": "heart_rate",
    "rr": "respiratory_rate",
    "spo2": "spo2",
    "sbp": "systolic_bp",
    "temp_c": "temperature",
}
UNITS = {"hr": "bpm", "rr": "irpm", "spo2": "%", "sbp": "mmHg", "temp_c": "°C"}
REFERENCE_RANGE = {
    "hr": "51-90",
    "rr": "12-20",
    "spo2": "96-100",
    "sbp": "111-219",
    "temp_c": "36.1-38.0",
}

# (limite superior da faixa, pontos). Faixas do NEWS2 (Royal College of
# Physicians, 2017), escala 1 de SpO2.
_BANDS: dict[str, tuple[tuple[float, int], ...]] = {
    "rr": ((8, 3), (11, 1), (20, 0), (24, 2), (math.inf, 3)),
    "spo2": ((91, 3), (93, 2), (95, 1), (math.inf, 0)),
    "sbp": ((90, 3), (100, 2), (110, 1), (219, 0), (math.inf, 3)),
    "hr": ((40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (math.inf, 3)),
    "temp_c": ((35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (math.inf, 2)),
}

PRIORITIES = ("low", "medium", "high", "critical")

LAYER_RANGE = "faixa_referencia"
LAYER_BASELINE = "desvio_individual"
LAYER_MULTIVARIATE = "padrao_multivariado"
LAYER_TREND = "tendencia"


def band_points(signal: str, value: float) -> int:
    """Pontos do sinal na faixa de referência (0 = dentro do esperado)."""
    for upper, points in _BANDS[signal]:
        if value <= upper:
            return points
    return 0


@dataclass(frozen=True)
class VitalConfig:
    baseline_window: int = 12  # observações usadas como padrão individual
    baseline_min: int = 6  # mínimo para calcular o desvio
    z_threshold: float = 3.5
    # Variação mínima para um desvio contar, mesmo com z alto (série muito estável).
    min_delta: dict[str, float] = field(
        default_factory=lambda: {"hr": 20, "rr": 8, "spo2": 4, "sbp": 25, "temp_c": 1.0}
    )
    # Piso do desvio-padrão robusto, para a MAD nula não inflar o z.
    min_sigma: dict[str, float] = field(
        default_factory=lambda: {"hr": 3, "rr": 1.5, "spo2": 1, "sbp": 5, "temp_c": 0.2}
    )
    carry_minutes: int = 120  # por quanto tempo a última medida ainda vale
    medium_score: int = 5
    high_score: int = 7
    red_alone: bool = True  # um sinal com 3 pontos basta para a camada de faixas
    trend_points: int = 3  # aumento do escore que caracteriza piora rápida
    trend_lookback: int = 3
    trend_min_score: int = 3
    realert_minutes: int = 120
    clear_after: int = 2  # observações normais seguidas que encerram o episódio

    @classmethod
    def ward(cls) -> "VitalConfig":
        """Limiares do NEWS2 para enfermaria: escore 5 (médio) e 7 (alto)."""
        return cls()

    @classmethod
    def icu(cls) -> "VitalConfig":
        """
        Perfil para terapia intensiva, onde o paciente já parte de um escore
        alto: limiares 7 e 9, e um sinal isolado não basta para a camada de
        faixas. Reduz os avisos repetidos na população do MIMIC-IV (UTI).
        """
        return cls(medium_score=7, high_score=9, red_alone=False, trend_points=4, trend_min_score=5)


PROFILES = {"ward": VitalConfig.ward, "icu": VitalConfig.icu}


class PopulationModel:
    """Isolation Forest sobre o conjunto de sinais vitais da população."""

    def __init__(self, contamination: float = 0.02, n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._forest = None
        self.trained_rows = 0

    @staticmethod
    def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Linhas completas de FC, FR, SpO2 e PAS, com o índice de choque."""
        features = frame[list(MODEL_SIGNALS)].dropna().astype(float)
        features = features.assign(shock_index=features["hr"] / features["sbp"])
        return features[list(MODEL_FEATURES)]

    def fit(self, carried: pd.DataFrame) -> "PopulationModel":
        from sklearn.ensemble import IsolationForest

        features = self.feature_frame(carried)
        if len(features) < 50:
            raise ValueError("poucas observações completas para treinar o modelo")
        self._forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(features.to_numpy())
        self.trained_rows = len(features)
        return self

    def score_frame(self, carried: pd.DataFrame) -> pd.Series:
        """Escore por linha (negativo = atípico); NaN onde falta algum sinal."""
        scores = pd.Series(np.nan, index=carried.index, dtype=float)
        features = self.feature_frame(carried)
        if self._forest is not None and len(features):
            scores.loc[features.index] = self._forest.decision_function(features.to_numpy())
        return scores

    def score(self, values: dict[str, float]) -> float | None:
        if self._forest is None or any(values.get(name) is None for name in MODEL_SIGNALS):
            return None
        row = [float(values[name]) for name in MODEL_SIGNALS]
        row.append(row[0] / row[3])
        return float(self._forest.decision_function(np.array([row]))[0])


def carry_forward(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Repete a última medida de cada sinal por até `minutes`, dentro da internação.

    É a mesma regra que o monitor aplica em tempo real: a pressão costuma ser
    medida com menos frequência que a FC, e a observação precisa do conjunto.
    """
    carried = frame.copy()
    limit = pd.Timedelta(minutes=minutes)
    for signal in SCORED_SIGNALS:
        measured_at = carried["charttime"].where(carried[signal].notna())
        last_time = measured_at.groupby(carried["stay_id"]).ffill()
        filled = carried[signal].groupby(carried["stay_id"]).ffill()
        carried[signal] = filled.where((carried["charttime"] - last_time) <= limit)
    return carried


@dataclass
class VitalFinding:
    """Resultado da avaliação de uma observação."""

    patient_key: str
    timestamp: datetime
    measured: dict[str, float]
    current: dict[str, float]
    points: dict[str, int]
    score: int
    red_signals: list[str]
    deviations: list[dict[str, Any]]
    multivariate_score: float | None
    multivariate_flag: bool
    trend_rise: int | None
    layers: list[str]
    anomaly: bool
    priority: str
    emit: bool
    emit_reason: str | None
    meta: dict[str, Any] = field(default_factory=dict)


class _PatientState:
    def __init__(self, config: VitalConfig):
        self.history = {s: deque(maxlen=config.baseline_window) for s in SCORED_SIGNALS}
        self.last: dict[str, tuple[datetime, float]] = {}
        self.scores: deque[int] = deque(maxlen=config.trend_lookback)
        self.active = False
        self.normal_streak = 0
        self.last_emit: datetime | None = None
        self.last_priority = "low"


_UNSET = object()


class VitalSignsMonitor:
    """Avalia observações de sinais vitais uma a uma, mantendo estado por paciente."""

    def __init__(self, config: VitalConfig | None = None, model: PopulationModel | None = None):
        self.config = config or VitalConfig()
        self.model = model
        self._states: dict[str, _PatientState] = {}

    def update(
        self,
        patient_key: str,
        timestamp: datetime,
        values: dict[str, float | None],
        *,
        multivariate_score: Any = _UNSET,
        meta: dict[str, Any] | None = None,
    ) -> VitalFinding:
        """
        Avalia uma observação. `values` traz os sinais medidos neste horário.

        `multivariate_score` permite informar o escore do Isolation Forest já
        calculado em lote; sem ele, o monitor consulta o modelo.
        """
        config = self.config
        state = self._states.setdefault(str(patient_key), _PatientState(config))

        measured = {
            signal: float(value)
            for signal, value in values.items()
            if signal in SCORED_SIGNALS and value is not None and not math.isnan(float(value))
        }
        for signal, value in measured.items():
            state.last[signal] = (timestamp, value)
        carry = timedelta(minutes=config.carry_minutes)
        current = {
            signal: value
            for signal, (measured_at, value) in state.last.items()
            if timestamp - measured_at <= carry
        }

        # 1. Faixas de referência.
        points = {signal: band_points(signal, current[signal]) for signal in SCORED_SIGNALS if signal in current}
        score = sum(points.values())
        red_signals = [signal for signal, value in points.items() if value == 3]

        # 2. Desvio do padrão individual (só para o que foi medido agora).
        deviations = []
        for signal, value in measured.items():
            history = state.history[signal]
            if len(history) >= config.baseline_min:
                median = float(np.median(history))
                mad = float(np.median(np.abs(np.asarray(history) - median)))
                sigma = max(1.4826 * mad, config.min_sigma[signal])
                delta = value - median
                z = delta / sigma
                relevant = signal != "spo2" or delta < 0  # SpO2 subindo não preocupa
                if relevant and abs(z) >= config.z_threshold and abs(delta) >= config.min_delta[signal]:
                    deviations.append(
                        {
                            "signal": signal,
                            "value": value,
                            "median": round(median, 2),
                            "delta": round(delta, 2),
                            "z": round(z, 2),
                        }
                    )
            history.append(value)

        # 3. Padrão multivariado.
        if multivariate_score is _UNSET:
            multivariate_score = self.model.score(current) if self.model else None
        if multivariate_score is not None and math.isnan(multivariate_score):
            multivariate_score = None
        multivariate_flag = multivariate_score is not None and multivariate_score < 0

        # 4. Tendência do escore.
        trend_rise = None
        if len(state.scores) >= config.trend_lookback:
            rise = score - min(state.scores)
            if rise >= config.trend_points and score >= config.trend_min_score:
                trend_rise = rise
        state.scores.append(score)

        layers = []
        if score >= config.medium_score or (config.red_alone and red_signals):
            layers.append(LAYER_RANGE)
        if deviations:
            layers.append(LAYER_BASELINE)
        if multivariate_flag:
            layers.append(LAYER_MULTIVARIATE)
        if trend_rise is not None:
            layers.append(LAYER_TREND)
        anomaly = bool(layers)

        # Prioridade sugerida: o escore define a base; um sinal estatístico
        # (mudança abrupta, padrão atípico ou piora) sobe um nível.
        if score >= config.high_score:
            rank = 2
        elif LAYER_RANGE in layers:
            rank = 1
        else:
            rank = 0
        if rank >= 1 and len(layers) > 1:
            rank += 1
        priority = PRIORITIES[min(rank, 3)]

        emit, reason = self._episode(state, timestamp, anomaly, priority)
        return VitalFinding(
            patient_key=str(patient_key),
            timestamp=timestamp,
            measured=measured,
            current=current,
            points=points,
            score=score,
            red_signals=red_signals,
            deviations=deviations,
            multivariate_score=multivariate_score,
            multivariate_flag=multivariate_flag,
            trend_rise=trend_rise,
            layers=layers,
            anomaly=anomaly,
            priority=priority if anomaly else "low",
            emit=emit,
            emit_reason=reason,
            meta=dict(meta or {}),
        )

    def _episode(
        self, state: _PatientState, timestamp: datetime, anomaly: bool, priority: str
    ) -> tuple[bool, str | None]:
        """Decide se a observação vira evento (início, agravamento ou persistência)."""
        if not anomaly:
            state.normal_streak += 1
            if state.active and state.normal_streak >= self.config.clear_after:
                state.active = False
            return False, None

        state.normal_streak = 0
        reason = None
        if not state.active:
            reason = "início do episódio"
        elif PRIORITIES.index(priority) > PRIORITIES.index(state.last_priority):
            reason = "agravamento"
        elif state.last_emit and timestamp - state.last_emit >= timedelta(
            minutes=self.config.realert_minutes
        ):
            reason = "persistência"
        if reason is None:
            return False, None
        state.active = True
        state.last_emit = timestamp
        state.last_priority = priority
        return True, reason


def analyze_frame(
    frame: pd.DataFrame,
    config: VitalConfig | None = None,
    model: PopulationModel | None = None,
) -> list[VitalFinding]:
    """
    Avalia todas as observações de `frame` (saída de `load_vitals`).

    Reproduz o fluxo em tempo real: cada internação é percorrida em ordem
    cronológica pelo mesmo monitor incremental. O Isolation Forest é aplicado
    em lote por desempenho, com a mesma regra de validade das medidas.
    """
    config = config or VitalConfig()
    frame = frame.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    scores = None
    if model is not None:
        scores = model.score_frame(carry_forward(frame, config.carry_minutes))

    monitor = VitalSignsMonitor(config, model=None)
    findings = []
    for index, row in enumerate(frame.itertuples(index=False)):
        values = {signal: getattr(row, signal) for signal in SCORED_SIGNALS}
        score = None if scores is None else float(scores.iloc[index])
        findings.append(
            monitor.update(
                str(row.stay_id),
                row.charttime.to_pydatetime(),
                values,
                multivariate_score=score,
                meta={
                    "subject_id": int(row.subject_id),
                    "hadm_id": int(row.hadm_id),
                    "stay_id": int(row.stay_id),
                },
            )
        )
    return findings


def _format(signal: str, value: float) -> str:
    text = f"{value:.1f}" if signal == "temp_c" else f"{value:.0f}"
    return f"{LABELS[signal]} {text} {UNITS[signal]}"


def describe(finding: VitalFinding) -> str:
    """Texto objetivo do achado, sem diagnóstico."""
    flagged = [s for s in SCORED_SIGNALS if finding.points.get(s, 0) > 0]
    parts = []
    if flagged:
        parts.append(
            "Sinais vitais fora da faixa de referência: "
            + ", ".join(_format(s, finding.current[s]) for s in flagged)
            + f". Escore parcial inspirado no NEWS2: {finding.score}."
        )
    else:
        parts.append(f"Sinais vitais dentro das faixas de referência (escore {finding.score}).")
    if finding.deviations:
        parts.append(
            "Desvio do padrão individual: "
            + "; ".join(
                f"{LABELS[d['signal']]} {d['delta']:+.0f} {UNITS[d['signal']]} em relação à mediana "
                f"recente ({d['median']:.0f})"
                for d in finding.deviations
            )
            + "."
        )
    if finding.multivariate_flag:
        parts.append("Combinação de sinais atípica para a população (Isolation Forest).")
    if finding.trend_rise is not None:
        parts.append(f"Piora rápida: escore subiu {finding.trend_rise} pontos nas últimas observações.")
    if finding.emit_reason:
        parts.append(f"Motivo do aviso: {finding.emit_reason}.")
    return " ".join(parts)


def to_event(finding: VitalFinding, context: EventContext | None = None) -> dict[str, Any]:
    """Converte um achado em evento `vital_signs` do contrato da Fase 4."""
    context = context or EventContext()
    meta = finding.meta
    subject = meta.get("subject_id", finding.patient_key)
    stay = meta.get("stay_id", finding.patient_key)
    stamp = finding.timestamp.strftime("%Y%m%dT%H%M%S")
    uri = f"clinical://mimic-iv-demo/chartevents/stay-{stay}/{stamp}"

    deviations = {d["signal"]: d for d in finding.deviations}
    evidence = []
    for signal in SCORED_SIGNALS:
        if signal not in finding.current:
            continue
        points = finding.points.get(signal, 0)
        in_model = finding.multivariate_flag and signal in MODEL_SIGNALS
        if finding.anomaly and points == 0 and signal not in deviations and not in_model:
            continue
        notes = [f"{points} ponto(s) na faixa de referência"]
        if signal in deviations:
            d = deviations[signal]
            notes.append(f"z robusto {d['z']:+.1f} contra mediana individual {d['median']:.1f}")
        if signal not in finding.measured:
            notes.append("última medida válida, repetida")
        evidence.append(
            measurement(
                uri=uri,
                metric=METRICS[signal],
                value=finding.current[signal],
                unit=UNITS[signal],
                reference_range=REFERENCE_RANGE[signal],
                description="; ".join(notes) + ".",
            )
        )
    evidence.append(
        measurement(
            uri=uri,
            metric="news2_partial_score",
            value=finding.score,
            unit="pontos",
            reference_range="0-4",
            description="Soma dos pontos de FR, SpO2, PAS, FC e temperatura; sem consciência e O2.",
        )
    )
    if finding.multivariate_score is not None and finding.multivariate_flag:
        evidence.append(
            measurement(
                uri=uri,
                metric="isolation_forest_score",
                value=finding.multivariate_score,
                unit="score",
                reference_range=">= 0",
                description="Escore do Isolation Forest (FC, FR, SpO2, PAS, FC/PAS); negativo = atípico.",
            )
        )

    return build_event(
        module="clinical",
        patient_id=context.patient_id(subject),
        local_id=f"vs-{stay}-{stamp}",
        timestamp=context.timestamp(finding.timestamp),
        event_type="vital_signs" if finding.anomaly else "clinical_check",
        finding=describe(finding),
        anomaly_detected=finding.anomaly,
        priority=finding.priority,
        evidence=evidence,
        system=SYSTEM,
        model=MODEL_NAME,
    )
