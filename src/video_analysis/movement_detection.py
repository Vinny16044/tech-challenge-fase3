"""
Detecção de movimentos e eventos fora do padrão a partir das poses.

Trabalha só sobre as observações (`Observation`): não precisa do vídeo nem dos
modelos, então as regras podem ser testadas com poses sintéticas.

Para cada pessoa rastreada:

1. Postura por quadro — `upright` (em pé), `low` (sentado, agachado,
   curvado) ou `lying` (deitado) — pela inclinação do tronco e pela proporção
   da caixa, com filtro de maioria para remover oscilações.
2. Queda (`fall_detected`): entrada na postura deitada, fora de cama ou sofá,
   precedida de descida rápida e ampla da parte superior do corpo.
3. Saída do leito (`bed_exit_attempt`): estava deitado no leito e passa a
   ficar em pé ou fora da área do leito.
4. Imobilidade e agitação: série de atividade corporal avaliada por
   `src.anomaly_detection.movement_patterns`.

As distâncias são normalizadas pela altura da pessoa em pé (`ref_h`), para
que os limites não dependam da distância até a câmera.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.anomaly_detection.movement_patterns import ActivityConfig, detect_activity_patterns, runs
from src.video_analysis.pose import INDEX, Observation, group_tracks, trunk_angle

UPRIGHT, LOW, LYING = "upright", "low", "lying"
POSTURE_LABELS = {UPRIGHT: "em pé", LOW: "sentado/agachado/curvado", LYING: "deitado"}


@dataclass(frozen=True)
class MovementConfig:
    keypoint_conf: float = 0.35
    min_track_seconds: float = 0.5  # trilhas mais curtas são ruído do detector
    # Postura
    lying_aspect: float = 1.1  # largura/altura da caixa
    lying_angle: float = 55.0  # tronco quase horizontal...
    lying_angle_aspect: float = 0.75  # ...com caixa já larga (exclui quem só se curva)
    upright_aspect: float = 0.6
    upright_angle: float = 30.0
    smooth_frames: int = 5
    # Queda
    lying_confirm_seconds: float = 0.4
    fall_window_seconds: float = 1.5  # janela antes de deitar onde se mede a descida
    fall_settle_seconds: float = 1.0  # e depois, até o corpo chegar ao chão
    fall_min_drop: float = 0.45  # descida mínima, em alturas de corpo
    fall_min_speed: float = 0.7  # velocidade de pico mínima, em alturas de corpo/s
    speed_lag_seconds: float = 0.3
    support_overlap: float = 0.5  # fração da pessoa sobre cama/sofá para contar como apoiada
    # Saída do leito
    bed_min_seconds: float = 1.0
    bed_exit_window_seconds: float = 15.0
    bed_exit_confirm_seconds: float = 0.5
    bed_exit_overlap: float = 0.35
    # Atividade
    activity: ActivityConfig = field(default_factory=ActivityConfig)


@dataclass
class VideoEvent:
    kind: str  # fall_detected | bed_exit_attempt | prolonged_immobility | agitation | video_check
    track_id: int
    start_s: float
    end_s: float
    frame_index: int
    confidence: float
    priority: str
    anomaly: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "track_id": self.track_id,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "frame_index": self.frame_index,
            "confidence": round(self.confidence, 3),
            "priority": self.priority,
            "anomaly": self.anomaly,
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass
class TrackSeries:
    """Séries por quadro de uma pessoa, usadas pelas regras e pelo relatório."""

    track_id: int
    frames: np.ndarray
    times: np.ndarray
    posture: list[str]
    angle: np.ndarray  # NaN quando o tronco não foi localizado
    aspect: np.ndarray
    ref_h: np.ndarray
    upper_y: np.ndarray  # altura da parte superior do corpo (pixels, cresce para baixo)
    on_support: np.ndarray
    support_overlap: np.ndarray
    activity: np.ndarray  # atividade total (alturas de corpo/s)
    limb_activity: np.ndarray
    travel: np.ndarray
    confidence: np.ndarray

    def posture_share(self) -> dict[str, float]:
        counts = Counter(self.posture)
        total = max(len(self.posture), 1)
        return {name: round(counts.get(name, 0) / total, 3) for name in (UPRIGHT, LOW, LYING)}

    def summary(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "observations": int(len(self.times)),
            "start_s": round(float(self.times[0]), 3),
            "end_s": round(float(self.times[-1]), 3),
            "posture_share": self.posture_share(),
            "mean_activity": round(float(np.nanmean(self.activity)), 3) if np.isfinite(self.activity).any() else None,
            "mean_confidence": round(float(np.mean(self.confidence)), 3),
        }


def _raw_posture(angle: float | None, aspect: float, config: MovementConfig) -> str:
    horizontal_trunk = angle is not None and angle >= config.lying_angle
    if aspect >= config.lying_aspect or (horizontal_trunk and aspect >= config.lying_angle_aspect):
        return LYING
    if aspect <= config.upright_aspect and (angle is None or angle <= config.upright_angle):
        return UPRIGHT
    return LOW


def _majority(labels: list[str], window: int) -> list[str]:
    if window <= 1 or len(labels) < 3:
        return list(labels)
    half = window // 2
    smoothed = []
    for index in range(len(labels)):
        neighbours = labels[max(0, index - half) : index + half + 1]
        smoothed.append(Counter(neighbours).most_common(1)[0][0])
    return smoothed


def _reference_height(observations: list[Observation], aspects: np.ndarray, config: MovementConfig) -> np.ndarray:
    """
    Altura da pessoa em pé ao longo do tempo, em pixels.

    É a maior altura de caixa "alta" (pessoa ereta) nos 3 s anteriores. Sem
    quadro ereto recente, vale a última referência; sem nenhuma (a pessoa já
    começa sentada ou deitada), estima-se pelo comprimento do tronco.
    """
    times = np.array([o.time_s for o in observations])
    heights = np.array([o.height for o in observations])
    reference = np.full(len(observations), np.nan)
    last = math.nan
    for index, observation in enumerate(observations):
        recent = (times >= times[index] - 3.0) & (times <= times[index]) & (aspects <= config.upright_aspect)
        if recent.any():
            last = float(heights[recent].max())
        elif math.isnan(last):
            shoulders = observation.midpoint("left_shoulder", "right_shoulder", config.keypoint_conf)
            hips = observation.midpoint("left_hip", "right_hip", config.keypoint_conf)
            torso = math.dist(shoulders, hips) if shoulders and hips else 0.0
            last = max(observation.height, observation.width, 3.0 * torso)
        reference[index] = last
    return reference


def _activity(
    observations: list[Observation], ref_h: np.ndarray, config: MovementConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Atividade total, atividade dos membros e deslocamento do corpo.

    Compara cada observação com a de ~`speed_lag_seconds` antes, para que o
    tremor quadro a quadro do estimador de pose não conte como movimento.
    """
    count = len(observations)
    times = np.array([o.time_s for o in observations])
    total = np.full(count, np.nan)
    limbs = np.full(count, np.nan)
    travel = np.full(count, np.nan)
    lag = config.speed_lag_seconds
    past = 0
    for index in range(count):
        while past + 1 < index and times[past + 1] <= times[index] - lag:
            past += 1
        dt = times[index] - times[past]
        if index == past or dt < lag * 0.6 or dt > lag * 3:
            continue
        now, before = observations[index], observations[past]
        scale = ref_h[index] * dt
        shift = (now.center[0] - before.center[0], now.center[1] - before.center[1])
        travel[index] = math.hypot(*shift) / scale
        moves, relative = [], []
        for (x1, y1, c1), (x0, y0, c0) in zip(now.keypoints, before.keypoints):
            if c1 >= config.keypoint_conf and c0 >= config.keypoint_conf:
                moves.append(math.hypot(x1 - x0, y1 - y0))
                relative.append(math.hypot(x1 - x0 - shift[0], y1 - y0 - shift[1]))
        if len(moves) >= 4:
            total[index] = float(np.mean(moves)) / scale
            limbs[index] = float(np.mean(relative)) / scale
        else:
            total[index] = travel[index]
            limbs[index] = 0.0
    return total, limbs, travel


def build_series(track_id: int, observations: list[Observation], config: MovementConfig) -> TrackSeries:
    """Calcula as séries de postura e movimento de uma pessoa."""
    angles = [trunk_angle(o, config.keypoint_conf) for o in observations]
    aspects = np.array([o.width / max(o.height, 1e-6) for o in observations])
    raw = [_raw_posture(angle, aspect, config) for angle, aspect in zip(angles, aspects)]
    ref_h = _reference_height(observations, aspects, config)

    upper = []
    for observation in observations:
        shoulders = observation.midpoint("left_shoulder", "right_shoulder", config.keypoint_conf)
        upper.append(shoulders[1] if shoulders else observation.bbox[1] + 0.18 * observation.height)

    total, limbs, travel = _activity(observations, ref_h, config)
    overlap = np.array([o.support_overlap for o in observations])
    return TrackSeries(
        track_id=track_id,
        frames=np.array([o.frame_index for o in observations]),
        times=np.array([o.time_s for o in observations]),
        posture=_majority(raw, config.smooth_frames),
        angle=np.array([math.nan if a is None else a for a in angles]),
        aspect=aspects,
        ref_h=ref_h,
        upper_y=np.array(upper),
        on_support=overlap >= config.support_overlap,
        support_overlap=overlap,
        activity=total,
        limb_activity=limbs,
        travel=travel,
        confidence=np.array([o.confidence for o in observations]),
    )


def _descent(series: TrackSeries, onset: int, config: MovementConfig) -> tuple[float, float, int, int]:
    """
    Mede a descida em torno do instante em que a pessoa passa a deitar.

    Devolve a descida (alturas de corpo), a velocidade de pico e os índices do
    ponto mais alto e do ponto mais baixo. A janela continua depois do início
    da postura deitada porque o tronco fica horizontal antes de o corpo chegar
    ao chão.
    """
    times, upper = series.times, series.upper_y
    before = np.where((times >= times[onset] - config.fall_window_seconds) & (times <= times[onset] + 0.2))[0]
    if len(before) < 2:
        return 0.0, 0.0, onset, onset
    # Início da descida: o último instante em que o corpo ainda estava no ponto
    # mais alto (com tolerância para o tremor da estimativa de pose).
    highest = float(upper[before].min())
    tolerance = 0.03 * float(series.ref_h[onset])
    top = int(before[upper[before] <= highest + tolerance][-1])
    span = np.where((times >= times[top]) & (times <= times[onset] + config.fall_settle_seconds))[0]
    bottom = int(span[int(np.argmax(upper[span]))])
    span = span[span <= bottom]
    reference = float(series.ref_h[top])
    drop = float(upper[bottom] - upper[top]) / reference

    peak = 0.0
    lag = config.speed_lag_seconds
    for index in span:
        earlier = span[(times[span] <= times[index] - lag * 0.6) & (times[span] >= times[index] - lag * 1.5)]
        if len(earlier) == 0:
            continue
        start = earlier[0]
        peak = max(peak, float(upper[index] - upper[start]) / reference / float(times[index] - times[start]))
    return drop, peak, top, bottom


def _detect_falls(series: TrackSeries, config: MovementConfig) -> list[VideoEvent]:
    events = []
    lying = np.array([label == LYING for label in series.posture])
    for start, end in runs(lying, series.times, bridge=0.2):
        duration = series.times[end] - series.times[start]
        ends_with_track = end >= len(lying) - 2
        if duration < config.lying_confirm_seconds and not (ends_with_track and duration >= 0.15):
            continue
        if start == 0:
            continue  # já aparece deitado: não há descida para avaliar
        supported = float(np.mean(series.on_support[start : end + 1])) >= 0.5
        drop, speed, top, bottom = _descent(series, start, config)
        descent_seconds = float(series.times[bottom] - series.times[top])
        if supported or drop < config.fall_min_drop or speed < config.fall_min_speed:
            continue
        margin = min(drop / config.fall_min_drop, speed / config.fall_min_speed, 2.0)
        confidence = float(np.clip(0.5 + 0.25 * (margin - 1.0), 0.5, 0.95))
        confidence *= float(np.clip(np.mean(series.confidence[top : end + 1]) + 0.25, 0.6, 1.0))
        events.append(
            VideoEvent(
                kind="fall_detected",
                track_id=series.track_id,
                start_s=float(series.times[top]),
                end_s=float(series.times[end]),
                frame_index=int(series.frames[start]),
                confidence=confidence,
                priority="high",
                anomaly=True,
                detail=(
                    f"Descida rápida seguida de postura deitada fora de cama ou sofá: parte superior do "
                    f"corpo desceu {drop:.2f} altura de corpo em {descent_seconds:.1f} s "
                    f"(pico de {speed:.1f} alturas/s)."
                ),
                metrics={
                    "drop_body_heights": round(float(drop), 3),
                    "peak_speed_body_heights_s": round(float(speed), 3),
                    "descent_seconds": round(descent_seconds, 3),
                    "lying_seconds": round(float(duration), 3),
                    "lying_from_s": round(float(series.times[start]), 3),
                    "lying_from_index": int(start),
                },
            )
        )
    return events


def _detect_bed_exits(series: TrackSeries, config: MovementConfig) -> list[VideoEvent]:
    events = []
    in_bed = np.array([label == LYING for label in series.posture]) & series.on_support
    for start, end in runs(in_bed, series.times, bridge=0.3):
        if series.times[end] - series.times[start] < config.bed_min_seconds:
            continue
        after = np.where(
            (series.times > series.times[end])
            & (series.times <= series.times[end] + config.bed_exit_window_seconds)
        )[0]
        if len(after) == 0:
            continue
        out = np.array(
            [
                series.posture[i] == UPRIGHT
                or (series.posture[i] != LYING and series.support_overlap[i] < config.bed_exit_overlap)
                for i in after
            ]
        )
        for run_start, run_end in runs(out, series.times[after], bridge=0.2):
            first, last = after[run_start], after[run_end]
            if series.times[last] - series.times[first] < config.bed_exit_confirm_seconds:
                continue
            standing = series.posture[first] == UPRIGHT or series.posture[last] == UPRIGHT
            events.append(
                VideoEvent(
                    kind="bed_exit_attempt",
                    track_id=series.track_id,
                    start_s=float(series.times[end]),
                    end_s=float(series.times[last]),
                    frame_index=int(series.frames[first]),
                    confidence=float(np.clip(np.mean(series.confidence[first : last + 1]), 0.4, 0.9)),
                    priority="high" if standing else "medium",
                    anomaly=True,
                    detail=(
                        f"Paciente estava deitado no leito por {series.times[end] - series.times[start]:.0f} s e "
                        + ("levantou-se, ficando em pé." if standing else "deixou a área do leito.")
                    ),
                    metrics={
                        "in_bed_seconds": round(float(series.times[end] - series.times[start]), 3),
                        "standing": bool(standing),
                    },
                )
            )
            break
    return events


def analyze_track(track_id: int, observations: list[Observation], config: MovementConfig) -> tuple[TrackSeries, list[VideoEvent]]:
    """Aplica todas as regras a uma pessoa."""
    series = build_series(track_id, observations, config)
    events = _detect_falls(series, config)
    events += _detect_bed_exits(series, config)

    lying = np.array([label == LYING for label in series.posture])
    # Imobilidade: usa o deslocamento do corpo (centro da caixa). Em pessoa
    # deitada as articulações estimadas oscilam mesmo com o corpo parado.
    patterns = detect_activity_patterns(
        series.times,
        series.travel,
        lying,
        on_support=series.on_support,
        travel=series.travel,
        config=config.activity,
    )
    # Agitação: usa só o movimento dos membros, para não confundir com caminhada.
    patterns = [p for p in patterns if p.kind == "prolonged_immobility"] + [
        p
        for p in detect_activity_patterns(
            series.times, series.limb_activity, lying, series.on_support, series.travel, config.activity
        )
        if p.kind == "agitation"
    ]

    falls = [event for event in events if event.kind == "fall_detected"]
    for pattern in patterns:
        nearest = int(np.argmin(np.abs(series.times - pattern.start_s)))
        if pattern.kind == "prolonged_immobility":
            # Imobilidade logo depois de uma queda agrava a queda, em vez de virar outro evento.
            fall = next(
                (f for f in falls if f.metrics["lying_from_s"] - 0.5 <= pattern.start_s <= f.end_s + 2.0),
                None,
            )
            if fall is not None:
                fall.metrics["immobile_seconds"] = round(pattern.duration_s, 3)
                fall.metrics["immobile_until_end"] = pattern.extra.get("until_end", False)
                fall.priority = "critical"
                fall.end_s = max(fall.end_s, pattern.end_s)
                fall.detail += (
                    f" Depois da queda permaneceu praticamente imóvel por {pattern.duration_s:.1f} s"
                    + (" até o fim do registro." if pattern.extra.get("until_end") else ".")
                )
                continue
        events.append(
            VideoEvent(
                kind=pattern.kind,
                track_id=track_id,
                start_s=pattern.start_s,
                end_s=pattern.end_s,
                frame_index=int(series.frames[nearest]),
                confidence=float(np.clip(np.mean(series.confidence), 0.4, 0.9)),
                priority=pattern.priority,
                anomaly=True,
                detail=pattern.detail,
                metrics={
                    "mean_activity": round(float(pattern.mean_activity), 3),
                    "location": pattern.location,
                    **pattern.extra,
                },
            )
        )
    return series, sorted(events, key=lambda e: e.start_s)


@dataclass
class MovementAnalysis:
    events: list[VideoEvent]
    series: dict[int, TrackSeries]

    @property
    def anomalies(self) -> list[VideoEvent]:
        return [event for event in self.events if event.anomaly]


def analyze_observations(observations: list[Observation], config: MovementConfig | None = None) -> MovementAnalysis:
    """
    Analisa todas as pessoas de um vídeo.

    Sem achado anômalo, devolve um único `video_check` resumindo o que foi
    observado, para que a integração saiba que o vídeo foi avaliado.
    """
    config = config or MovementConfig()
    events: list[VideoEvent] = []
    all_series: dict[int, TrackSeries] = {}
    for track_id, track in group_tracks(observations).items():
        if len(track) < 3 or track[-1].time_s - track[0].time_s < config.min_track_seconds:
            continue
        series, found = analyze_track(track_id, track, config)
        all_series[track_id] = series
        events.extend(found)

    if not events:
        if all_series:
            main = max(all_series.values(), key=lambda s: len(s.times))
            share = main.posture_share()
            seen = ", ".join(
                f"{POSTURE_LABELS[name]} {share[name] * 100:.0f}%" for name in (UPRIGHT, LOW, LYING) if share[name] > 0
            )
            detail = f"Nenhum movimento fora do padrão. Posturas observadas: {seen}."
            start, end = float(main.times[0]), float(main.times[-1])
            frame, track_id = int(main.frames[len(main.frames) // 2]), main.track_id
            confidence = float(np.mean(main.confidence))
        else:
            detail = "Nenhuma pessoa identificada no vídeo."
            start = end = 0.0
            frame, track_id, confidence = 0, -1, 0.5
        events.append(
            VideoEvent(
                kind="video_check",
                track_id=track_id,
                start_s=start,
                end_s=end,
                frame_index=frame,
                confidence=float(np.clip(confidence, 0.3, 0.95)),
                priority="low",
                anomaly=False,
                detail=detail,
            )
        )
    return MovementAnalysis(events=sorted(events, key=lambda e: e.start_s), series=all_series)


def keypoint_index(name: str) -> int:
    """Posição de uma articulação no vetor de 17 pontos (útil para montar poses de teste)."""
    return INDEX[name]
