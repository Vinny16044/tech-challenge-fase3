"""
Análise de vídeo com poses sintéticas: não usa vídeo nem os modelos YOLOv8.

Cada cena é uma sequência de "bonecos" de 17 articulações gerada por
`figure`, girando o corpo em torno dos pés para simular quedas e deitar.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

np = pytest.importorskip("numpy")

from src.anomaly_detection.movement_patterns import ActivityConfig, detect_activity_patterns, runs  # noqa: E402
from src.contracts import MonitoringEventV1  # noqa: E402
from src.video_analysis.extraction import Extraction  # noqa: E402
from src.video_analysis.movement_detection import LOW, LYING, UPRIGHT, analyze_observations, build_series, MovementConfig  # noqa: E402
from src.video_analysis.objects import DetectedObject, assign_supports, overlap_fraction, stable_objects  # noqa: E402
from src.video_analysis.pose import KEYPOINTS, Observation, Tracker, iou, trunk_angle  # noqa: E402
from src.video_analysis.preprocessing import ALL_CLIPS, SAMPLE_CLIPS, VideoInfo, clip_split, expected_label  # noqa: E402
from src.video_analysis.report import evaluate, render_markdown, to_events, video_report  # noqa: E402

FPS = 30
HEIGHT = 160.0  # altura do boneco em pé, em pixels
FLOOR = 220.0

# Posição de cada articulação ao longo do corpo (0 = pés, 1 = topo) e para o lado.
_ALONG = {
    "nose": 0.95, "left_eye": 0.97, "right_eye": 0.97, "left_ear": 0.96, "right_ear": 0.96,
    "left_shoulder": 0.8, "right_shoulder": 0.8, "left_elbow": 0.65, "right_elbow": 0.65,
    "left_wrist": 0.5, "right_wrist": 0.5, "left_hip": 0.5, "right_hip": 0.5,
    "left_knee": 0.25, "right_knee": 0.25, "left_ankle": 0.0, "right_ankle": 0.0,
}
_LIMBS = (
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def figure(index, angle=0.0, *, feet_x=160.0, scale=1.0, wide=False, support=0.0, limb_swing=0.0, track=1):
    """Boneco inclinado `angle` graus em relação à vertical; `scale` encolhe o corpo (agachar)."""
    theta = math.radians(angle)
    along = (math.sin(theta), -math.cos(theta))
    across = (math.cos(theta), math.sin(theta))
    points = []
    for name in KEYPOINTS:
        side = (-1 if name.startswith("left") else 1) * 0.1 * HEIGHT
        reach = _ALONG[name] * HEIGHT * scale
        x = feet_x + along[0] * reach + across[0] * side
        y = FLOOR + along[1] * reach + across[1] * side
        if limb_swing and name in _LIMBS:
            y += limb_swing * HEIGHT * (1 if "left" in name else -1)
        points.append([x, y, 0.9])
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    pad = 0.05 * HEIGHT
    x1, x2 = min(xs) - pad, max(xs) + pad
    if wide:  # agachado: a caixa fica quase tão larga quanto alta
        x1, x2 = x1 - 0.12 * HEIGHT, x2 + 0.12 * HEIGHT
    return Observation(
        frame_index=index,
        time_s=index / FPS,
        bbox=(x1, min(ys) - pad, x2, max(ys) + pad),
        confidence=0.9,
        keypoints=points,
        track_id=track,
        support_label="bed" if support else None,
        support_overlap=support,
    )


def scene(*parts):
    """Monta uma cena a partir de trechos (duração em s, função(progresso 0..1) -> kwargs de `figure`)."""
    observations, index = [], 0
    for seconds, make in parts:
        frames = int(seconds * FPS)
        for step in range(frames):
            observations.append(figure(index, **make(step / max(frames - 1, 1))))
            index += 1
    return observations


standing = lambda _: {"angle": 0.0}  # noqa: E731
lying_floor = lambda _: {"angle": 90.0}  # noqa: E731
lying_bed = lambda _: {"angle": 90.0, "support": 1.0}  # noqa: E731


def kinds(analysis):
    return [event.kind for event in analysis.events]


# --------------------------------------------------------------------------
# Postura
# --------------------------------------------------------------------------


def test_trunk_angle_of_standing_and_lying_figures():
    assert trunk_angle(figure(0, 0)) == pytest.approx(0, abs=1)
    assert trunk_angle(figure(0, 90)) == pytest.approx(90, abs=1)
    unknown = figure(0, 0)
    for point in unknown.keypoints:
        point[2] = 0.1
    assert trunk_angle(unknown) is None


def test_posture_classification():
    observations = scene((1, standing), (1, lambda _: {"scale": 0.55, "wide": True}), (1, lying_floor))
    series = build_series(1, observations, MovementConfig())
    assert series.posture[10] == UPRIGHT
    assert series.posture[45] == LOW
    assert series.posture[80] == LYING
    share = series.posture_share()
    assert share[UPRIGHT] == pytest.approx(1 / 3, abs=0.05)
    # A referência de altura é a da pessoa em pé, mesmo depois de deitar.
    assert series.ref_h[80] == pytest.approx(HEIGHT * 1.07, rel=0.05)


def test_bending_over_is_not_lying():
    """Tronco horizontal com as pernas na vertical (curvar-se): caixa continua alta."""
    bent = figure(0, 0)
    for name in ("left_shoulder", "right_shoulder", "nose"):
        point = bent.keypoints[KEYPOINTS.index(name)]
        point[0], point[1] = point[0] + 0.3 * HEIGHT, FLOOR - 0.5 * HEIGHT
    assert trunk_angle(bent) > 55
    series = build_series(1, [bent] * 5, MovementConfig())
    assert set(series.posture) == {LOW}


# --------------------------------------------------------------------------
# Eventos
# --------------------------------------------------------------------------


def test_fall_followed_by_immobility_is_critical():
    observations = scene((2, standing), (0.6, lambda p: {"angle": 90 * p}), (3, lying_floor))
    analysis = analyze_observations(observations)

    assert kinds(analysis) == ["fall_detected"]
    fall = analysis.events[0]
    assert fall.anomaly and fall.priority == "critical"
    assert 1.9 <= fall.start_s <= 2.3
    assert fall.metrics["drop_body_heights"] > 0.6
    assert fall.metrics["peak_speed_body_heights_s"] > 0.7
    assert fall.metrics["immobile_seconds"] >= 2.0
    assert "imóvel" in fall.detail


def test_fall_with_movement_afterwards_is_high():
    """Depois da queda a pessoa se arrasta pelo chão: sem imobilidade."""
    crawl = lambda p: {"angle": 90.0, "feet_x": 160 + 120 * p}  # noqa: E731
    analysis = analyze_observations(scene((2, standing), (0.6, lambda p: {"angle": 90 * p}), (3, crawl)))
    assert kinds(analysis) == ["fall_detected"]
    assert analysis.events[0].priority == "high"
    assert "immobile_seconds" not in analysis.events[0].metrics


def test_lying_down_on_bed_is_not_a_fall():
    observations = scene((2, standing), (3, lambda p: {"angle": 90 * p, "support": p}), (3, lying_bed))
    analysis = analyze_observations(observations)
    assert kinds(analysis) == ["video_check"]
    assert not analysis.events[0].anomaly
    assert "deitado" in analysis.events[0].detail


def test_fast_drop_onto_bed_is_not_a_fall():
    observations = scene((2, standing), (0.6, lambda p: {"angle": 90 * p, "support": 1.0}), (1.5, lying_bed))
    assert "fall_detected" not in kinds(analyze_observations(observations))


def test_slow_lying_on_floor_reports_immobility_not_fall():
    observations = scene((2, standing), (4, lambda p: {"angle": 90 * p}), (3, lying_floor))
    analysis = analyze_observations(observations)
    assert kinds(analysis) == ["prolonged_immobility"]
    event = analysis.events[0]
    assert event.priority == "high" and event.metrics["location"] == "floor"


def test_crouching_is_not_an_event():
    crouch = lambda p: {"scale": 1 - 0.45 * p, "wide": p > 0.5}  # noqa: E731
    analysis = analyze_observations(scene((2, standing), (1.2, crouch), (2, lambda _: {"scale": 0.55, "wide": True})))
    assert kinds(analysis) == ["video_check"]


def test_bed_exit_is_detected():
    rise = lambda p: {"angle": 90 * (1 - p), "support": 1.0 - 0.9 * p}  # noqa: E731
    analysis = analyze_observations(scene((2.5, lying_bed), (1.0, rise), (2, lambda _: {"angle": 0.0, "support": 0.1})))
    assert "bed_exit_attempt" in kinds(analysis)
    event = next(e for e in analysis.events if e.kind == "bed_exit_attempt")
    assert event.priority == "high" and event.metrics["standing"] is True


def test_agitation_in_bed_is_detected():
    calm = lambda _: {"angle": 90.0, "support": 1.0}  # noqa: E731

    def restless(progress):
        return {"angle": 90.0, "support": 1.0, "limb_swing": 0.45 * math.sin(2 * math.pi * 12 * progress)}

    analysis = analyze_observations(scene((5, calm), (6, restless)))
    assert "agitation" in kinds(analysis)
    event = next(e for e in analysis.events if e.kind == "agitation")
    assert event.priority == "medium" and event.start_s >= 4.5


def test_short_noise_track_is_ignored_and_empty_video_is_checked():
    assert kinds(analyze_observations([figure(0), figure(1)])) == ["video_check"]
    empty = analyze_observations([])
    assert kinds(empty) == ["video_check"] and "Nenhuma pessoa" in empty.events[0].detail


def test_each_person_is_analysed_separately():
    faller = scene((2, standing), (0.6, lambda p: {"angle": 90 * p}), (2.5, lying_floor))
    bystander = [figure(o.frame_index, feet_x=40.0, track=2) for o in faller]
    analysis = analyze_observations(faller + bystander)
    assert [(e.kind, e.track_id) for e in analysis.events] == [("fall_detected", 1)]
    assert set(analysis.series) == {1, 2}


# --------------------------------------------------------------------------
# Padrões de atividade (função usada pelo vídeo)
# --------------------------------------------------------------------------


def test_runs_merges_short_gaps():
    times = np.arange(10) / 10
    mask = [True, True, False, True, True, False, False, False, True, True]
    assert runs(mask, times) == [(0, 1), (3, 4), (8, 9)]
    assert runs(mask, times, bridge=0.25) == [(0, 4), (8, 9)]


def test_immobility_threshold_depends_on_location():
    times = np.arange(0, 60, 0.5)
    still = np.zeros(len(times))
    lying = np.ones(len(times), bool)
    on_floor = detect_activity_patterns(times, still, lying)
    in_bed = detect_activity_patterns(times, still, lying, on_support=np.ones(len(times), bool))
    long_bed = detect_activity_patterns(
        times, still, lying, on_support=np.ones(len(times), bool), config=ActivityConfig(bed_immobility_seconds=30)
    )
    assert [f.kind for f in on_floor] == ["prolonged_immobility"] and on_floor[0].location == "floor"
    assert in_bed == []  # 1 min parado no leito é esperado
    assert long_bed[0].location == "support" and long_bed[0].priority == "medium"


def test_walking_is_not_agitation():
    times = np.arange(0, 20, 0.1)
    activity = np.full(len(times), 0.8)
    lying = np.zeros(len(times), bool)
    walking = detect_activity_patterns(times, activity, lying, travel=np.full(len(times), 0.9))
    in_place = detect_activity_patterns(times, activity, lying, travel=np.zeros(len(times)))
    assert walking == []
    assert [f.kind for f in in_place] == ["agitation"]


# --------------------------------------------------------------------------
# Rastreamento e objetos
# --------------------------------------------------------------------------


def test_tracker_keeps_identities_through_a_fall():
    tracker = Tracker()
    faller = scene((1, standing), (0.6, lambda p: {"angle": 90 * p}), (1, lying_floor))
    for observation in faller:
        other = figure(observation.frame_index, feet_x=40.0)
        tracker.update([Observation(**{**observation.__dict__, "track_id": -1}), other])
    ids = {track.track_id: len(track.observations) for track in tracker._tracks}
    assert ids == {1: len(faller), 2: len(faller)}


def test_tracker_reuses_track_after_short_gap():
    tracker = Tracker(max_missed=10)
    first = tracker.update([figure(0)])[0].track_id
    for _ in range(5):
        tracker.update([])
    assert tracker.update([figure(6)])[0].track_id == first


def test_stable_objects_and_support_overlap():
    bed = DetectedObject("bed", 0.8, (100, 100, 300, 200))
    frames = [[bed], [DetectedObject("bed", 0.7, (104, 98, 302, 204))], [bed], [DetectedObject("couch", 0.3, (0, 0, 50, 50))], []]
    stable = stable_objects(frames)
    assert [item.label for item in stable] == ["bed"]  # o sofá apareceu em 1 de 5 quadros
    assert stable[0].frames == 3

    assert overlap_fraction((150, 120, 250, 180), bed.bbox) == 1.0
    assert overlap_fraction((250, 150, 350, 250), bed.bbox) == pytest.approx(0.25)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0

    inside = figure(0, 90)
    inside.bbox = (150, 120, 250, 180)
    outside = figure(1, 0)
    outside.bbox = (400, 0, 440, 100)
    assign_supports([inside, outside], stable + [DetectedObject("chair", 0.9, (390, 0, 450, 110))])
    assert inside.support_label == "bed" and inside.support_overlap == 1.0
    assert outside.support_label is None  # cadeira não é apoio para deitar


# --------------------------------------------------------------------------
# Relatório e contrato
# --------------------------------------------------------------------------


def extraction_for(name, observations):
    video = VideoInfo(name=name, fps=FPS, frame_count=len(observations), width=320, height=240, duration_s=len(observations) / FPS, crop="right")
    return Extraction(video=video, observations=observations, objects=[DetectedObject("bed", 0.8, (200, 150, 320, 240), 5)], processed_frames=[o.frame_index for o in observations])


def test_events_follow_contract():
    observations = scene((2, standing), (0.6, lambda p: {"angle": 90 * p}), (3, lying_floor))
    extraction = extraction_for("fall-99-cam0.mp4", observations)
    analysis = analyze_observations(observations)
    recorded_at = datetime(2026, 11, 12, 10, 5, tzinfo=timezone(timedelta(hours=-3)))
    events = to_events(extraction, analysis, patient_id="P0001", recorded_at=recorded_at, keyframes={0: "evento_1.jpg"})

    assert len(events) == 1
    event = events[0]
    validated = MonitoringEventV1.model_validate(event)
    assert validated.module == "video" and validated.event_type == "fall_detected"
    assert validated.priority == "critical"
    assert event["event_id"].startswith("video-P0001-fall-99-cam0-p1-fall_detected-")
    assert event["timestamp"].startswith("2026-11-12T10:05:02")
    clip, frame = event["evidence"]
    assert clip["type"] == "video_clip" and clip["uri"] == "video://urfd/fall-99-cam0.mp4"
    assert clip["end_s"] > clip["start_s"] and 0 < clip["confidence"] <= 1
    assert frame["type"] == "video_frame" and "evento_1.jpg" in frame["description"]
    assert events == to_events(extraction, analysis, patient_id="P0001", recorded_at=recorded_at, keyframes={0: "evento_1.jpg"})


def test_video_without_deviation_publishes_video_check():
    observations = scene((3, standing))
    events = to_events(
        extraction_for("adl-99-cam0.mp4", observations),
        analyze_observations(observations),
        patient_id="P0001",
        recorded_at=datetime(2026, 11, 12, 10, 5, tzinfo=timezone.utc),
    )
    assert [(e["event_type"], e["anomaly_detected"], e["priority"]) for e in events] == [("video_check", False, "low")]
    MonitoringEventV1.model_validate(events[0])


def test_report_and_dataset_evaluation():
    fall = scene((2, standing), (0.6, lambda p: {"angle": 90 * p}), (3, lying_floor))
    walk = scene((3, standing))
    missed = scene((2, standing), (4, lambda p: {"angle": 90 * p}), (0.5, lying_floor))  # "queda" lenta demais
    reports = [
        video_report(extraction_for(name, obs), analyze_observations(obs))
        for name, obs in (("fall-01-cam0.mp4", fall), ("adl-01-cam0.mp4", walk), ("fall-02-cam0.mp4", missed), ("outro.mp4", walk))
    ]
    assert expected_label("outro.mp4") is None and reports[3]["predicted_label"] is None

    result = evaluate(reports)
    assert result["videos"] == 3
    assert (result["true_positive"], result["false_negative"], result["false_positive"], result["true_negative"]) == (1, 1, 0, 1)
    assert result["sensitivity"] == 0.5 and result["specificity"] == 1.0
    assert result["errors"] == ["fall-02-cam0.mp4"]

    text = render_markdown(reports, result)
    assert "# Relatório automático" in text and "| `fall-01-cam0.mp4` |" in text
    assert "**Desvio** — Queda" in text and "Sensibilidade 0.5" in text


def test_calibration_and_validation_clips_do_not_overlap():
    assert len(ALL_CLIPS) == 70 and set(SAMPLE_CLIPS) <= set(ALL_CLIPS)
    splits = [clip_split(name) for name in ALL_CLIPS]
    assert splits.count("ajuste") == 8 and splits.count("validacao") == 62
    assert clip_split("data/video/raw/fall-01-cam0.mp4") == "ajuste"
    assert clip_split("fall-02-cam0.mp4") == "validacao"
    assert clip_split("outro.mp4") is None


def test_evaluation_is_reported_per_split():
    fall = scene((2, standing), (0.6, lambda p: {"angle": 90 * p}), (3, lying_floor))
    walk = scene((3, standing))
    slow = scene((2, standing), (4, lambda p: {"angle": 90 * p}), (0.5, lying_floor))
    clips = (("fall-01-cam0.mp4", fall), ("adl-01-cam0.mp4", walk), ("fall-02-cam0.mp4", slow), ("adl-02-cam0.mp4", fall))
    reports = [video_report(extraction_for(name, obs), analyze_observations(obs)) for name, obs in clips]
    assert [r["split"] for r in reports] == ["ajuste", "ajuste", "validacao", "validacao"]

    result = evaluate(reports)
    assert result["by_split"]["ajuste"]["accuracy"] == 1.0
    validation = result["by_split"]["validacao"]
    assert (validation["false_negative"], validation["false_positive"]) == (1, 1)
    assert validation["sensitivity"] == 0.0 and validation["specificity"] == 0.0

    text = render_markdown(reports, result)
    assert "só a validação é uma medida independente" in text
    assert "| Validação (clipes fora do ajuste) | 2 | 0 de 1 | 1 de 1 |" in text


def test_extraction_round_trip(tmp_path):
    observations = scene((1, standing))
    original = extraction_for("adl-99-cam0.mp4", observations)
    loaded = Extraction.load(original.save(tmp_path / "cache.json"))
    assert loaded.video == original.video
    assert len(loaded.observations) == len(observations)
    assert loaded.observations[5].bbox == pytest.approx(observations[5].bbox, abs=0.1)
    assert loaded.objects[0].label == "bed"
