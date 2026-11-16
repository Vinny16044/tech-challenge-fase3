"""
Análise de vídeo de ponta a ponta.

    py -m src.video_analysis.run --input data/video/raw --crop right --annotate

Para cada vídeo: estima as poses (YOLOv8-pose), localiza os objetos da cena
(YOLOv8), aplica as regras de movimento, grava quadros anotados dos achados e
publica os eventos no contrato da Fase 4. No fim gera o relatório automático
(`report.md` e `report.json`) e `events.json`.

Use `--from-cache` para reaplicar as regras sem rodar os modelos de novo.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.anomaly_detection.events import save_json
from src.video_analysis.extraction import Extraction, extract, resolve_weights
from src.video_analysis.movement_detection import LOW, LYING, UPRIGHT, MovementAnalysis, MovementConfig, analyze_observations
from src.video_analysis.objects import OBJECT_MODEL, ObjectDetector
from src.video_analysis.pose import INDEX, POSE_MODEL, SKELETON, Observation, PoseEstimator
from src.video_analysis.preprocessing import RAW_DIR, ROOT, iter_frames
from src.video_analysis.report import KIND_LABELS, evaluate, render_markdown, to_events, video_report

OUTPUT_DIR = ROOT / "output" / "video"

# BGR
_POSTURE_COLORS = {UPRIGHT: (80, 200, 80), LOW: (0, 200, 255), LYING: (60, 60, 230)}
_OBJECT_COLOR = (230, 150, 60)
_ALERT_COLOR = (40, 40, 220)
# Textos sem acento: as fontes do OpenCV não desenham caracteres acentuados.
_POSTURE_TEXT = {UPRIGHT: "em pe", LOW: "abaixado", LYING: "deitado"}
_KIND_TEXT = {
    "fall_detected": "QUEDA DETECTADA",
    "bed_exit_attempt": "SAIDA DO LEITO",
    "prolonged_immobility": "IMOBILIDADE",
    "agitation": "AGITACAO",
}


def _draw(frame, scale: float, people: list[tuple[Observation, str]], extraction: Extraction, banners: list[str]):
    """Desenha objetos, esqueletos e avisos num quadro (já ampliado por `scale`)."""
    import cv2

    for item in extraction.objects:
        x1, y1, x2, y2 = (int(v * scale) for v in item.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _OBJECT_COLOR, 1)
        cv2.putText(frame, item.label, (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _OBJECT_COLOR, 1, cv2.LINE_AA)

    for observation, posture in people:
        color = _POSTURE_COLORS.get(posture, (200, 200, 200))
        x1, y1, x2, y2 = (int(v * scale) for v in observation.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"P{observation.track_id} {_POSTURE_TEXT.get(posture, '')}", (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
        points = observation.keypoints
        for first, second in SKELETON:
            a, b = points[INDEX[first]], points[INDEX[second]]
            if a[2] >= 0.35 and b[2] >= 0.35:
                cv2.line(frame, (int(a[0] * scale), int(a[1] * scale)), (int(b[0] * scale), int(b[1] * scale)), color, 2)
        for x, y, conf in points:
            if conf >= 0.35:
                cv2.circle(frame, (int(x * scale), int(y * scale)), 3, (255, 255, 255), -1)

    for row, text in enumerate(banners):
        top = 6 + row * 30
        cv2.rectangle(frame, (6, top), (14 + 13 * len(text), top + 26), _ALERT_COLOR, -1)
        cv2.putText(frame, text, (12, top + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def render_outputs(
    path: Path, extraction: Extraction, analysis: MovementAnalysis, out_dir: Path, *, annotate: bool
) -> dict[int, str]:
    """
    Grava um quadro anotado por achado e, com `annotate`, o vídeo anotado.

    Devolve {posição do evento: nome do arquivo do quadro}.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[tuple[Observation, str]]] = {}
    postures = {
        (series.track_id, int(frame)): posture
        for series in analysis.series.values()
        for frame, posture in zip(series.frames, series.posture)
    }
    for observation in extraction.observations:
        posture = postures.get((observation.track_id, observation.frame_index))
        if posture is not None:
            by_frame.setdefault(observation.frame_index, []).append((observation, posture))

    # Quadro mais expressivo de cada achado: o último do trecho, quando a postura final já é visível.
    wanted: dict[int, int] = {}
    for index, event in enumerate(analysis.events):
        frame = event.frame_index
        if event.kind == "fall_detected":
            series = analysis.series[event.track_id]
            inside = [int(f) for f, t in zip(series.frames, series.times) if event.start_s <= t <= event.end_s]
            frame = inside[min(len(inside) - 1, int(len(inside) * 0.75))] if inside else frame
        wanted[index] = frame

    keyframes: dict[int, str] = {}
    writer = None
    for frame_index, time_s, frame, scale in iter_frames(path, stride=extraction.stride, crop=extraction.video.crop):
        active = [e for e in analysis.events if e.anomaly and e.start_s <= time_s <= e.end_s + 1.0]
        banners = [f"{_KIND_TEXT[e.kind]} (P{e.track_id}, {e.priority})" for e in active]
        drawn = _draw(frame, scale, by_frame.get(frame_index, []), extraction, banners)
        cv2.putText(
            drawn, f"{time_s:5.2f}s", (drawn.shape[1] - 78, drawn.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        for index, target in wanted.items():
            if target == frame_index:
                name = f"evento_{index + 1}_{analysis.events[index].kind}.jpg"
                cv2.imwrite(str(out_dir / name), drawn)
                keyframes[index] = name
        if annotate:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                size = (drawn.shape[1], drawn.shape[0])
                writer = cv2.VideoWriter(str(out_dir / "anotado.mp4"), fourcc, extraction.video.fps / extraction.stride, size)
            writer.write(drawn)
    if writer is not None:
        writer.release()
    return keyframes


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de vídeo (YOLOv8-pose + YOLOv8) — Fase 4")
    parser.add_argument("--input", type=Path, default=RAW_DIR, help="vídeo ou pasta de vídeos .mp4")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--crop", choices=["left", "right"], default=None, help="URFD: use right (metade RGB)")
    parser.add_argument("--stride", type=int, default=1, help="processa 1 a cada N quadros")
    parser.add_argument("--annotate", action="store_true", help="grava o vídeo anotado de cada clipe")
    parser.add_argument("--from-cache", action="store_true", help="reutiliza as poses já extraídas")
    parser.add_argument("--patient-id", default="P0001")
    parser.add_argument("--recorded-at", default="2026-11-12T10:05:00-03:00", help="início da 1ª gravação, com fuso")
    parser.add_argument("--spacing-minutes", type=float, default=20.0, help="intervalo entre o início dos vídeos")
    parser.add_argument("--source-name", default="urfd", help="nome da origem usado na URI das evidências")
    args = parser.parse_args()

    recorded_at = datetime.fromisoformat(args.recorded_at)
    if recorded_at.utcoffset() is None:
        parser.error("--recorded-at precisa de fuso, como -03:00")
    videos = [args.input] if args.input.is_file() else sorted(args.input.glob("*.mp4"))
    if not videos:
        parser.error(f"nenhum vídeo .mp4 em {args.input}. Rode: py -m src.video_analysis.preprocessing --download")

    pose = detector = None
    config = MovementConfig()
    reports, all_events = [], []
    for position, path in enumerate(videos):
        cache = args.output / "cache" / f"{path.stem}.json"
        started = time.time()
        if args.from_cache and cache.exists():
            extraction = Extraction.load(cache)
        else:
            pose = pose or PoseEstimator(resolve_weights(POSE_MODEL))
            detector = detector or ObjectDetector(resolve_weights(OBJECT_MODEL))
            extraction = extract(path, pose=pose, detector=detector, stride=args.stride, crop=args.crop)
            extraction.save(cache)
        elapsed = time.time() - started

        analysis = analyze_observations(extraction.observations, config)
        out_dir = args.output / path.stem
        keyframes = render_outputs(path, extraction, analysis, out_dir, annotate=args.annotate)
        events = to_events(
            extraction,
            analysis,
            patient_id=args.patient_id,
            recorded_at=recorded_at + timedelta(minutes=args.spacing_minutes * position),
            source_name=args.source_name,
            keyframes=keyframes,
        )
        report = video_report(extraction, analysis)
        report["keyframes"] = sorted(keyframes.values())
        report["extraction_seconds"] = round(elapsed, 2)
        save_json({**report, "contract_events": events}, out_dir / "analysis.json")
        reports.append(report)
        all_events.extend(events)

        found = ", ".join(KIND_LABELS[e.kind] for e in analysis.events)
        print(f"{path.name}: {len(extraction.processed_frames)} quadros em {elapsed:.1f} s -> {found}")

    evaluation = evaluate(reports)
    save_json(all_events, args.output / "events.json")
    save_json({"videos": reports, "evaluation": evaluation}, args.output / "report.json")
    (args.output / "report.md").write_text(render_markdown(reports, evaluation), encoding="utf-8")
    print(f"\n{len(all_events)} evento(s) em {args.output / 'events.json'}; relatório em {args.output / 'report.md'}")
    if evaluation:
        print(
            f"URFD: {evaluation['videos']} vídeos, sensibilidade {evaluation['sensitivity']}, "
            f"especificidade {evaluation['specificity']}."
        )


if __name__ == "__main__":
    main()
