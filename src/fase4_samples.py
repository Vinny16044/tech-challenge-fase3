"""
Gera as saídas do Vini (vídeo e dados clínicos) para a integração da Fase 4.

    py -m src.fase4_samples

Pré-requisitos: dados clínicos baixados (`py -m src.anomaly_detection.data_loader
--download`) e vídeos já processados (`py -m src.video_analysis.run --crop right`).

Escreve:

- `data/fase4_recebidos/P0001_video.json` e `P0001_clinical.json` — amostras do
  teste conjunto. O paciente `P0001` é COMPOSTO: o vídeo é um clipe de queda do
  UR Fall Detection Dataset e os dados clínicos são de um paciente do MIMIC-IV
  Demo, com os horários deslocados para a janela combinada pela equipe
  (12/11/2026, 10:00–10:15, -03:00). Serve para demonstrar a fusão das
  modalidades com saídas reais dos módulos, não um caso clínico único.
- `data/fase4_vini/` — exemplos completos, com os horários de origem: os eventos
  de vídeo de todos os clipes processados, a comparação com o rótulo do dataset
  (ajuste x validação) e a internação inteira do paciente clínico escolhido.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.anomaly_detection.data_loader import RAW_DIR, ROOT
from src.anomaly_detection.events import EventContext, save_json
from src.anomaly_detection.run import analyze, build_events
from src.video_analysis.extraction import Extraction
from src.video_analysis.movement_detection import analyze_observations
from src.video_analysis.report import evaluate, to_events, video_report

RECEIVED_DIR = ROOT / "data" / "fase4_recebidos"
EXAMPLES_DIR = ROOT / "data" / "fase4_vini"
VIDEO_CACHE = ROOT / "output" / "video" / "cache"

BRT = timezone(timedelta(hours=-3))
WINDOW_START = datetime(2026, 11, 12, 10, 0, 0, tzinfo=BRT)
WINDOW_END = WINDOW_START + timedelta(minutes=15)

# Vídeo do teste conjunto e início fictício da gravação.
VIDEO_CLIP = "fall-01-cam0"
VIDEO_RECORDED_AT = datetime(2026, 11, 12, 10, 5, 0, tzinfo=BRT)

# Paciente do MIMIC-IV Demo e o instante de origem que passa a valer 10:08 (-03:00):
# sinais vitais críticos no mesmo horário da associação de um segundo vasoativo.
CLINICAL_SUBJECT = 10002495
CLINICAL_ANCHOR = datetime(2141, 5, 23, 20, 0, 0, tzinfo=timezone.utc)
CLINICAL_ANCHOR_AT = datetime(2026, 11, 12, 10, 8, 0, tzinfo=BRT)


def _load_extraction(clip: str, cache_dir: Path) -> Extraction:
    path = cache_dir / f"{clip}.json"
    if not path.exists():
        raise SystemExit(f"{path} não encontrado. Rode antes: py -m src.video_analysis.run --crop right")
    return Extraction.load(path)


def video_sample(cache_dir: Path = VIDEO_CACHE) -> list[dict]:
    """Eventos do clipe de queda, como paciente P0001 na janela do teste conjunto."""
    extraction = _load_extraction(VIDEO_CLIP, cache_dir)
    analysis = analyze_observations(extraction.observations)
    return to_events(extraction, analysis, patient_id="P0001", recorded_at=VIDEO_RECORDED_AT)


def video_examples(cache_dir: Path = VIDEO_CACHE) -> tuple[list[dict], dict]:
    """
    Eventos de todos os clipes processados e a comparação com o rótulo do dataset.

    Cada clipe vira um paciente fictício. A comparação separa os clipes de
    ajuste (usados para calibrar os limiares) dos de validação.
    """
    events, reports = [], []
    for position, path in enumerate(sorted(cache_dir.glob("*.json"))):
        extraction = Extraction.load(path)
        analysis = analyze_observations(extraction.observations)
        reports.append(video_report(extraction, analysis))
        events += to_events(
            extraction,
            analysis,
            patient_id=f"P{9001 + position}",
            recorded_at=VIDEO_RECORDED_AT + timedelta(minutes=20 * position),
        )
    summary = {
        "evaluation": evaluate(reports),
        "videos": [
            {
                "name": report["video"]["name"],
                "split": report["split"],
                "dataset_label": report["dataset_label"],
                "predicted_label": report["predicted_label"],
                "events": [{"kind": e["kind"], "priority": e["priority"]} for e in report["events"]],
            }
            for report in reports
        ],
    }
    return events, summary


def clinical_sample(result: dict) -> list[dict]:
    """Eventos clínicos do paciente escolhido que caem na janela do teste conjunto."""
    context = EventContext(
        patient_alias={str(CLINICAL_SUBJECT): "P0001"},
        time_shift=CLINICAL_ANCHOR_AT - CLINICAL_ANCHOR,
        tz=BRT,
    )
    events = build_events(result, context, {CLINICAL_SUBJECT})
    return [e for e in events if WINDOW_START <= datetime.fromisoformat(e["timestamp"]) < WINDOW_END]


def main() -> None:
    parser = argparse.ArgumentParser(description="Amostras de vídeo e dados clínicos para a integração")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--video-cache", type=Path, default=VIDEO_CACHE)
    parser.add_argument("--received-dir", type=Path, default=RECEIVED_DIR)
    parser.add_argument("--examples-dir", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args()

    result = analyze(args.raw_dir)
    video = video_sample(args.video_cache)
    clinical = clinical_sample(result)
    if not video or not clinical:
        raise SystemExit("amostra vazia: confira o clipe e o paciente escolhidos")
    save_json(video, args.received_dir / "P0001_video.json")
    save_json(clinical, args.received_dir / "P0001_clinical.json")
    print(f"Teste conjunto: {len(video)} evento(s) de vídeo e {len(clinical)} clínico(s) em {args.received_dir}")

    examples_video, video_summary = video_examples(args.video_cache)
    examples_clinical = build_events(result, EventContext(), {CLINICAL_SUBJECT})
    save_json(examples_video, args.examples_dir / "video_events_urfd.json")
    save_json(video_summary, args.examples_dir / "video_evaluation_urfd.json")
    save_json(examples_clinical, args.examples_dir / f"clinical_events_P{CLINICAL_SUBJECT}.json")
    print(
        f"Exemplos: {len(examples_video)} evento(s) de vídeo e {len(examples_clinical)} clínico(s) "
        f"em {args.examples_dir}"
    )


if __name__ == "__main__":
    main()
