"""
Relatório automático da análise de vídeo e conversão para o contrato da Fase 4.

`video_report` descreve um vídeo (pessoas, objetos, posturas e desvios);
`render_markdown` junta os vídeos num relatório legível; `evaluate` compara o
resultado com o rótulo do UR Fall Detection Dataset; `to_events` publica os
achados como eventos `video` do contrato.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.anomaly_detection.events import build_event
from src.video_analysis.extraction import Extraction
from src.video_analysis.movement_detection import POSTURE_LABELS, MovementAnalysis, VideoEvent
from src.video_analysis.preprocessing import SAMPLE_CLIPS, clip_split, expected_label

SYSTEM = "fase4-video-analysis"

KIND_LABELS = {
    "fall_detected": "Queda",
    "bed_exit_attempt": "Saída do leito",
    "prolonged_immobility": "Imobilidade prolongada",
    "agitation": "Agitação",
    "video_check": "Sem desvio",
}
OBJECT_LABELS = {"bed": "cama", "couch": "sofá", "chair": "cadeira", "bench": "banco"}


def video_report(extraction: Extraction, analysis: MovementAnalysis) -> dict[str, Any]:
    """Resumo estruturado de um vídeo."""
    video = extraction.video
    processed = max(len(extraction.processed_frames), 1)
    frames_with_person = len({o.frame_index for o in extraction.observations})
    label = expected_label(video.name)
    predicted = "fall" if any(e.kind == "fall_detected" for e in analysis.events) else "adl"
    return {
        "video": video.to_dict(),
        "description": SAMPLE_CLIPS.get(video.name),
        "models": {"pose": extraction.pose_model, "objects": extraction.object_model},
        "processed_frames": len(extraction.processed_frames),
        "person_coverage": round(frames_with_person / processed, 3),
        "objects": [item.to_dict() for item in extraction.objects],
        "tracks": [series.summary() for series in analysis.series.values()],
        "events": [event.to_dict() for event in analysis.events],
        "deviations": len(analysis.anomalies),
        "dataset_label": label,
        "split": clip_split(video.name),
        "predicted_label": predicted if label else None,
    }


def evaluate(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Matriz de confusão queda x não queda para os vídeos com rótulo do dataset.

    Quando há clipes de ajuste e de validação, `by_split` traz a matriz de
    cada conjunto: só a de validação é uma medida independente.
    """
    labelled = [r for r in reports if r["dataset_label"]]
    if not labelled:
        return None
    result = _confusion(labelled)
    splits = sorted({r.get("split") for r in labelled if r.get("split")})
    if len(splits) > 1:
        result["by_split"] = {name: _confusion([r for r in labelled if r.get("split") == name]) for name in splits}
    return result


def _confusion(labelled: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(r["dataset_label"], r["predicted_label"]) for r in labelled]
    tp = sum(1 for truth, guess in pairs if truth == "fall" and guess == "fall")
    fn = sum(1 for truth, guess in pairs if truth == "fall" and guess != "fall")
    fp = sum(1 for truth, guess in pairs if truth != "fall" and guess == "fall")
    tn = sum(1 for truth, guess in pairs if truth != "fall" and guess != "fall")
    return {
        "videos": len(labelled),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
        "accuracy": round((tp + tn) / len(labelled), 3),
        "errors": [r["video"]["name"] for r in labelled if r["dataset_label"] != r["predicted_label"]],
    }


def render_markdown(reports: list[dict[str, Any]], evaluation: dict[str, Any] | None = None) -> str:
    """Relatório em Markdown: um quadro geral e os desvios de cada vídeo."""
    lines = [
        "# Relatório automático — análise de vídeo",
        "",
        "> Achados automatizados para avaliação profissional. Não constituem diagnóstico.",
        "",
        "| Vídeo | Duração | Pessoas | Objetos da cena | Desvios | Resultado |",
        "|---|---|---|---|---|---|",
    ]
    for report in reports:
        objects = ", ".join(sorted({OBJECT_LABELS.get(o["label"], o["label"]) for o in report["objects"]})) or "—"
        kinds = [KIND_LABELS[e["kind"]] for e in report["events"] if e["anomaly"]]
        lines.append(
            f"| `{report['video']['name']}` | {report['video']['duration_s']:.1f} s | {len(report['tracks'])} | "
            f"{objects} | {report['deviations']} | {', '.join(kinds) or 'Sem desvio'} |"
        )

    if evaluation:
        lines += [
            "",
            "## Comparação com o rótulo do dataset",
            "",
            f"{evaluation['videos']} vídeos rotulados (queda ou atividade do dia a dia):",
            "",
            "| | Detectou queda | Não detectou |",
            "|---|---|---|",
            f"| **Queda** | {evaluation['true_positive']} | {evaluation['false_negative']} |",
            f"| **Atividade normal** | {evaluation['false_positive']} | {evaluation['true_negative']} |",
            "",
            f"Sensibilidade {evaluation['sensitivity']}, especificidade {evaluation['specificity']}, "
            f"acurácia {evaluation['accuracy']}."
            + (f" Erros: {', '.join(evaluation['errors'])}." if evaluation["errors"] else ""),
        ]
        if evaluation.get("by_split"):
            names = {"ajuste": "Ajuste (limiares calibrados nestes clipes)", "validacao": "Validação (clipes fora do ajuste)"}
            lines += [
                "",
                "Por conjunto — só a validação é uma medida independente:",
                "",
                "| Conjunto | Vídeos | Quedas detectadas | Falsos alarmes | Sensibilidade | Especificidade |",
                "|---|---|---|---|---|---|",
            ]
            for name, part in evaluation["by_split"].items():
                falls = part["true_positive"] + part["false_negative"]
                normal = part["false_positive"] + part["true_negative"]
                lines.append(
                    f"| {names.get(name, name)} | {part['videos']} | {part['true_positive']} de {falls} | "
                    f"{part['false_positive']} de {normal} | {part['sensitivity']} | {part['specificity']} |"
                )

    for report in reports:
        lines += ["", f"## {report['video']['name']}", ""]
        if report["description"]:
            lines.append(f"Conteúdo segundo o dataset: {report['description']}.")
        lines.append(
            f"{report['processed_frames']} quadros processados a {report['video']['fps']:.0f} fps; pessoa visível em "
            f"{report['person_coverage'] * 100:.0f}% deles. Modelos: {report['models']['pose']} (postura) e "
            f"{report['models']['objects']} (objetos)."
        )
        for track in report["tracks"]:
            share = track["posture_share"]
            seen = ", ".join(f"{POSTURE_LABELS[k]} {v * 100:.0f}%" for k, v in share.items() if v > 0)
            lines.append(f"- Pessoa {track['track_id']} ({track['start_s']:.1f}–{track['end_s']:.1f} s): {seen}.")
        lines.append("")
        for event in report["events"]:
            marker = "**Desvio**" if event["anomaly"] else "Verificação"
            lines.append(
                f"- {marker} — {KIND_LABELS[event['kind']]} ({event['start_s']:.1f}–{event['end_s']:.1f} s, "
                f"prioridade sugerida {event['priority']}, confiança {event['confidence']:.2f}): {event['detail']}"
            )
    return "\n".join(lines) + "\n"


def to_events(
    extraction: Extraction,
    analysis: MovementAnalysis,
    *,
    patient_id: str,
    recorded_at: datetime,
    source_name: str = "urfd",
    keyframes: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Converte os achados em eventos `video` do contrato da Fase 4.

    `recorded_at` é o horário (com fuso) do início da gravação; o horário de
    cada evento é esse início mais o instante do achado no vídeo.
    """
    video = extraction.video
    uri = f"video://{source_name}/{video.name}"
    model = f"{extraction.pose_model} + {extraction.object_model}"
    keyframes = keyframes or {}
    events = []
    for index, event in enumerate(analysis.events):
        end = max(event.end_s, event.start_s + 0.1)
        frame_note = f"Quadro de referência do achado ({KIND_LABELS[event.kind].lower()})."
        if index in keyframes:
            frame_note += f" Imagem anotada: {keyframes[index]}."
        evidence = [
            {
                "type": "video_clip",
                "uri": uri,
                "start_s": round(event.start_s, 3),
                "end_s": round(end, 3),
                "confidence": round(event.confidence, 3),
                "description": f"Trecho do achado, pessoa {event.track_id}." if event.track_id >= 0 else "Vídeo inteiro.",
            },
            {
                "type": "video_frame",
                "uri": uri,
                "frame_index": int(event.frame_index),
                "description": frame_note[:300],
            },
        ]
        events.append(
            build_event(
                module="video",
                patient_id=patient_id,
                local_id=f"{video.name.rsplit('.', 1)[0]}-p{max(event.track_id, 0)}-{event.kind}-{int(event.start_s * 1000):06d}",
                timestamp=recorded_at + timedelta(seconds=event.start_s),
                event_type=event.kind,
                finding=event.detail,
                anomaly_detected=event.anomaly,
                priority=event.priority,
                evidence=evidence,
                system=SYSTEM,
                model=model,
            )
        )
    return events
