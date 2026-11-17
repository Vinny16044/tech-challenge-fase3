"""
Painel da Fase 4 — monitoramento multimodal (áudio, vídeo e dados clínicos).

Rotas acrescentadas ao servidor da Fase 3 (`src/api/server.py`):

  GET    /fase4                      -> painel (src/api/static/fase4.html)
  GET    /api/fase4/status           -> o que está disponível nesta máquina
  GET    /api/fase4/alertas          -> fluxo integrado sobre data/fase4_recebidos
  POST   /api/fase4/revisao          -> registra a decisão do profissional
  DELETE /api/fase4/revisao          -> apaga as decisões (para refazer a demonstração)
  GET    /api/fase4/videos           -> clipes analisados, eventos e validação
  GET    /api/fase4/clinico          -> eventos e série de sinais vitais do paciente de exemplo
  GET    /api/fase4/audio            -> resultado do módulo de áudio
  GET    /media/video/{clipe}/{arq}  -> quadros e vídeos anotados (output/video)
  GET    /media/audio/{arq}          -> gravação da consulta simulada

O painel lê as saídas já geradas pelos módulos. Vídeos anotados e a série de
sinais vitais dependem de arquivos locais (`output/video/` e
`data/clinical/raw/`); sem eles, o painel mostra só os resultados versionados.

Uso:  python manage.py web   ->   http://localhost:8000/fase4
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"

RECEIVED_DIR = ROOT / "data" / "fase4_recebidos"
EXAMPLES_DIR = ROOT / "data" / "fase4_vini"
VIDEO_OUTPUT = ROOT / "output" / "video"
PANEL_DIR = ROOT / "output" / "fase4_painel"
REVIEWS_FILE = PANEL_DIR / "revisoes.json"
AUDIT_FILE = ROOT / "logs" / "audit-fase4-painel.jsonl"
AUDIO_DIR = ROOT / "modules" / "audio_thamy"
AUDIO_RESULT = AUDIO_DIR / "docs" / "consulta_simulada_resultado.json"
AUDIO_RAW = AUDIO_DIR / "data" / "raw"

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
MEDIA_TYPES = {".jpg": "image/jpeg", ".webm": "video/webm", ".mp4": "video/mp4", ".wav": "audio/wav"}

router = APIRouter()
_lock = threading.Lock()
_clinical_lock = threading.Lock()  # separado: o primeiro cálculo leva alguns segundos
_clinical_cache: dict[str, Any] | None = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------
# Página e estado
# --------------------------------------------------------------------------


@router.get("/fase4", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(STATIC / "fase4.html")


@router.get("/api/fase4/status")
def status() -> dict[str, Any]:
    return {
        "received_samples": sorted(p.name for p in RECEIVED_DIR.glob("*.json")),
        "video_processed": (VIDEO_OUTPUT / "report.json").exists(),
        "video_annotated": any(VIDEO_OUTPUT.glob("*/anotado.webm")),
        "clinical_raw": (ROOT / "data" / "clinical" / "raw" / "chartevents.csv.gz").exists(),
        "audio_result": AUDIO_RESULT.exists(),
        "reviews": len(_load_reviews()),
    }


# --------------------------------------------------------------------------
# Alerta integrado e revisão humana
# --------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    alert_id: str = Field(pattern=r"^ALT-[0-9a-f]{12}$")
    decision: Literal["confirmed", "dismissed"]
    reviewer_id: str = Field(min_length=2, max_length=60)
    reviewer_name: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=1000)


def _load_reviews() -> list[dict[str, Any]]:
    if not REVIEWS_FILE.exists():
        return []
    try:
        data = _read_json(REVIEWS_FILE)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save_reviews(reviews: list[dict[str, Any]]) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_FILE.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_flow(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Roda o fluxo de integração (o mesmo da linha de comando) com as decisões guardadas."""
    try:
        from src.monitoring.workflow import MonitoringIntegrationGraph, load_inputs
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Integração indisponível: {exc}") from exc
    if not any(RECEIVED_DIR.glob("*.json")):
        raise HTTPException(status_code=404, detail=f"Sem amostras em {RECEIVED_DIR}")
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    graph = MonitoringIntegrationGraph(audit_path=AUDIT_FILE)
    return graph.invoke(load_inputs([RECEIVED_DIR]), review_decisions=reviews)


@router.get("/api/fase4/alertas")
def alerts() -> dict[str, Any]:
    with _lock:
        report = _run_flow(_load_reviews())
    return {
        "run_id": report["run_id"],
        "generated_at": report["generated_at"],
        "contract_version": report["contract_version"],
        "rules_version": report["rules_version"],
        "config": report["config"],
        "disclaimer": report["disclaimer"],
        "intake": {key: report["intake"][key] for key in ("events_received", "accepted", "rejected", "duplicates")},
        "rejected_events": report["intake"]["rejected_events"],
        "groups": report["groups"],
        "review_results": report["review_decisions"],
    }


@router.post("/api/fase4/revisao")
def review(request: ReviewRequest) -> dict[str, Any]:
    """Registra a decisão de um profissional. Cada alerta aceita uma única decisão."""
    decision = {
        **request.model_dump(),
        "decided_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    with _lock:
        reviews = _load_reviews()
        report = _run_flow(reviews + [decision])
        outcome = report["review_decisions"][-1] if report["review_decisions"] else None
        if not outcome or outcome.get("outcome") != "recorded":
            reason = (outcome or {}).get("reason", "decisão não aplicada")
            raise HTTPException(status_code=409, detail=f"Decisão recusada: {reason}")
        _save_reviews(reviews + [decision])
    return {"recorded": outcome}


@router.delete("/api/fase4/revisao")
def reset_reviews() -> dict[str, Any]:
    with _lock:
        removed = len(_load_reviews())
        if REVIEWS_FILE.exists():
            REVIEWS_FILE.unlink()
    return {"removed": removed}


# --------------------------------------------------------------------------
# Vídeo
# --------------------------------------------------------------------------


def _media_url(clip: str, name: str | None) -> str | None:
    if not name or not (VIDEO_OUTPUT / clip / name).exists():
        return None
    return f"/media/video/{clip}/{name}"


@router.get("/api/fase4/videos")
def videos() -> dict[str, Any]:
    """Clipes analisados. Usa a execução local se existir; senão, o resumo versionado."""
    local = VIDEO_OUTPUT / "report.json"
    if local.exists():
        report = _read_json(local)
        items = []
        for video in report["videos"]:
            clip = video["video"]["name"].rsplit(".", 1)[0]
            keyframes = [url for name in video.get("keyframes", []) if (url := _media_url(clip, name))]
            items.append(
                {
                    "name": video["video"]["name"],
                    "clip": clip,
                    "description": video.get("description"),
                    "split": video.get("split"),
                    "dataset_label": video.get("dataset_label"),
                    "predicted_label": video.get("predicted_label"),
                    "duration_s": video["video"]["duration_s"],
                    "fps": video["video"]["fps"],
                    "processed_frames": video["processed_frames"],
                    "person_coverage": video["person_coverage"],
                    "models": video["models"],
                    "objects": video["objects"],
                    "tracks": video["tracks"],
                    "events": video["events"],
                    "video_url": _media_url(clip, video.get("annotated_video") or "anotado.webm"),
                    "keyframes": keyframes,
                }
            )
        return {"source": "local", "evaluation": report.get("evaluation"), "videos": items}

    versioned = EXAMPLES_DIR / "video_evaluation_urfd.json"
    if not versioned.exists():
        raise HTTPException(status_code=404, detail="Nenhum resultado de vídeo disponível")
    summary = _read_json(versioned)
    items = [
        {
            "name": video["name"],
            "clip": video["name"].rsplit(".", 1)[0],
            "split": video["split"],
            "dataset_label": video["dataset_label"],
            "predicted_label": video["predicted_label"],
            "events": video["events"],
            "video_url": None,
            "keyframes": [],
        }
        for video in summary["videos"]
    ]
    return {"source": "versioned", "evaluation": summary["evaluation"], "videos": items}


def _serve(base: Path, *parts: str) -> FileResponse:
    """Entrega um arquivo de mídia, sem permitir sair da pasta base."""
    if not all(SAFE_NAME.fullmatch(part) for part in parts):
        raise HTTPException(status_code=404)
    path = base.joinpath(*parts).resolve()
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None or base.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type=media_type)


@router.get("/media/video/{clip}/{name}", include_in_schema=False)
def video_media(clip: str, name: str) -> FileResponse:
    return _serve(VIDEO_OUTPUT, clip, name)


@router.get("/media/audio/{name}", include_in_schema=False)
def audio_media(name: str) -> FileResponse:
    return _serve(AUDIO_RAW, name)


# --------------------------------------------------------------------------
# Dados clínicos
# --------------------------------------------------------------------------


def _clinical_series(subject: int) -> dict[str, Any]:
    """
    Série de sinais vitais do paciente, avaliada observação a observação.

    Precisa do MIMIC-IV Demo baixado e de pandas/scikit-learn. O modelo da
    população é treinado uma vez e fica em memória.
    """
    from src.anomaly_detection.data_loader import load_vitals
    from src.anomaly_detection.vital_signs import (
        PROFILES,
        SCORED_SIGNALS,
        PopulationModel,
        analyze_frame,
        carry_forward,
    )

    config = PROFILES["icu"]()
    vitals = load_vitals()
    model = PopulationModel().fit(carry_forward(vitals, config.carry_minutes))
    frame = vitals[vitals["subject_id"] == subject]
    points = []
    for finding in analyze_frame(frame, config, model):
        point: dict[str, Any] = {
            "t": finding.timestamp.isoformat(),
            "stay_id": finding.meta["stay_id"],
            "score": finding.score,
            "anomaly": finding.anomaly,
            "emit": finding.emit,
            "priority": finding.priority,
            "layers": finding.layers,
            "reason": finding.emit_reason,
        }
        for signal in SCORED_SIGNALS:
            point[signal] = finding.measured.get(signal)
        points.append(point)
    return {"profile": "icu", "points": points}


@router.get("/api/fase4/clinico")
def clinical() -> dict[str, Any]:
    global _clinical_cache
    files = sorted(EXAMPLES_DIR.glob("clinical_events_P*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="Nenhum resultado clínico disponível")
    events = _read_json(files[0])
    patient_id = events[0]["patient_id"] if events else files[0].stem.rsplit("_", 1)[-1]

    with _clinical_lock:
        if _clinical_cache is None:
            try:
                _clinical_cache = _clinical_series(int(patient_id.lstrip("P")))
            except Exception as exc:  # noqa: BLE001 — dados ou bibliotecas ausentes: o painel segue sem o gráfico
                _clinical_cache = {"profile": None, "points": [], "unavailable": f"{type(exc).__name__}: {exc}"}
        series = _clinical_cache
    return {"patient_id": patient_id, "source_file": files[0].name, "events": events, "series": series}


# --------------------------------------------------------------------------
# Áudio
# --------------------------------------------------------------------------


@router.get("/api/fase4/audio")
def audio() -> dict[str, Any]:
    if not AUDIO_RESULT.exists():
        raise HTTPException(status_code=404, detail="Resultado do módulo de áudio não encontrado")
    result = _read_json(AUDIO_RESULT)
    features = result.get("acoustic_features", {})
    file_name = result.get("audio_metadata", {}).get("file_name")
    event_file = RECEIVED_DIR / "P0001_audio.json"
    return {
        "processed_at": result.get("processed_at"),
        "metadata": result.get("audio_metadata"),
        "transcription": result.get("transcription"),
        "text_analysis": result.get("text_analysis"),
        "alerts": result.get("alerts", []),
        "limitations": result.get("limitations"),
        "acoustic": {key: value for key, value in features.items() if key not in ("mfcc_mean", "mfcc_std")},
        "audio_url": f"/media/audio/{file_name}" if file_name and (AUDIO_RAW / file_name).exists() else None,
        "event": _read_json(event_file) if event_file.exists() else None,
    }
