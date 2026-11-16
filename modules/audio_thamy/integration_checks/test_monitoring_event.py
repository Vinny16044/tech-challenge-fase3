"""
Conversor audio_analysis 1.0.0 -> evento v1 da Fase 4, validado pelo mesmo
contrato usado no fluxo (MonitoringEventV1 e schemas/event_v1.schema.json).

Execução, a partir da raiz da Fase 4: py -m pytest modules/audio_thamy/integration_checks
"""

import copy
import json
from pathlib import Path

import pytest

from modules.audio_thamy.to_monitoring_event import (
    convert_audio_result,
    save_event,
    validate_event,
)
from src.contracts import MonitoringEventV1
from src.monitoring.workflow import MonitoringIntegrationGraph, load_inputs

MODULE_DIR = Path(__file__).resolve().parents[1]
ROOT = MODULE_DIR.parents[1]
SCHEMA = ROOT / "schemas" / "event_v1.schema.json"
RESULT_FILE = MODULE_DIR / "docs" / "consulta_simulada_resultado.json"


@pytest.fixture
def result():
    return json.loads(RESULT_FILE.read_text(encoding="utf-8"))


def convert(result, timestamp="2026-11-12T10:03:20-03:00"):
    return convert_audio_result(
        result, patient_id="P0001", recording_id="consulta-simulada-01", recorded_at=timestamp
    )


def test_package_result_maps_to_valid_audio_check(result):
    event = convert(result)
    validate_event(event, SCHEMA)

    assert event["event_id"] == "audio-P0001-consulta-simulada-01"
    assert event["module"] == "audio"
    assert event["event_type"] == "audio_check"
    assert event["patient_id"] == "P0001"
    assert event["timestamp"] == "2026-11-12T10:03:20-03:00"
    assert event["anomaly_detected"] is False
    assert event["priority"] == "low"
    assert event["finding"].startswith(
        "Termos sinalizados na transcrição para revisão humana: cansaço, falta de ar, tosse persistente."
    )
    assert event["evidence"] == [
        {
            "type": "audio_segment",
            "uri": "audio://thamy/consulta-simulada-01/consulta_simulada.wav",
            "start_s": 0.0,
            "end_s": 19.817,
            "description": "Gravação completa: o resultado não informa o instante de cada termo.",
        }
    ]
    assert event["source"] == {
        "system": "audio_analysis",
        "version": "1.0.0",
        "model": "azure_speech + azure_ai_language",
    }
    # A transcrição e a análise de texto não vão para o evento.
    assert "Doutor" not in json.dumps(event, ensure_ascii=False)


def test_reprocessing_keeps_identity(result):
    before = convert(result)
    result.update(analysis_id="different-run", processed_at="2026-11-16T17:00:00Z")
    assert convert(result) == before


def test_all_flagged_terms_are_kept(result):
    result["alerts"][0]["detected_terms"] = ["chiado", "cansaço", "chiado"]
    event = convert(result)
    assert "revisão humana: chiado, cansaço." in event["finding"]
    MonitoringEventV1.model_validate(event)


def test_no_keyword_alert(result):
    result["alerts"] = []
    event = convert(result)
    validate_event(event, SCHEMA)
    assert event["anomaly_detected"] is False
    assert "não sinalizou termos" in event["finding"]


def test_rejects_failed_analysis(result):
    result["status"] = "error"
    with pytest.raises(ValueError, match="status success"):
        convert(result)


def test_requires_recording_timezone(result):
    with pytest.raises(ValueError, match="fuso"):
        convert(result, "2026-11-12T10:03:20")


def test_rejects_invalid_patient_id(result):
    with pytest.raises(ValueError, match="P0001"):
        convert_audio_result(
            result, patient_id="Maria", recording_id="x1", recorded_at="2026-11-12T10:03:20-03:00"
        )


@pytest.mark.parametrize("duration", [0, float("nan"), "19.817", None])
def test_rejects_invalid_duration(result, duration):
    result["audio_metadata"]["duration_seconds"] = duration
    with pytest.raises(ValueError, match="duração"):
        convert(result)


def test_save_replaces_file_with_complete_utf8_event(tmp_path, result):
    outfile = tmp_path / "output" / "audio.json"
    save_event({"old": True}, outfile)
    expected = convert(result)
    save_event(expected, outfile)
    assert json.loads(outfile.read_text(encoding="utf-8")) == expected
    assert list(outfile.parent.iterdir()) == [outfile]


def test_converted_audio_joins_demo_video_and_clinical(tmp_path, result):
    """O áudio convertido consolida com o vídeo e o clínico da demonstração."""
    inputs = tmp_path / "janela"
    save_event(convert(result), inputs / "P0001_audio.json")
    for name in ("P0001_video.json", "P0001_clinical.json"):
        (inputs / name).write_text((ROOT / "data" / "fase4" / name).read_text(encoding="utf-8"), encoding="utf-8")

    report = MonitoringIntegrationGraph(audit_path=tmp_path / "audit.jsonl").invoke(load_inputs([inputs]))

    assert report["intake"]["accepted"] == 3
    assert report["intake"]["rejected"] == 0 and report["intake"]["duplicates"] == 0
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["sources_received"] == ["audio", "video", "clinical"]
    audio = next(e for e in group["events"] if e["module"] == "audio")
    assert audio["event_id"] == "audio-P0001-consulta-simulada-01"
    assert audio["source"]["system"] == "audio_analysis"
    # Sem anomalia no áudio: só vídeo e clínico corroboram (R1 high -> R3 critical, sem R4).
    alert = group["alert"]
    assert alert["anomalous_modalities"] == ["video", "clinical"]
    assert [r["rule_id"] for r in alert["priority_reasons"]] == ["R1", "R3"]
    assert alert["priority"] == "critical"
    assert alert["review"]["status"] == "pending_medical_review"
    assert audio["event_id"] not in alert["event_ids"]


def test_result_document_is_not_modified_by_conversion(result):
    snapshot = copy.deepcopy(result)
    convert(result)
    assert result == snapshot
