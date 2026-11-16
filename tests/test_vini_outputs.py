"""
Contrato 1.1 e saídas versionadas dos módulos de vídeo e dados clínicos.

`data/fase4_vini/` e as amostras de vídeo e clínico em `data/fase4_recebidos/`
são geradas por `py -m src.fase4_samples` a partir de dados públicos (UR Fall
Detection Dataset e MIMIC-IV Clinical Database Demo).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.contracts import EVENT_TYPES_BY_MODULE, SCHEMA_VERSION, MonitoringEventV1
from src.contracts.generate_schema import SCHEMA_PATH
from src.monitoring.workflow import MonitoringIntegrationGraph, load_inputs

from conftest import ROOT

EXAMPLES_DIR = ROOT / "data" / "fase4_vini"
RECEIVED_DIR = ROOT / "data" / "fase4_recebidos"
# Arquivos de eventos; `video_evaluation_urfd.json` é um resumo, não uma lista de eventos.
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*_events_*.json"))


def _events(path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else [payload]


def test_contract_version_is_1_1_and_still_accepts_1_0(clinical_event):
    assert SCHEMA_VERSION == "1.1"
    assert MonitoringEventV1.model_validate(clinical_event).schema_version == "1.0"
    assert MonitoringEventV1.model_validate(dict(clinical_event, schema_version="1.1")).schema_version == "1.1"
    assert {"prescription_change", "patient_transfer"} <= set(EVENT_TYPES_BY_MODULE["clinical"])


def test_prescription_evidence_rules(clinical_event):
    evidence = {"type": "prescription", "uri": "clinical://rx/1", "metric": "Heparin", "value": 5000, "unit": "unit"}
    event = dict(clinical_event, event_type="prescription_change", evidence=[evidence])
    assert MonitoringEventV1.model_validate(event).evidence[0].type == "prescription"

    incomplete = dict(event, evidence=[{"type": "prescription", "uri": "clinical://rx/1", "metric": "Heparin"}])
    with pytest.raises(ValidationError, match="exige value, unit"):
        MonitoringEventV1.model_validate(incomplete)


def test_new_types_belong_to_the_clinical_module_only(video_event):
    with pytest.raises(ValidationError, match="não é aceito para o módulo"):
        MonitoringEventV1.model_validate(dict(video_event, event_type="prescription_change"))


def test_examples_exist():
    names = [path.name for path in EXAMPLE_FILES]
    assert "video_events_urfd.json" in names
    assert any(name.startswith("clinical_events_P") for name in names)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_examples_follow_contract_and_schema(path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    events = _events(path)
    assert events
    for payload in events:
        MonitoringEventV1.model_validate(payload)
        jsonschema.validate(payload, schema)
    ids = [payload["event_id"] for payload in events]
    assert len(ids) == len(set(ids))


def _clip(event) -> str:
    return event["evidence"][0]["uri"].rsplit("/", 1)[-1]


def test_video_examples_are_correct_on_the_calibration_clips():
    from src.video_analysis.preprocessing import SAMPLE_CLIPS

    events = _events(EXAMPLES_DIR / "video_events_urfd.json")
    by_clip = {_clip(event): event for event in events}
    assert set(SAMPLE_CLIPS) <= set(by_clip)
    for clip in SAMPLE_CLIPS:
        event = by_clip[clip]
        if clip.startswith("fall-"):
            assert event["event_type"] == "fall_detected"
        else:
            assert not event["anomaly_detected"]
    assert all(event["source"]["model"] == "yolov8n-pose + yolov8n" for event in events)


def test_video_evaluation_matches_the_events_and_separates_validation():
    """O resumo versionado bate com os eventos e não mistura ajuste com validação."""
    summary = json.loads((EXAMPLES_DIR / "video_evaluation_urfd.json").read_text(encoding="utf-8"))
    events = _events(EXAMPLES_DIR / "video_events_urfd.json")
    detected = {_clip(event) for event in events if event["event_type"] == "fall_detected"}
    for video in summary["videos"]:
        assert (video["predicted_label"] == "fall") == (video["name"] in detected)

    overall, splits = summary["evaluation"], summary["evaluation"]["by_split"]
    assert set(splits) == {"ajuste", "validacao"}
    assert splits["ajuste"]["videos"] == 8 and splits["ajuste"]["accuracy"] == 1.0
    assert splits["ajuste"]["videos"] + splits["validacao"]["videos"] == overall["videos"]
    for key in ("true_positive", "false_negative", "false_positive", "true_negative"):
        assert splits["ajuste"][key] + splits["validacao"][key] == overall[key]
    misses = [v["name"] for v in summary["videos"] if v["dataset_label"] != v["predicted_label"]]
    assert sorted(misses) == sorted(overall["errors"])
    assert all(v["split"] == "validacao" for v in summary["videos"] if v["name"] in misses)


def test_clinical_examples_cover_the_three_detectors():
    path = next(p for p in EXAMPLE_FILES if p.name.startswith("clinical_events_P"))
    events = _events(path)
    assert {"vital_signs", "prescription_change"} <= {event["event_type"] for event in events}
    assert all(event["evidence"][0]["uri"].startswith("clinical://mimic-iv-demo/") for event in events)


def test_received_video_and_clinical_are_real_module_outputs():
    video = _events(RECEIVED_DIR / "P0001_video.json")
    clinical = _events(RECEIVED_DIR / "P0001_clinical.json")
    assert {event["source"]["system"] for event in video} == {"fase4-video-analysis"}
    assert all("ficticio" not in (event["source"].get("model") or "") for event in video + clinical)
    assert any(event["anomaly_detected"] for event in video)
    assert {event["event_type"] for event in clinical} >= {"vital_signs"}


def test_received_samples_raise_a_corroborated_alert(tmp_path):
    report = MonitoringIntegrationGraph(audit_path=tmp_path / "audit.jsonl").invoke(load_inputs([RECEIVED_DIR]))
    assert report["intake"]["rejected_events"] == []
    alert = report["groups"][0]["alert"]
    assert alert["priority"] == "critical"
    assert alert["anomalous_modalities"] == ["video", "clinical"]
    assert {"R1", "R2", "R3"} <= {reason["rule_id"] for reason in alert["priority_reasons"]}
    assert alert["review"]["status"] == "pending_medical_review"


def test_full_stay_runs_through_the_integration(tmp_path):
    """A internação inteira do paciente de exemplo passa pelo fluxo sem rejeições."""
    path = next(p for p in EXAMPLE_FILES if p.name.startswith("clinical_events_P"))
    report = MonitoringIntegrationGraph(audit_path=tmp_path / "audit.jsonl").invoke(load_inputs([path]))
    assert report["intake"]["rejected"] == 0
    assert report["intake"]["accepted"] == len(_events(path))
    assert len(report["alerts"]) >= 1
    assert all(group["sources_missing"] == ["audio", "video"] for group in report["groups"])
