"""Contrato JSON v1: exemplos válidos, regras de validação e schema gerado."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.contracts import MonitoringEventV1, build_event_id
from src.contracts.generate_schema import SCHEMA_PATH, render_schema


@pytest.mark.parametrize("fixture_name", ["audio_event", "video_event", "clinical_event"])
def test_reference_examples_are_valid(fixture_name, request):
    payload = request.getfixturevalue(fixture_name)
    event = MonitoringEventV1.model_validate(payload)
    assert event.patient_id == "P0001"
    assert event.timestamp.tzinfo is not None


def test_reference_examples_share_patient_and_15min_window(audio_event, video_event, clinical_event):
    events = [MonitoringEventV1.model_validate(p) for p in (audio_event, video_event, clinical_event)]
    assert {e.module for e in events} == {"audio", "video", "clinical"}
    starts = {int(e.timestamp.timestamp()) // 900 for e in events}
    assert len(starts) == 1


def test_published_schema_matches_model():
    """O schema versionado deve ser exatamente o gerado pelo modelo."""
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_schema()


def test_examples_validate_against_json_schema(audio_event, video_event, clinical_event):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for payload in (audio_event, video_event, clinical_event):
        jsonschema.validate(payload, schema)

    bad = dict(audio_event, event_type="fall_detected")  # tipo de vídeo em evento de áudio
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"timestamp": "2026-11-12T10:03:20"}, "timezone"),
        ({"event_type": "fall_detected"}, "não é aceito para o módulo"),
        ({"event_id": "video-P0001-x123"}, "prefixo do módulo"),
        ({"patient_id": "Maria"}, "pattern"),
        ({"schema_version": "2.0"}, "1.0"),
        ({"priority": "urgent"}, "low"),
        ({"anomaly_detected": False, "priority": "high"}, "sem anomalia"),
        ({"evidence": []}, "at least 1"),
        ({"campo_extra": 1}, "Extra inputs"),
    ],
)
def test_invalid_events_are_rejected(make_event, overrides, message):
    with pytest.raises(ValidationError) as exc:
        MonitoringEventV1.model_validate(make_event(**overrides))
    assert message in str(exc.value)


def test_missing_required_field(audio_event):
    payload = dict(audio_event)
    del payload["finding"]
    with pytest.raises(ValidationError, match="finding"):
        MonitoringEventV1.model_validate(payload)


def test_evidence_must_match_module(make_event):
    evidence = [{"type": "clinical_measurement", "uri": "clinical://x/1", "metric": "hr", "value": 1, "unit": "bpm"}]
    with pytest.raises(ValidationError, match="não é aceita para o módulo"):
        MonitoringEventV1.model_validate(make_event(evidence=evidence))


def test_audio_segment_requires_offsets(make_event):
    evidence = [{"type": "audio_segment", "uri": "audio://leito/1.wav", "start_s": 10}]
    with pytest.raises(ValidationError, match="start_s e end_s"):
        MonitoringEventV1.model_validate(make_event(evidence=evidence))


def test_build_event_id_is_stable():
    first = build_event_id("audio", "P0001", "rec0007.wav#120-160")
    assert first == build_event_id("audio", "P0001", "rec0007.wav#120-160")
    assert first != build_event_id("audio", "P0001", "rec0007.wav#160-200")
    assert first.startswith("audio-P0001-")


def test_content_hash_ignores_key_order(audio_event):
    reordered = dict(reversed(list(audio_event.items())))
    a = MonitoringEventV1.model_validate(audio_event).content_hash()
    b = MonitoringEventV1.model_validate(reordered).content_hash()
    assert a == b
