
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[1]
JSON_FILE = BASE_DIR / "outputs" / "audio_pipeline_result.json"



@pytest.fixture
def result(tmp_path, monkeypatch):
    """
    Gera um resultado simulado do pipeline para
    validar o contrato JSON sem acessar o Azure.
    """
    import numpy as np
    from src import audio_pipeline

    monkeypatch.setenv("AZURE_SPEECH_KEY", "fake-key")
    monkeypatch.setenv(
        "AZURE_SPEECH_REGION", "brazilsouth"
    )
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "fake-key")
    monkeypatch.setenv(
        "AZURE_LANGUAGE_ENDPOINT",
        "https://example.test"
    )

    monkeypatch.setattr(
        audio_pipeline,
        "validate_audio",
        lambda path: (np.zeros(16000), 16000)
    )

    monkeypatch.setattr(
        audio_pipeline,
        "transcribe_audio",
        lambda **kwargs: (
            "Paciente relata cansaço e tosse persistente."
        )
    )

    monkeypatch.setattr(
        audio_pipeline,
        "analyze_text",
        lambda **kwargs: {
            "provider": "azure_ai_language",
            "sentiment": {
                "label": "negative",
                "confidence": {
                    "positive": 0.1,
                    "neutral": 0.2,
                    "negative": 0.7
                }
            },
            "key_phrases": [
                "cansaço",
                "tosse persistente"
            ],
            "entities": []
        }
    )

    monkeypatch.setattr(
        audio_pipeline,
        "extract_acoustic_features",
        lambda path: {
            "duration_seconds": 1.0,
            "sample_rate_hz": 16000,
            "rms_mean": 0.02,
            "zcr_mean": 0.1,
            "spectral_centroid_mean_hz": 500.0,
            "mfcc_mean": [0.0] * 13,
            "mfcc_std": [0.0] * 13,
            "detected_event_count": 0,
            "detected_events": [],
            "event_detection_method": "rms_threshold",
            "event_interpretation": (
                "sound_activity_not_clinical_events"
            )
        }
    )

    output_path = tmp_path / "result.json"
    error_path = tmp_path / "error.json"

    return audio_pipeline.run_pipeline(
        audio_path=tmp_path / "audio.wav",
        output_path=output_path,
        error_path=error_path
    )

def test_required_fields(result):
    """Verifica os campos obrigatórios do contrato."""
    required = [
        "module",
        "schema_version",
        "analysis_id",
        "status",
        "processed_at",
        "audio_metadata",
        "transcription",
        "text_analysis",
        "acoustic_features",
        "alerts",
    ]

    for field in required:
        assert field in result, f"Campo ausente: {field}"


def test_module_identity(result):
    """Verifica identificação e versão do módulo."""
    assert result["module"] == "audio_analysis"
    assert result["schema_version"] == "1.0.0"
    assert result["status"] == "success"


def test_transcription(result):
    """Verifica se existe uma transcrição válida."""
    transcription = result["transcription"]

    assert isinstance(transcription["text"], str)
    assert transcription["text"].strip()
    assert transcription["language"] == "pt-BR"


def test_acoustic_features(result):
    """Verifica os principais indicadores acústicos."""
    features = result["acoustic_features"]

    assert features["duration_seconds"] > 0
    assert features["sample_rate_hz"] == 16000
    assert features["rms_mean"] >= 0
    assert 0 <= features["zcr_mean"] <= 1
    assert len(features["mfcc_mean"]) == 13
    assert len(features["mfcc_std"]) == 13


def test_detected_events(result):
    """Verifica os intervalos de atividade sonora."""
    features = result["acoustic_features"]
    duration = features["duration_seconds"]
    events = features["detected_events"]

    assert features["detected_event_count"] == len(events)

    for event in events:
        assert 0 <= event["start_seconds"]
        assert event["start_seconds"] < event["end_seconds"]
        assert event["end_seconds"] <= duration


def test_alerts_structure(result):
    """Verifica a estrutura dos alertas."""
    assert isinstance(result["alerts"], list)

    for alert in result["alerts"]:
        assert "type" in alert
        assert "status" in alert
        assert "requires_human_review" in alert
        assert isinstance(alert["requires_human_review"], bool)

