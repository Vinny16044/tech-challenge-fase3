import json

import numpy as np
import pytest

from src import audio_pipeline


@pytest.fixture
def azure_credentials(monkeypatch):
    """Simula credenciais sem acessar o Azure."""

    monkeypatch.setenv("AZURE_SPEECH_KEY", "fake-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "brazilsouth")
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "fake-key")
    monkeypatch.setenv(
        "AZURE_LANGUAGE_ENDPOINT",
        "https://example.test"
    )


def test_configuration_error(
    tmp_path,
    monkeypatch
):
    """Verifica falha de configuração."""

    monkeypatch.setattr(
        audio_pipeline,
        "load_dotenv",
        lambda *args, **kwargs: None
    )

    for key in [
        "AZURE_SPEECH_KEY",
        "AZURE_SPEECH_REGION",
        "AZURE_LANGUAGE_KEY",
        "AZURE_LANGUAGE_ENDPOINT"
    ]:
        monkeypatch.delenv(key, raising=False)

    output_path = tmp_path / "success.json"
    error_path = tmp_path / "error.json"

    with pytest.raises(ValueError):
        audio_pipeline.run_pipeline(
            audio_path=tmp_path / "audio.wav",
            output_path=output_path,
            error_path=error_path
        )

    result = json.loads(
        error_path.read_text(encoding="utf-8")
    )

    assert result["status"] == "error"
    assert result["error"]["stage"] == "configuration"
    assert not output_path.exists()


def test_transcription_error(
    tmp_path,
    monkeypatch,
    azure_credentials
):
    """Simula uma falha no Azure Speech."""

    monkeypatch.setattr(
        audio_pipeline,
        "validate_audio",
        lambda path: (
            np.zeros(16000),
            16000
        )
    )

    def fake_transcription(**kwargs):
        raise TimeoutError(
            "Timeout simulado do Azure Speech"
        )

    monkeypatch.setattr(
        audio_pipeline,
        "transcribe_audio",
        fake_transcription
    )

    output_path = tmp_path / "success.json"
    error_path = tmp_path / "error.json"

    with pytest.raises(TimeoutError):
        audio_pipeline.run_pipeline(
            audio_path=tmp_path / "audio.wav",
            output_path=output_path,
            error_path=error_path
        )

    result = json.loads(
        error_path.read_text(encoding="utf-8")
    )

    assert result["status"] == "error"
    assert result["error"]["stage"] == "transcription"
    assert result["error"]["type"] == "TimeoutError"
    assert not output_path.exists()


def test_pipeline_success(
    tmp_path,
    monkeypatch,
    azure_credentials
):
    """Valida a execução completa com serviços simulados."""

    monkeypatch.setattr(
        audio_pipeline,
        "validate_audio",
        lambda path: (
            np.zeros(16000),
            16000
        )
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

    output_path = tmp_path / "success.json"
    error_path = tmp_path / "error.json"

    error_path.write_text(
        '{"status": "error"}',
        encoding="utf-8"
    )

    result = audio_pipeline.run_pipeline(
        audio_path=tmp_path / "audio.wav",
        output_path=output_path,
        error_path=error_path
    )

    assert result["status"] == "success"
    assert result["module"] == "audio_analysis"
    assert result["schema_version"] == "1.0.0"
    assert len(result["alerts"]) == 1

    assert output_path.exists()
    assert not error_path.exists()

    saved_result = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_result["status"] == "success"
    assert saved_result["transcription"]["text"] == (
        "Paciente relata cansaço e tosse persistente."
    )
