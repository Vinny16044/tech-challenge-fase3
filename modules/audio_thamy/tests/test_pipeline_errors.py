
import json

from src.pipeline_errors import (
    build_error_result,
    save_error_result
)


def test_error_status():
    result = build_error_result(
        TimeoutError("Tempo excedido"),
        "transcription",
        "consulta.wav"
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "TimeoutError"
    assert result["error"]["stage"] == "transcription"


def test_error_has_required_fields():
    result = build_error_result(
        RuntimeError("Falha simulada"),
        "text_analysis",
        "consulta.wav"
    )

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
        "error"
    ]

    for field in required:
        assert field in result


def test_save_error_json(tmp_path):
    output_path = tmp_path / "pipeline_error.json"

    save_error_result(
        ValueError("Áudio inválido"),
        "audio_validation",
        "consulta.wav",
        output_path
    )

    assert output_path.exists()

    with open(output_path, encoding="utf-8") as file:
        result = json.load(file)

    assert result["status"] == "error"
    assert result["error"]["stage"] == "audio_validation"
    assert result["audio_metadata"]["file_name"] == "consulta.wav"
