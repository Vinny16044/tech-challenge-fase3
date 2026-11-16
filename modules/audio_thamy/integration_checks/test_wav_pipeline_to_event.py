"""
Teste de ponta a ponta sem credenciais: o pipeline da Thamy processa o WAV
simulado (validação e atributos acústicos calculados com librosa), com Azure
Speech e Azure Language simulados; o resultado audio_analysis 1.0.0 passa
pelo conversor e o evento v1 é validado pelo contrato.

O pipeline roda em um subprocesso com cwd = modules/audio_thamy, porque o
pacote `src` da Thamy tem o mesmo nome do `src` do projeto.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from modules.audio_thamy.to_monitoring_event import convert_audio_result, validate_event

pytest.importorskip("librosa", reason="instale modules/audio_thamy/requirements.txt")
pytest.importorskip("azure.cognitiveservices.speech", reason="instale modules/audio_thamy/requirements.txt")
pytest.importorskip("azure.ai.textanalytics", reason="instale modules/audio_thamy/requirements.txt")

MODULE_DIR = Path(__file__).resolve().parents[1]
ROOT = MODULE_DIR.parents[1]
WAV = MODULE_DIR / "data" / "raw" / "consulta_simulada.wav"
RESULT_FILE = MODULE_DIR / "docs" / "consulta_simulada_resultado.json"

# Só as chamadas ao Azure são substituídas; a leitura do WAV e o librosa são executados.
RUNNER = r"""
import json, sys
from src import audio_pipeline

reference = json.load(open(sys.argv[1], encoding="utf-8"))
audio_pipeline.load_dotenv = lambda *args, **kwargs: None  # nunca lê um .env real
audio_pipeline.transcribe_audio = lambda **kwargs: reference["transcription"]["text"]
audio_pipeline.analyze_text = lambda **kwargs: reference["text_analysis"]
audio_pipeline.run_pipeline(audio_path=sys.argv[2], output_path=sys.argv[3], error_path=sys.argv[4])
"""


def test_wav_pipeline_with_simulated_azure_produces_valid_event(tmp_path):
    output = tmp_path / "audio_pipeline_result.json"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AZURE_") and key != "PYTHONPATH"
    }
    env.update(
        AZURE_SPEECH_KEY="fake-key",
        AZURE_SPEECH_REGION="brazilsouth",
        AZURE_LANGUAGE_KEY="fake-key",
        AZURE_LANGUAGE_ENDPOINT="https://example.test",
        PYTHONIOENCODING="utf-8",
    )
    process = subprocess.run(
        [sys.executable, "-c", RUNNER, str(RESULT_FILE), str(WAV), str(output), str(tmp_path / "error.json")],
        cwd=MODULE_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert process.returncode == 0, process.stderr

    result = json.loads(output.read_text(encoding="utf-8"))
    reference = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["audio_metadata"]["file_name"] == "consulta_simulada.wav"
    # A duração calculada do WAV deve coincidir com a do resultado enviado pela Thamy.
    assert result["audio_metadata"]["duration_seconds"] == pytest.approx(
        reference["audio_metadata"]["duration_seconds"], abs=0.01
    )
    assert result["acoustic_features"]["detected_event_count"] > 0
    assert result["alerts"][0]["detected_terms"] == ["cansaço", "falta de ar", "tosse persistente"]

    kwargs = dict(patient_id="P0001", recording_id="consulta-simulada-01", recorded_at="2026-11-12T10:03:20-03:00")
    event = convert_audio_result(result, **kwargs)
    validate_event(event, ROOT / "schemas" / "event_v1.schema.json")
    # O evento deve ser idêntico ao convertido a partir do resultado enviado.
    assert event == convert_audio_result(reference, **kwargs)
