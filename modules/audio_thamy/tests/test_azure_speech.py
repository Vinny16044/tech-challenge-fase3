
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import azure.cognitiveservices.speech as speechsdk

from src.azure_speech_service import transcribe_audio


@pytest.fixture
def mock_azure():
    """Simula o Azure Speech sem acessar a internet."""

    with (
        patch(
            "src.azure_speech_service.speechsdk.SpeechConfig"
        ),
        patch(
            "src.azure_speech_service.speechsdk.AudioConfig"
        ),
        patch(
            "src.azure_speech_service.speechsdk.SpeechRecognizer"
        ) as recognizer_class,
    ):
        recognizer = MagicMock()
        recognizer_class.return_value = recognizer

        yield recognizer


def test_transcricao_sucesso(mock_azure):
    """Simula uma transcrição concluída com sucesso."""

    def simulate_start():
        callback = mock_azure.recognized.connect.call_args.args[0]

        callback(
            SimpleNamespace(
                result=SimpleNamespace(
                    reason=speechsdk.ResultReason.RecognizedSpeech,
                    text="Estou com tosse."
                )
            )
        )

        stop_callback = (
            mock_azure.session_stopped.connect.call_args.args[0]
        )
        stop_callback(None)

    mock_azure.start_continuous_recognition_async.return_value.get.side_effect = (
        simulate_start
    )

    result = transcribe_audio(
        "audio_teste.wav",
        "chave_simulada",
        "brazilsouth"
    )

    assert result == "Estou com tosse."


def test_transcricao_sem_fala(mock_azure):
    """Simula um áudio sem fala reconhecida."""

    def simulate_start():
        callback = (
            mock_azure.session_stopped.connect.call_args.args[0]
        )
        callback(None)

    mock_azure.start_continuous_recognition_async.return_value.get.side_effect = (
        simulate_start
    )

    with pytest.raises(RuntimeError, match="nenhuma fala"):
        transcribe_audio(
            "audio_teste.wav",
            "chave_simulada",
            "brazilsouth"
        )


def test_transcricao_timeout(mock_azure):
    """Simula uma transcrição que não termina no prazo."""

    with pytest.raises(TimeoutError, match="não terminou"):
        transcribe_audio(
            "audio_teste.wav",
            "chave_simulada",
            "brazilsouth",
            timeout=0
        )


def test_credenciais_incompletas():
    """Verifica a rejeição de credenciais ausentes."""

    with pytest.raises(ValueError, match="Credenciais"):
        transcribe_audio(
            "audio_teste.wav",
            "",
            "brazilsouth"
        )


def test_transcricao_cancelada(mock_azure):
    """Simula um cancelamento do reconhecimento."""

    def simulate_start():
        callback = mock_azure.canceled.connect.call_args.args[0]

        callback(
            SimpleNamespace(
                reason="Error",
                error_details="Falha simulada de autenticação"
            )
        )

    mock_azure.start_continuous_recognition_async.return_value.get.side_effect = (
        simulate_start
    )

    with pytest.raises(RuntimeError, match="Reconhecimento cancelado"):
        transcribe_audio(
            "audio_teste.wav",
            "chave_simulada",
            "brazilsouth"
        )
