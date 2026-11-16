
from pathlib import Path
import os

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

if not SPEECH_KEY or not SPEECH_REGION:
    raise ValueError(
        "Configure AZURE_SPEECH_KEY e "
        "AZURE_SPEECH_REGION no arquivo .env"
    )

# Áudio de consulta simulada em português
AUDIO_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "consulta_simulada.wav"
)

if not AUDIO_FILE.exists():
    raise FileNotFoundError(
        f"Áudio não encontrado: {AUDIO_FILE}"
    )

speech_config = speechsdk.SpeechConfig(
    subscription=SPEECH_KEY,
    region=SPEECH_REGION
)

# A amostra Coswara de contagem está em inglês
speech_config.speech_recognition_language = "pt-BR"

audio_config = speechsdk.audio.AudioConfig(
    filename=str(AUDIO_FILE)
)

recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config
)

print("Enviando áudio para transcrição...")

result = recognizer.recognize_once()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("\nTranscrição:")
    print(result.text)

elif result.reason == speechsdk.ResultReason.NoMatch:
    print("Nenhuma fala reconhecida.")

elif result.reason == speechsdk.ResultReason.Canceled:
    details = result.cancellation_details
    print("Reconhecimento cancelado.")
    print("Motivo:", details.reason)
    print("Detalhes:", details.error_details)
