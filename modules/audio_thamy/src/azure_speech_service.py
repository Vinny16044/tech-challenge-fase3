
import threading

import azure.cognitiveservices.speech as speechsdk


def transcribe_audio(audio_path, speech_key, speech_region, timeout=180):
    """Transcreve um arquivo de áudio usando Azure Speech."""

    if not speech_key or not speech_region:
        raise ValueError("Credenciais do Azure Speech incompletas.")

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )
    speech_config.speech_recognition_language = "pt-BR"

    audio_config = speechsdk.audio.AudioConfig(
        filename=str(audio_path)
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    transcription_parts = []
    recognition_errors = []
    recognition_finished = threading.Event()

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text.strip()

            if text:
                transcription_parts.append(text)

    def on_canceled(evt):
        recognition_errors.append(
            f"Reconhecimento cancelado: {evt.reason}; "
            f"detalhes: {evt.error_details}"
        )
        recognition_finished.set()

    def on_session_stopped(evt):
        recognition_finished.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    started = False

    try:
        recognizer.start_continuous_recognition_async().get()
        started = True

        if not recognition_finished.wait(timeout=timeout):
            raise TimeoutError(
                f"A transcrição não terminou em {timeout} segundos."
            )

    finally:
        if started:
            recognizer.stop_continuous_recognition_async().get()

    if recognition_errors:
        raise RuntimeError(recognition_errors[0])

    transcription = " ".join(transcription_parts).strip()

    if not transcription:
        raise RuntimeError(
            "O Azure Speech não reconheceu nenhuma fala no áudio."
        )

    return transcription
