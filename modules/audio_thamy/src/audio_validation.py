
from pathlib import Path

import librosa
import numpy as np


def validate_audio(audio_path):
    """Valida um arquivo antes do processamento."""

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de áudio não encontrado: {audio_path}"
        )

    try:
        signal, sample_rate = librosa.load(
            str(audio_path),
            sr=16000,
            mono=True
        )
    except Exception as exc:
        raise ValueError(
            f"Não foi possível ler o arquivo de áudio: {audio_path}"
        ) from exc

    if signal.size == 0:
        raise ValueError("O arquivo de áudio está vazio.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "O áudio contém valores numéricos inválidos."
        )

    return signal, sample_rate
