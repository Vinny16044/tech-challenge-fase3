
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from src.audio_validation import validate_audio

def test_audio_inexistente(tmp_path):
    caminho = tmp_path / "nao_existe.wav"

    with pytest.raises(FileNotFoundError):
        validate_audio(caminho)


def test_audio_vazio(tmp_path):
    caminho = tmp_path / "vazio.wav"

    sf.write(
        str(caminho),
        np.array([], dtype=np.float32),
        16000
    )

    with pytest.raises(ValueError, match="vazio"):
        validate_audio(caminho)


def test_audio_valido(tmp_path):
    caminho = tmp_path / "valido.wav"

    sample_rate = 16000
    tempo = np.arange(sample_rate) / sample_rate

    signal = 0.1 * np.sin(
        2 * np.pi * 440 * tempo
    )

    sf.write(str(caminho), signal, sample_rate)

    audio, sr = validate_audio(caminho)

    assert len(audio) == sample_rate
    assert sr == 16000
    assert np.all(np.isfinite(audio))
