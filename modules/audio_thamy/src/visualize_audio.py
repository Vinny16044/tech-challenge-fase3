
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. CONFIGURAÇÕES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "breathing-deep",
    "breathing-shallow",
    "cough-heavy",
    "counting-normal",
    "vowel-a",
]

# Parâmetros para análise espectral
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
N_MFCC = 13


# --------------------------------------------------
# 2. LOCALIZAR UMA AMOSTRA DE CADA CATEGORIA
# --------------------------------------------------

audio_files = sorted(AUDIO_DIR.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(
        f"Nenhum arquivo WAV encontrado em {AUDIO_DIR}"
    )

selected_files = {}

for category in CATEGORIES:
    matches = [
        file for file in audio_files
        if file.stem.endswith("_" + category)
    ]

    if matches:
        selected_files[category] = matches[0]
    else:
        print(f"Aviso: nenhuma amostra encontrada para {category}")


print("\n========== VISUALIZAÇÃO DOS ÁUDIOS ==========")
print(f"Categorias encontradas: {len(selected_files)}")


# --------------------------------------------------
# 3. PROCESSAR E VISUALIZAR CADA ÁUDIO
# --------------------------------------------------

for category, audio_path in selected_files.items():

    print(f"\nProcessando: {audio_path.name}")

    # Carregar áudio sem alterar sua taxa original
    signal, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    duration = librosa.get_duration(
        y=signal,
        sr=sample_rate
    )

    time_axis = np.arange(len(signal)) / sample_rate

    # Espectrograma Mel
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )

    # Extração dos MFCCs
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------
    # 4. CRIAR FIGURA COM TRÊS VISUALIZAÇÕES
    # --------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 12)
    )

    fig.suptitle(
        f"Análise de áudio: {category}",
        fontsize=16,
        fontweight="bold"
    )

    # A) Forma de onda
    librosa.display.waveshow(
        signal,
        sr=sample_rate,
        ax=axes[0]
    )

    axes[0].set_title("Forma de onda (Waveform)")
    axes[0].set_xlabel("Tempo (segundos)")
    axes[0].set_ylabel("Amplitude")

    # B) Espectrograma Mel
    mel_image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[1]
    )

    axes[1].set_title("Espectrograma Mel")
    fig.colorbar(
        mel_image,
        ax=axes[1],
        format="%+2.0f dB"
    )

    # C) MFCC
    mfcc_visual = mfcc[1:, :]

    mfcc_image = librosa.display.specshow(
        mfcc_visual,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis=None,
        cmap="coolwarm",
        ax=axes[2]
    )

    axes[2].set_yticks(range(12))
    axes[2].set_yticklabels(range(1, 13))
    axes[2].set_title("Coeficientes MFCC")
    axes[2].set_ylabel("Coeficiente")
    axes[2].set_xlabel("Tempo (segundos)")

    fig.colorbar(
        mfcc_image,
        ax=axes[2]
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Salvar figura
    output_file = OUTPUT_DIR / f"{category}_analysis.png"

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"  Duração: {duration:.2f} segundos")
    print(f"  Taxa de amostragem: {sample_rate} Hz")
    print(f"  MFCCs calculados: {mfcc.shape[0]}")
    print(f"  Imagem salva: {output_file}")


# --------------------------------------------------
# 5. FINALIZAÇÃO
# --------------------------------------------------

print("\n========== PROCESSAMENTO CONCLUÍDO ==========")
print(f"Total de categorias visualizadas: {len(selected_files)}")
print(f"Arquivos salvos em: {OUTPUT_DIR}")
