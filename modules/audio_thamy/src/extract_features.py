
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# Diretórios do projeto
BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_features(audio_path):
    """
    Extrai características acústicas de um arquivo WAV.
    Retorna um dicionário com as métricas calculadas.
    """

    # Carrega o áudio mantendo a taxa de amostragem original
    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Arquivo de áudio vazio.")

    # Características gerais
    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # Energia do sinal
    rms = librosa.feature.rms(y=audio)[0]

    # Cruzamentos pelo zero
    zero_crossing = librosa.feature.zero_crossing_rate(audio)[0]

    # Características espectrais
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )[0]

    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio
    )[0]

    # MFCC: 13 coeficientes acústicos
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    # Identificação da amostra
    file_name = audio_path.name
    participant_id = audio_path.stem.rsplit("_", 1)[0]
    audio_category = audio_path.stem.rsplit("_", 1)[-1]

    features = {
        "file_name": file_name,
        "participant_id": participant_id,
        "audio_category": audio_category,
        "duration_seconds": duration,
        "sample_rate_hz": sample_rate,

        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),

        "zero_crossing_mean": np.mean(zero_crossing),

        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_rolloff_mean": np.mean(spectral_rolloff),
        "spectral_flatness_mean": np.mean(spectral_flatness),
    }

    # Média e desvio padrão dos 13 MFCCs
    for index in range(13):
        features[f"mfcc_{index + 1}_mean"] = np.mean(mfcc[index])
        features[f"mfcc_{index + 1}_std"] = np.std(mfcc[index])

    return features


def main():
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        print("Nenhum arquivo WAV encontrado em data/raw.")
        return

    results = []

    print(f"Encontrados {len(audio_files)} arquivos de áudio.\n")

    for audio_path in audio_files:
        try:
            features = calculate_features(audio_path)
            results.append(features)

            print(f"Processado: {audio_path.name}")

        except Exception as error:
            print(f"Erro ao processar {audio_path.name}: {error}")

    if not results:
        print("Nenhum áudio foi processado com sucesso.")
        return

    dataframe = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "audio_features.csv"
    dataframe.to_csv(output_file, index=False)

    print("\nExtração concluída!")
    print(f"Áudios processados: {len(results)}")
    print(f"Total de características: {len(dataframe.columns)}")
    print(f"Arquivo salvo em: {output_file}")


if __name__ == "__main__":
    main()
