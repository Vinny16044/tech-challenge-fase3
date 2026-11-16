
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "raw"
EVENTS_FILE = BASE_DIR / "outputs" / "detected_audio_events.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "event_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

events_df = pd.read_csv(EVENTS_FILE)

categories = [
    "breathing-deep",
    "breathing-shallow",
    "cough-heavy",
    "counting-normal",
    "vowel-a",
]

# Selecionar um participante com todas as categorias
participant_id = "0zexHIcM7tQDdnFiEj2Eb0v3g212"

for category in categories:

    file_name = f"{participant_id}_{category}.wav"
    audio_path = AUDIO_DIR / file_name

    if not audio_path.exists():
        print(f"Arquivo não encontrado: {file_name}")
        continue

    signal, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    file_events = events_df[
        events_df["file_name"] == file_name
    ].sort_values("start_seconds")

    fig, ax = plt.subplots(figsize=(13, 4))

    librosa.display.waveshow(
        signal,
        sr=sr,
        ax=ax
    )

    for _, event in file_events.iterrows():

        start = event["start_seconds"]
        end = event["end_seconds"]
        number = int(event["event_number"])

        ax.axvspan(
            start,
            end,
            color="orange",
            alpha=0.25
        )

        ax.axvline(
            start,
            color="red",
            linestyle="--",
            alpha=0.7
        )

        ax.text(
            (start + end) / 2,
            1.02,
            str(number),
            ha="center",
            transform=ax.get_xaxis_transform()
        )

    ax.set_title(
        f"{category} — {len(file_events)} eventos detectados"
    )

    ax.set_xlabel("Tempo (segundos)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{category}_events.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Salvo: {output_path}")

print("\nVisualizações concluídas!")
