
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_LENGTH = 1024
HOP_LENGTH = 256

# Parâmetros experimentais de segmentação
THRESHOLD_RATIO = 0.15
MIN_EVENT_SECONDS = 0.10
MERGE_GAP_SECONDS = 0.15


def detect_events(signal, sample_rate):
    """Detecta regiões de atividade acústica usando energia RMS."""

    rms = librosa.feature.rms(
        y=signal,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=False
    )[0]

    if len(rms) == 0 or np.max(rms) <= 0:
        return []

    threshold = np.max(rms) * THRESHOLD_RATIO
    active = rms >= threshold

    # Identificar sequências de frames ativos
    padded = np.pad(
        active.astype(int),
        (1, 1),
        constant_values=0
    )

    changes = np.diff(padded)

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # Converter frames para intervalos em segundos
    intervals = []

    for start, end in zip(starts, ends):
        start_sample = start * HOP_LENGTH
        end_sample = min(
            (end - 1) * HOP_LENGTH + FRAME_LENGTH,
            len(signal)
        )

        intervals.append([
            start_sample / sample_rate,
            end_sample / sample_rate
        ])

    # Unir intervalos separados por pausas muito curtas
    merged = []

    for start, end in intervals:
        if merged and start - merged[-1][1] <= MERGE_GAP_SECONDS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Descartar eventos muito curtos
    return [
        (start, end)
        for start, end in merged
        if end - start >= MIN_EVENT_SECONDS
    ]


results = []

audio_files = sorted(AUDIO_DIR.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(
        f"Nenhum WAV encontrado em {AUDIO_DIR}"
    )

print(f"\nAnalisando {len(audio_files)} arquivos...\n")

for audio_path in audio_files:
    signal, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    events = detect_events(signal, sr)

    for event_number, (start, end) in enumerate(events, start=1):
        segment = signal[
            int(start * sr):int(end * sr)
        ]

        segment_rms = float(
            np.sqrt(np.mean(segment ** 2))
        ) if len(segment) else 0.0

        results.append({
            "file_name": audio_path.name,
            "event_number": event_number,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "rms": round(segment_rms, 6)
        })

    print(
        f"{audio_path.name}: "
        f"{len(events)} eventos detectados"
    )


columns = [
    "file_name",
    "event_number",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "rms"
]

df = pd.DataFrame(results, columns=columns)

output_file = OUTPUT_DIR / "detected_audio_events.csv"
df.to_csv(output_file, index=False)

print("\nDetecção concluída!")
print(f"Total de eventos: {len(df)}")
print(f"Arquivo salvo: {output_file}")
