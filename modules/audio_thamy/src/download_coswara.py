from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DATASET_ID = "szzs1693/coswara-data"
CONFIG_NAME = "audio"

# Tipos escolhidos para cobrir sons respiratórios e amostras de fala.
SELECTED_TYPES = [
    "breathing-deep",
    "breathing-shallow",
    "cough-heavy",
    "vowel-a",
    "counting-normal",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa uma amostra pequena do dataset Coswara."
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=3,
        help="Quantidade máxima por tipo de áudio (padrão: 3).",
    )
    args = parser.parse_args()

    if args.per_type < 1:
        raise SystemExit("--per-type precisa ser pelo menos 1.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando {DATASET_ID} (configuração: {CONFIG_NAME})...")
    print("O download é feito em streaming; apenas uma pequena amostra será salva.")

    dataset = load_dataset(
        DATASET_ID,
        CONFIG_NAME,
        split="train",
        streaming=True,
    )

    counts = {audio_type: 0 for audio_type in SELECTED_TYPES}
    metadata_rows: list[dict] = []
    saved = 0

    for row in dataset:
        audio_type = row.get("audio_type")

        if audio_type not in counts or counts[audio_type] >= args.per_type:
            continue

        # Exclui registros marcados como qualidade ruim (0), quando esse campo existe.
        quality_score = row.get("quality_score")
        if quality_score is not None and int(quality_score) == 0:
            continue

        audio = row.get("audio")
        if not isinstance(audio, dict) or audio.get("array") is None:
            print(f"Aviso: registro sem áudio decodificado; ignorado ({audio_type}).")
            continue

        participant = safe_name(row.get("participant_id", f"sample_{saved + 1}"))
        file_name = f"{participant}_{audio_type}.wav"
        output_path = RAW_DIR / file_name

        samples = audio["array"]
        sample_rate = int(audio.get("sampling_rate", 16000))
        sf.write(output_path, samples, sample_rate)

        metadata_rows.append({
            "file_name": file_name,
            "participant_id": participant,
            "audio_type": audio_type,
            "quality_score": quality_score,
            "record_date": row.get("record_date"),
            "covid_status": row.get("covid_status"),
            "sample_rate": sample_rate,
        })

        counts[audio_type] += 1
        saved += 1
        print(f"Salvo: {file_name}")

        if all(value >= args.per_type for value in counts.values()):
            break

    csv_path = PROCESSED_DIR / "coswara_sample_metadata.csv"
    if metadata_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

    print("\nResumo do download")
    print(f"Total de arquivos salvos: {saved}")
    for audio_type, quantity in counts.items():
        print(f"- {audio_type}: {quantity}")
    print(f"Metadados: {csv_path}")

    if saved == 0:
        print(
            "\nNenhum áudio foi salvo. Confira a conexão, as dependências e "
            "a disponibilidade do dataset."
        )


if __name__ == "__main__":
    main()
