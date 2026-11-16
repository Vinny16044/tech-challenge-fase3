from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(RAW_DIR.glob("*.wav"))

    if not audio_files:
        print("Nenhum WAV encontrado em data/raw/. Execute primeiro:")
        print("python src/download_coswara.py")
        return

    rows = []
    for audio_path in audio_files:
        info = sf.info(audio_path)
        rows.append({
            "file_name": audio_path.name,
            "duration_seconds": round(info.duration, 3),
            "sample_rate_hz": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
        })

    inventory = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "audio_inventory.csv"
    inventory.to_csv(output_path, index=False, encoding="utf-8")

    print(inventory.to_string(index=False))
    print(f"\nInventário salvo em: {output_path}")


if __name__ == "__main__":
    main()
