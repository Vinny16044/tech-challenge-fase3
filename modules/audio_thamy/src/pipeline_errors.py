
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def build_error_result(error, stage, audio_path):
    """Cria um resultado padronizado para falhas do pipeline."""

    return {
        "module": "audio_analysis",
        "schema_version": "1.0.0",
        "analysis_id": str(uuid4()),
        "status": "error",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "audio_metadata": {
            "file_name": Path(audio_path).name
        },
        "transcription": None,
        "text_analysis": None,
        "acoustic_features": None,
        "alerts": [],
        "error": {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error)
        },
        "limitations": {
            "clinical_diagnosis": False,
            "transcription_requires_review": True
        }
    }


def save_error_result(error, stage, audio_path, output_path):
    """Salva o erro em JSON separado do resultado de sucesso."""

    result = build_error_result(error, stage, audio_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4,
            allow_nan=False
        )

    return result
