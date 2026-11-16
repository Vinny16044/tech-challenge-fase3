from pathlib import Path
import json
import os
import sys
import tempfile
import librosa
import numpy as np

from datetime import datetime, timezone
from uuid import uuid4
from dotenv import load_dotenv

from src.audio_validation import validate_audio
from src.azure_speech_service import transcribe_audio
from src.azure_language_service import analyze_text
from src.pipeline_errors import save_error_result


BASE_DIR = Path(__file__).resolve().parents[1]

AUDIO_FILE = BASE_DIR / "data" / "raw" / "consulta_simulada.wav"
OUTPUT_FILE = BASE_DIR / "outputs" / "audio_pipeline_result.json"
ERROR_FILE = BASE_DIR / "outputs" / "audio_pipeline_error.json"


# ANÁLISE ACÚSTICA
def extract_acoustic_features(audio_path):
    print("\nAnalisando características acústicas...")

    signal, sample_rate = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    if signal.size == 0:
        raise ValueError("O arquivo de áudio está vazio.")

    duration = librosa.get_duration(
        y=signal,
        sr=sample_rate
    )

    rms = librosa.feature.rms(y=signal)[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=signal
    )[0]

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=13
    )

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate
    )[0]

    # Detecção de atividade sonora por RMS
    frame_length = 1024
    hop_length = 256

    event_rms = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    threshold = 0.15 * float(np.max(event_rms))
    active = event_rms > threshold

    transitions = np.diff(
        np.concatenate(
            ([False], active, [False])
        ).astype(int)
    )

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    intervals = []

    for start, end in zip(starts, ends):
        start_time = max(
            0.0,
            (start * hop_length - frame_length / 2)
            / sample_rate
        )

        end_time = min(
            duration,
            (end * hop_length - frame_length / 2)
            / sample_rate
        )

        if end_time - start_time >= 0.10:
            intervals.append([start_time, end_time])

    # Agrupa intervalos separados por pausas curtas
    merged = []

    for start, end in intervals:
        if merged and start - merged[-1][1] <= 0.15:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    events = [
        {
            "start_seconds": round(float(start), 3),
            "end_seconds": round(float(end), 3),
            "duration_seconds": round(
                float(end - start), 3
            )
        }
        for start, end in merged
    ]

    return {
        "duration_seconds": round(float(duration), 3),
        "sample_rate_hz": int(sample_rate),
        "rms_mean": round(float(np.mean(rms)), 6),
        "zcr_mean": round(float(np.mean(zcr)), 6),
        "spectral_centroid_mean_hz": round(
            float(np.mean(centroid)), 3
        ),
        "mfcc_mean": [
            round(float(value), 5)
            for value in np.mean(mfcc, axis=1)
        ],
        "mfcc_std": [
            round(float(value), 5)
            for value in np.std(mfcc, axis=1)
        ],
        "detected_event_count": len(events),
        "detected_events": events,
        "event_detection_method": "rms_threshold",
        "event_interpretation": (
            "sound_activity_not_clinical_events"
        )
    }


def save_success_result(output, output_path):
    """
    Salva o resultado de sucesso de forma atômica.
    Evita deixar um JSON parcialmente escrito.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=".audio_pipeline_",
            dir=output_path.parent,
            delete=False
        ) as file:
            temporary_path = Path(file.name)

            json.dump(
                output,
                file,
                ensure_ascii=False,
                indent=4,
                allow_nan=False
            )

        os.replace(temporary_path, output_path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def run_pipeline(
    audio_path=None,
    output_path=None,
    error_path=None
):
    """
    Executa o pipeline de áudio.

    Em caso de sucesso, retorna o JSON consolidado.
    Em caso de falha, registra um JSON de erro e
    propaga a exceção para o código chamador.
    """

    audio_path = Path(
        audio_path if audio_path is not None
        else AUDIO_FILE
    )

    output_path = Path(
        output_path if output_path is not None
        else OUTPUT_FILE
    )

    error_path = Path(
        error_path if error_path is not None
        else ERROR_FILE
    )

    stage = "configuration"

    try:
        # CONFIGURAÇÃO
        load_dotenv(BASE_DIR / ".env")

        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")

        language_key = os.getenv("AZURE_LANGUAGE_KEY")
        language_endpoint = os.getenv(
            "AZURE_LANGUAGE_ENDPOINT"
        )

        if not all([
            speech_key,
            speech_region,
            language_key,
            language_endpoint
        ]):
            raise ValueError(
                "Credenciais Azure incompletas no .env"
            )

        # 0. VALIDAÇÃO DO ÁUDIO
        stage = "audio_validation"

        print("0. Validando arquivo de áudio...")

        signal, sample_rate = validate_audio(
            audio_path
        )

        print("Arquivo validado com sucesso!")
        print(f"Amostras: {len(signal)}")
        print(
            f"Taxa de amostragem: {sample_rate} Hz"
        )

        # 1. TRANSCRIÇÃO
        stage = "transcription"

        print("\n1. Transcrevendo áudio...")

        transcription = transcribe_audio(
            audio_path=audio_path,
            speech_key=speech_key,
            speech_region=speech_region,
            timeout=180
        )

        print("Transcrição:", transcription)

        # 2. ANÁLISE TEXTUAL
        stage = "text_analysis"

        print("\n2. Analisando texto...")

        text_analysis = analyze_text(
            transcription=transcription,
            language_key=language_key,
            language_endpoint=language_endpoint
        )

        print("Análise textual concluída!")
        print(
            "Sentimento:",
            text_analysis["sentiment"]["label"]
        )
        print(
            "Frases-chave:",
            text_analysis["key_phrases"]
        )
        print(
            "Entidades:",
            len(text_analysis["entities"])
        )

        # 3. INDICADORES DEMONSTRATIVOS
        stage = "indicators"

        print("\n3. Gerando indicadores...")

        clinical_terms = [
            "cansaço",
            "falta de ar",
            "tosse persistente",
            "dificuldade para respirar"
        ]

        detected_terms = [
            term
            for term in clinical_terms
            if term in transcription.lower()
        ]

        # 4. CARACTERÍSTICAS ACÚSTICAS
        stage = "acoustic_features"

        print(
            "\n4. Extraindo características acústicas..."
        )

        acoustic_features = extract_acoustic_features(
            audio_path
        )

        print(
            "Duração:",
            acoustic_features["duration_seconds"]
        )
        print(
            "RMS médio:",
            acoustic_features["rms_mean"]
        )
        print(
            "ZCR médio:",
            acoustic_features["zcr_mean"]
        )
        print(
            "Eventos de atividade sonora:",
            acoustic_features["detected_event_count"]
        )

        # 5. JSON CONSOLIDADO
        stage = "json_output"

        print("\n5. Gerando Resultado...")

        output = {
            "module": "audio_analysis",
            "schema_version": "1.0.0",
            "analysis_id": str(uuid4()),
            "status": "success",
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "audio_metadata": {
                "file_name": audio_path.name,
                "duration_seconds": acoustic_features[
                    "duration_seconds"
                ],
                "sample_rate_hz": acoustic_features[
                    "sample_rate_hz"
                ],
                "language": "pt-BR",
                "source": "synthetic_consultation"
            },

            "transcription": {
                "text": transcription,
                "language": "pt-BR",
                "provider": "azure_speech",
                "recognition_mode": "continuous"
            },

            "text_analysis": text_analysis,

            "acoustic_features": acoustic_features,

            "alerts": [
                {
                    "type": "keyword_review",
                    "status": "review_required",
                    "detected_terms": detected_terms,
                    "requires_human_review": True,
                    "description": (
                        "Termos predefinidos identificados "
                        "na transcrição. Revisão humana "
                        "recomendada."
                    ),
                    "clinical_validation": False
                }
            ] if detected_terms else [],

            "limitations": {
                "clinical_diagnosis": False,
                "sentiment_is_clinical_assessment": False,
                "acoustic_events_are_clinically_validated": False,
                "transcription_requires_review": True
            }
        }

        save_success_result(
            output=output,
            output_path=output_path
        )

        # Remove o registro de falha anterior
        # somente após um novo sucesso.
        if error_path.exists():
            error_path.unlink()

        print("\nPipeline concluído com sucesso!")
        print(
            "ID da análise:",
            output["analysis_id"]
        )
        print("Módulo:", output["module"])
        print("Versão:", output["schema_version"])
        print("Status:", output["status"])
        print("Alertas:", len(output["alerts"]))
        print("Resultado salvo em:", output_path)

        return output

    except Exception as error:
        print(
            f"\nErro durante a etapa: {stage}",
            file=sys.stderr
        )
        print(
            f"Detalhes: {error}",
            file=sys.stderr
        )

        # Impede que um resultado anterior de sucesso
        # seja confundido com o resultado atual.
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as cleanup_error:
                print(
                    "Aviso: não foi possível remover "
                    f"o resultado anterior: {cleanup_error}",
                    file=sys.stderr
                )

        try:
            save_error_result(
                error=error,
                stage=stage,
                audio_path=audio_path,
                output_path=error_path
            )

            print(
                f"Erro registrado em: {error_path}",
                file=sys.stderr
            )

        except Exception as logging_error:
            print(
                "Falha ao registrar o erro: "
                f"{logging_error}",
                file=sys.stderr
            )

        raise


def main():
    try:
        run_pipeline()
        return 0

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
