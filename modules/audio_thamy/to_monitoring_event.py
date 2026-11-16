"""
Converte um resultado `audio_analysis` 1.0.0 (módulo da Thamy) em um evento do
contrato de monitoramento v1 da Fase 4 (`src.contracts.MonitoringEventV1`).

O mapeamento é deliberadamente conservador: a saída da Thamy indica palavras
encontradas na transcrição, não um achado clínico. Por isso o evento é
`audio_check`, sem anomalia e com prioridade `low`; os termos sinalizados ficam
no `finding` para a revisão humana. O evento só leva os termos, a referência à
gravação e os metadados do contrato — transcrição, sentimento e atributos
acústicos continuam no resultado de origem.

Execute a partir da raiz da Fase 4:
    py -m modules.audio_thamy.to_monitoring_event --input <resultado.json> \
        --patient-id P0001 --recording-id consulta-simulada-01 \
        --recorded-at 2026-11-12T10:03:20-03:00 --output output/thamy/P0001_audio.json
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import suppress
from datetime import datetime
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

SOURCE_MODULE = "audio_analysis"
SOURCE_VERSION = "1.0.0"
RECORDING_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}"
PATIENT_ID_PATTERN = r"P\d{4,}"
MAX_FINDING = 500


def load_object(path: Path, description: str) -> dict:
    with path.open(encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{description} deve ser um objeto JSON.")
    return value


def flagged_terms(result: dict) -> list[str]:
    """Termos dos alertas keyword_review que pedem revisão humana, sem repetição."""
    terms: list[str] = []
    for alert in result.get("alerts") or []:
        if alert.get("type") == "keyword_review" and alert.get("requires_human_review") is True:
            for term in alert.get("detected_terms") or []:
                if isinstance(term, str) and term.strip() and term.strip() not in terms:
                    terms.append(term.strip())
    return terms


def convert_audio_result(
    result: dict,
    *,
    patient_id: str,
    recording_id: str,
    recorded_at: str,
) -> dict:
    """Monta o evento v1. Mesma gravação (ID, horário e termos) => mesmo evento."""
    if (
        result.get("module") != SOURCE_MODULE
        or result.get("schema_version") != SOURCE_VERSION
        or result.get("status") != "success"
    ):
        raise ValueError(
            f"É necessário um resultado {SOURCE_MODULE} {SOURCE_VERSION} com status success."
        )
    if not re.fullmatch(PATIENT_ID_PATTERN, patient_id):
        raise ValueError("Use um ID de paciente fictício no formato P0001.")
    if not re.fullmatch(RECORDING_ID_PATTERN, recording_id):
        raise ValueError("Use até 40 caracteres ASCII no ID da gravação, incluindo '-' ou '_'.")
    timestamp = datetime.fromisoformat(recorded_at)
    if timestamp.utcoffset() is None:
        raise ValueError("O horário da gravação precisa incluir o fuso, como -03:00 ou Z.")

    metadata = result["audio_metadata"]
    duration = metadata["duration_seconds"]
    if type(duration) not in (int, float) or not isfinite(duration) or duration <= 0:
        raise ValueError("A duração do áudio deve ser um número finito maior que zero.")
    file_name = metadata.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("O resultado precisa informar audio_metadata.file_name.")

    terms = flagged_terms(result)
    if terms:
        finding = (
            "Termos sinalizados na transcrição para revisão humana: "
            + ", ".join(terms)
            + ". Busca por palavras-chave, sem validação clínica."
        )
        evidence_note = (
            "Gravação completa: o resultado não informa o instante de cada termo."
        )
    else:
        finding = "Áudio processado. A busca por palavras-chave não sinalizou termos para revisão."
        evidence_note = "Gravação completa analisada."
    if len(finding) > MAX_FINDING:
        finding = finding[: MAX_FINDING - 3] + "..."

    providers = [
        provider
        for provider in (
            (result.get("transcription") or {}).get("provider"),
            (result.get("text_analysis") or {}).get("provider"),
        )
        if isinstance(provider, str) and provider
    ]
    source = {"system": SOURCE_MODULE, "version": SOURCE_VERSION}
    if providers:
        source["model"] = " + ".join(providers)[:80]

    return {
        "schema_version": "1.0",
        "event_id": f"audio-{patient_id}-{recording_id}",
        "patient_id": patient_id,
        "module": "audio",
        "timestamp": timestamp.isoformat(),
        "event_type": "audio_check",
        "finding": finding,
        "anomaly_detected": False,
        "priority": "low",
        "evidence": [
            {
                "type": "audio_segment",
                "uri": f"audio://thamy/{recording_id}/{quote(file_name.strip())}",
                "start_s": 0.0,
                "end_s": float(duration),
                "description": evidence_note,
            }
        ],
        "source": source,
    }


def validate_event(event: dict, schema_path: Path) -> None:
    """Valida pelo modelo Pydantic e pelo JSON Schema do projeto (os mesmos do fluxo)."""
    from jsonschema import Draft202012Validator, FormatChecker

    from src.contracts import MonitoringEventV1

    MonitoringEventV1.model_validate(event)
    schema = load_object(schema_path, "JSON Schema de monitoramento")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    if error := next(validator.iter_errors(event), None):
        raise ValueError(f"JSON Schema: {error.json_path}: {error.message}")


def save_event(event: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=".audio-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(payload)
        temporary.replace(output)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte audio_analysis 1.0.0 em evento v1.")
    parser.add_argument("--input", type=Path, required=True, help="JSON de sucesso do módulo de áudio.")
    parser.add_argument("--schema", type=Path, default=Path("schemas/event_v1.schema.json"))
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--recording-id", required=True, help="Identificador estável da gravação.")
    parser.add_argument("--recorded-at", required=True, help="Horário ISO 8601 da gravação, com fuso.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.output.resolve() in {args.input.resolve(), args.schema.resolve()}:
            raise ValueError("Escolha um arquivo de saída diferente dos arquivos de entrada.")
        result = load_object(args.input, "Resultado de áudio")
        event = convert_audio_result(
            result,
            patient_id=args.patient_id,
            recording_id=args.recording_id,
            recorded_at=args.recorded_at,
        )
        validate_event(event, args.schema)
        save_event(event, args.output)
    except ImportError as exc:
        parser.exit(2, f"Execute a partir da raiz da Fase 4 com as dependências instaladas: {exc}\n")
    except (OSError, ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        # pydantic.ValidationError herda de ValueError.
        parser.exit(2, f"Não foi possível converter o áudio: {exc}\n")

    print(f"Evento {event['event_id']} validado e salvo em {args.output}.")


if __name__ == "__main__":
    main()
