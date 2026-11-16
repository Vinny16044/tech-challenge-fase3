"""
Contrato de eventos v1 (versão 1.1) — Tech Challenge Fase 4.

Os módulos de áudio, vídeo e dados clínicos publicam seus achados como
eventos JSON neste formato. O arquivo `schemas/event_v1.schema.json` é gerado
a partir deste modelo (`py -m src.contracts.generate_schema`); por isso, este
módulo é a referência do contrato.

A versão 1.1 acrescenta, para o módulo clínico, os tipos de evento
`prescription_change` e `patient_transfer` e a evidência `prescription`.
Eventos "1.0" continuam válidos.

Responsável: Paola (arquitetura e integração)
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.1"
# Versões aceitas na entrada: a 1.1 só acrescenta valores, sem mudar campos.
ACCEPTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1")

Module = Literal["audio", "video", "clinical"]
Priority = Literal["low", "medium", "high", "critical"]
EvidenceType = Literal[
    "audio_segment",
    "video_clip",
    "video_frame",
    "clinical_measurement",
    "lab_result",
    "prescription",
]

MODULES: tuple[str, ...] = ("audio", "video", "clinical")

# Ordem usada pelas regras de alerta (maior índice = mais urgente).
PRIORITY_ORDER: tuple[str, ...] = ("low", "medium", "high", "critical")

# Tipos de evento aceitos por módulo (v1). Novos tipos exigem acordo entre
# os módulos e incremento de versão menor do contrato (ex.: 1.1).
EVENT_TYPES_BY_MODULE: dict[str, tuple[str, ...]] = {
    "audio": (
        "vocal_distress",  # padrão vocal compatível com desconforto ou dor
        "call_for_help",  # pedido verbal de ajuda
        "respiratory_sound",  # som respiratório fora do padrão
        # Verificação sem anomalia; também registra a busca por palavras-chave
        # do módulo de áudio, que não é um achado clínico.
        "audio_check",
    ),
    "video": (
        "fall_detected",
        "bed_exit_attempt",  # saída do leito sem assistência
        "prolonged_immobility",  # imobilidade acima do esperado
        "agitation",
        "video_check",
    ),
    "clinical": (
        "vital_signs",  # FC, SpO2, PA, temperatura
        "lab_result",
        "prescription_change",  # 1.1: alteração inesperada na prescrição
        "patient_transfer",  # 1.1: movimentação atípica entre unidades
        "clinical_check",
    ),
}

EVIDENCE_TYPES_BY_MODULE: dict[str, tuple[str, ...]] = {
    "audio": ("audio_segment",),
    "video": ("video_clip", "video_frame"),
    "clinical": ("clinical_measurement", "lab_result", "prescription"),
}

PATIENT_ID_PATTERN = r"^P\d{4,}$"
EVENT_ID_PATTERN = r"^(audio|video|clinical)-[A-Za-z0-9][A-Za-z0-9._:-]{2,119}$"
URI_PATTERN = r"^[a-z][a-z0-9+.-]*://\S+$"


class EvidenceRef(BaseModel):
    """
    Referência a uma evidência mantida pelo módulo de origem.

    O evento NÃO carrega mídia nem prontuário: ele aponta (via `uri`) para o
    artefato guardado pelo módulo, e informa o trecho/medida relevante.
    """

    model_config = ConfigDict(extra="forbid")

    type: EvidenceType = Field(
        description="Tipo da evidência; deve ser compatível com o módulo."
    )
    uri: str = Field(
        pattern=URI_PATTERN,
        max_length=500,
        description=(
            "Referência estável ao artefato no módulo de origem, com esquema "
            "(ex.: audio://leito-12/rec0007.wav, video://cam-12/clip0315.mp4, "
            "clinical://sinais-vitais/obs-5521)."
        ),
        examples=["audio://leito-12/rec0007.wav"],
    )
    description: str | None = Field(
        default=None,
        max_length=300,
        description="Descrição curta do que a evidência mostra.",
    )
    start_s: float | None = Field(
        default=None,
        ge=0,
        description="Início do trecho em segundos (áudio/vídeo).",
    )
    end_s: float | None = Field(
        default=None,
        ge=0,
        description="Fim do trecho em segundos (áudio/vídeo).",
    )
    frame_index: int | None = Field(
        default=None,
        ge=0,
        description="Índice do quadro (video_frame).",
    )
    metric: str | None = Field(
        default=None,
        max_length=60,
        description=(
            "Nome da medida (clinical_measurement/lab_result) ou do medicamento "
            "(prescription). Ex.: heart_rate."
        ),
    )
    value: float | None = Field(
        default=None,
        description="Valor medido (clinical_measurement/lab_result).",
    )
    unit: str | None = Field(
        default=None,
        max_length=20,
        description="Unidade do valor. Ex.: bpm, %, mg/dL.",
    )
    reference_range: str | None = Field(
        default=None,
        max_length=40,
        description="Faixa de referência usada pelo módulo. Ex.: 60-100.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Confiança do modelo do módulo para esta evidência (0 a 1).",
    )

    @model_validator(mode="after")
    def _check_required_by_type(self) -> "EvidenceRef":
        if self.type in ("audio_segment", "video_clip"):
            if self.start_s is None or self.end_s is None:
                raise ValueError(f"evidência {self.type} exige start_s e end_s")
            if self.end_s <= self.start_s:
                raise ValueError("end_s deve ser maior que start_s")
        elif self.type == "video_frame":
            if self.frame_index is None and self.start_s is None:
                raise ValueError("evidência video_frame exige frame_index ou start_s")
        else:  # clinical_measurement / lab_result / prescription
            missing = [
                name
                for name in ("metric", "value", "unit")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"evidência {self.type} exige {', '.join(missing)}"
                )
        return self


class EventSource(BaseModel):
    """Identifica o sistema que produziu o evento."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(
        min_length=2,
        max_length=60,
        description="Nome do sistema/módulo emissor. Ex.: fase4-audio-analysis.",
    )
    version: str = Field(
        min_length=1,
        max_length=30,
        description="Versão do sistema emissor. Ex.: 0.1.0.",
    )
    model: str | None = Field(
        default=None,
        max_length=80,
        description="Modelo de IA usado na análise, se houver.",
    )


class MonitoringEventV1(BaseModel):
    """Evento de monitoramento publicado por um módulo (contrato v1)."""

    model_config = ConfigDict(
        extra="forbid",
        title="MonitoringEventV1",
        json_schema_extra={
            "$id": "https://fiap-tech-challenge/fase4/schemas/event_v1.schema.json",
            "description": (
                "Evento de monitoramento hospitalar (áudio, vídeo ou dados "
                "clínicos) — contrato v1 do protótipo da Fase 4. Os achados "
                "são para avaliação profissional e não constituem diagnóstico."
            ),
        },
    )

    schema_version: Literal["1.0", "1.1"] = Field(
        description="Versão do contrato: \"1.0\" ou \"1.1\"."
    )
    event_id: str = Field(
        pattern=EVENT_ID_PATTERN,
        description=(
            "ID estável do evento: '<module>-<id local estável>'. O reenvio do "
            "MESMO evento deve usar o MESMO event_id. Recomendado: "
            "'<module>-<patient_id>-<id local na origem>'."
        ),
        examples=["audio-P0001-rec0007-seg0042"],
    )
    patient_id: str = Field(
        pattern=PATIENT_ID_PATTERN,
        description="ID sintético do paciente (ex.: P0001). Nunca dados pessoais.",
    )
    module: Module = Field(description="Módulo emissor.")
    timestamp: AwareDatetime = Field(
        description=(
            "Momento do achado em ISO 8601 COM fuso horário "
            "(ex.: 2026-11-12T10:03:20-03:00)."
        )
    )
    event_type: str = Field(
        description="Tipo do achado; os valores aceitos dependem do módulo."
    )
    finding: str = Field(
        min_length=3,
        max_length=500,
        description="Descrição objetiva do achado (sem diagnóstico).",
    )
    anomaly_detected: bool = Field(
        description="true se o módulo considera o achado fora do padrão esperado."
    )
    priority: Priority = Field(
        description=(
            "Prioridade sugerida pelo módulo. Eventos sem anomalia devem usar 'low'. "
            "A prioridade final do alerta é calculada pelas regras do integrador."
        )
    )
    evidence: list[EvidenceRef] = Field(
        min_length=1,
        max_length=20,
        description="Referências às evidências que sustentam o achado.",
    )
    source: EventSource = Field(description="Sistema que produziu o evento.")

    @field_validator("event_type")
    @classmethod
    def _event_type_format(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,59}", value):
            raise ValueError("event_type deve ser snake_case")
        return value

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "MonitoringEventV1":
        allowed_types = EVENT_TYPES_BY_MODULE[self.module]
        if self.event_type not in allowed_types:
            raise ValueError(
                f"event_type '{self.event_type}' não é aceito para o módulo "
                f"'{self.module}'. Aceitos: {', '.join(allowed_types)}"
            )

        if not self.event_id.startswith(f"{self.module}-"):
            raise ValueError(
                f"event_id deve começar com o prefixo do módulo '{self.module}-'"
            )

        allowed_evidence = EVIDENCE_TYPES_BY_MODULE[self.module]
        for item in self.evidence:
            if item.type not in allowed_evidence:
                raise ValueError(
                    f"evidência '{item.type}' não é aceita para o módulo "
                    f"'{self.module}'. Aceitas: {', '.join(allowed_evidence)}"
                )

        if not self.anomaly_detected and self.priority != "low":
            raise ValueError("eventos sem anomalia devem usar priority 'low'")

        return self

    def content_hash(self) -> str:
        """Hash canônico do conteúdo; diferencia reenvio idêntico de conflito."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_event_id(module: str, patient_id: str, local_ref: str) -> str:
    """
    Gera um event_id estável para módulos que não têm um ID local legível.

    `local_ref` deve ser algo que não muda entre reenvios (ex.: caminho do
    arquivo + offset do segmento). Mesma entrada => mesmo ID.
    """
    if module not in MODULES:
        raise ValueError(f"módulo inválido: {module}")
    digest = hashlib.sha256(local_ref.encode("utf-8")).hexdigest()[:12]
    return f"{module}-{patient_id}-{digest}"


def json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) gerado a partir do modelo Pydantic."""
    schema = MonitoringEventV1.model_json_schema()
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}

    # Regras cruzadas por módulo expressas em JSON Schema (if/then), para que
    # quem valida só com JSON Schema obtenha o mesmo resultado do modelo Python.
    conditional_rules: list[dict[str, Any]] = []
    for module in MODULES:
        conditional_rules.append(
            {
                "if": {
                    "properties": {"module": {"const": module}},
                    "required": ["module"],
                },
                "then": {
                    "properties": {
                        "event_id": {"pattern": f"^{module}-"},
                        "event_type": {"enum": list(EVENT_TYPES_BY_MODULE[module])},
                        "evidence": {
                            "items": {
                                "properties": {
                                    "type": {
                                        "enum": list(EVIDENCE_TYPES_BY_MODULE[module])
                                    }
                                }
                            }
                        },
                    }
                },
            }
        )
    conditional_rules.append(
        {
            "if": {
                "properties": {"anomaly_detected": {"const": False}},
                "required": ["anomaly_detected"],
            },
            "then": {"properties": {"priority": {"const": "low"}}},
        }
    )
    schema["allOf"] = conditional_rules

    # Extensões x-*: as mesmas listas, legíveis para quem consulta só o schema.
    schema["x-event-types-by-module"] = {
        module: list(values) for module, values in EVENT_TYPES_BY_MODULE.items()
    }
    schema["x-evidence-types-by-module"] = {
        module: list(values) for module, values in EVIDENCE_TYPES_BY_MODULE.items()
    }
    return schema
