# Contrato de eventos v1 — Fase 4

Fonte da verdade: `src/contracts/event_v1.py` (modelo Pydantic).
JSON Schema gerado: `schemas/event_v1.schema.json` (`py -m src.contracts.generate_schema`).
Exemplos de referência (fictícios): `data/fase4/P0001_audio.json`, `P0001_video.json`, `P0001_clinical.json`.

Cada módulo publica **um objeto JSON por evento** (ou uma lista de eventos no mesmo arquivo).
Campos extras são rejeitados (`additionalProperties: false`), para que erros de digitação
apareçam já na validação.

## Campos (todos obrigatórios)

| Campo | Tipo | Valores aceitos / regra |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `event_id` | string | `<module>-<id local estável>`, regex `^(audio\|video\|clinical)-[A-Za-z0-9][A-Za-z0-9._:-]{2,119}$`. O prefixo deve ser igual a `module`. |
| `patient_id` | string | ID sintético, regex `^P\d{4,}$` (ex.: `P0001`). Nunca nome ou documento. |
| `module` | string | `audio`, `video`, `clinical` |
| `timestamp` | string | ISO 8601 **com fuso** (ex.: `2026-11-12T10:03:20-03:00`). Sem fuso = rejeitado. |
| `event_type` | string | Depende do módulo (tabela abaixo). |
| `finding` | string | 3–500 caracteres. Descrição objetiva do achado, **sem diagnóstico**. |
| `anomaly_detected` | boolean | `true` se o módulo considera o achado fora do padrão. |
| `priority` | string | `low`, `medium`, `high`, `critical`. Se `anomaly_detected = false`, deve ser `low`. É a prioridade *sugerida*; a do alerta é calculada pelo integrador. |
| `evidence` | lista | 1–20 referências (abaixo). |
| `source` | objeto | `system` (obrig.), `version` (obrig.), `model` (opcional). |

### `event_type` por módulo

| Módulo | Valores |
|---|---|
| `audio` | `vocal_distress`, `call_for_help`, `respiratory_sound`, `audio_check` |
| `video` | `fall_detected`, `bed_exit_attempt`, `prolonged_immobility`, `agitation`, `video_check` |
| `clinical` | `vital_signs`, `lab_result`, `clinical_check` |

Os tipos `*_check` são verificações de rotina, normalmente com `anomaly_detected = false`.
O resultado de busca por palavras-chave do módulo de áudio também entra como `audio_check`,
porque não é um achado clínico (ver `modules/audio_thamy/INTEGRAR_COM_FASE4.md`).
Novos tipos exigem acordo entre os módulos e nova versão menor do contrato (`1.1`).

## Como cada módulo referencia evidências

O evento **não carrega mídia nem prontuário**. Cada item de `evidence` aponta, via `uri`
(com esquema, ex.: `audio://`, `video://`, `clinical://`), para o artefato guardado pelo
módulo e diz qual trecho/medida sustenta o achado.

| Módulo | `type` aceito | Campos obrigatórios além de `type` e `uri` |
|---|---|---|
| áudio | `audio_segment` | `start_s`, `end_s` (segundos dentro do arquivo; `end_s > start_s`) |
| vídeo | `video_clip` | `start_s`, `end_s` |
| vídeo | `video_frame` | `frame_index` **ou** `start_s` |
| clínico | `clinical_measurement`, `lab_result` | `metric`, `value`, `unit` (recomendado: `reference_range`) |

Opcionais em qualquer evidência: `description` (até 300 caracteres) e `confidence` (0 a 1).

## IDs estáveis e reenvio

- O **reenvio do mesmo evento usa o mesmo `event_id`**. O integrador compara o conteúdo
  canônico (hash SHA-256 do JSON validado, independente da ordem das chaves):
  - mesmo `event_id` + mesmo conteúdo → **duplicado ignorado** (auditado como `evento_duplicado_ignorado`);
  - mesmo `event_id` + conteúdo diferente → **rejeitado** como `event_id_conflict`; vale o primeiro recebido.
- Recomendação: `<module>-<patient_id>-<id local na origem>` (ex.: `audio-P0001-rec0007-seg0042`).
- Sem ID local legível: `build_event_id(module, patient_id, local_ref)` gera
  `<module>-<patient_id>-<12 hex>` a partir de algo que não muda entre reenvios
  (ex.: arquivo + offset do segmento).

## Validação por quem produz o JSON

```powershell
py -c "import json,sys; from src.contracts import MonitoringEventV1; MonitoringEventV1.model_validate(json.load(open(sys.argv[1], encoding='utf-8-sig'))); print('ok')" data/fase4/P0001_audio.json
```

O JSON Schema traz as mesmas regras cruzadas por módulo em `allOf` (`if/then`), para quem
valida fora do Python.
