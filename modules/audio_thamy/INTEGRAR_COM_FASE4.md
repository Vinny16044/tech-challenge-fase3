# Integração do módulo de áudio com a Fase 4

A pasta `modules/audio_thamy/` reúne o módulo de áudio da Thamy (código, testes, documentação
e gravação simulada), o resultado da consulta simulada e o conversor para o contrato de
eventos v1 da Fase 4. O código do módulo fica isolado nesta pasta para não conflitar com o
`src/` do projeto.

## Conversão do resultado de áudio

O módulo produz um resultado `audio_analysis` 1.0.0. O conversor
`to_monitoring_event.py` transforma esse resultado em um evento aceito por
`src.contracts.MonitoringEventV1` e por `schemas/event_v1.schema.json`. A partir da raiz do
projeto, no PowerShell:

```powershell
py -m modules.audio_thamy.to_monitoring_event `
    --input modules/audio_thamy/docs/consulta_simulada_resultado.json `
    --patient-id P0001 `
    --recording-id consulta-simulada-01 `
    --recorded-at "2026-11-12T10:03:20-03:00" `
    --output output/thamy/P0001_audio.json
if ($LASTEXITCODE -ne 0) { throw "Falha na conversão do áudio." }
```

O evento é validado pelo modelo Pydantic e pelo JSON Schema antes de ser salvo.

| Campo | Valor gerado |
|---|---|
| `event_id` | `audio-<patient_id>-<recording-id>`; a mesma gravação gera sempre o mesmo ID |
| `event_type` | `audio_check` |
| `anomaly_detected` / `priority` | `false` / `low` |
| `finding` | termos dos alertas `keyword_review` com `requires_human_review: true` |
| `evidence` | `audio_segment` com `uri` `audio://thamy/<recording-id>/<audio_metadata.file_name>`, de 0 até a duração da gravação |
| `source` | `audio_analysis` 1.0.0 e os provedores da transcrição e da análise de texto |

**Por que `audio_check`:** o resultado do módulo indica palavras encontradas na transcrição,
sem validação clínica. Por isso o evento entra sem anomalia e com prioridade `low`. Os termos
ficam visíveis no relatório para a revisão humana, mas não acionam as regras de corroboração
(R3 e R4). A evidência cobre a gravação inteira, porque o resultado não informa o instante de
cada termo.

O evento leva apenas os termos sinalizados, a referência à gravação e os metadados exigidos
pelo contrato. A transcrição, a análise de texto e os atributos acústicos continuam no
resultado de origem.

O horário `2026-11-12T10:03:20-03:00` é fictício e foi escolhido para coincidir com a janela
dos exemplos de vídeo e dados clínicos. Em outras gravações, use o horário real com fuso e um
ID estável por gravação.

## Teste com o fluxo da Fase 4

Antes de substituir `data/fase4_recebidos/P0001_audio.json`, o evento convertido é testado
numa pasta separada, junto com cópias do vídeo e dos dados clínicos:

```powershell
$amostras = "output/thamy/janela"
New-Item -ItemType Directory -Force $amostras | Out-Null
Copy-Item "data/fase4_recebidos/P0001_video.json" $amostras
Copy-Item "data/fase4_recebidos/P0001_clinical.json" $amostras
Copy-Item "output/thamy/P0001_audio.json" $amostras
py -m src.monitoring.workflow --input $amostras --output "output/thamy/relatorio.json" --audit-log "logs/audit-fase4-thamy.jsonl" --summary-only
```

Resultado esperado: três eventos aceitos em um único grupo do `P0001`, com as fontes `audio`,
`video` e `clinical`, e o áudio como `audio-P0001-consulta-simulada-01` (`audio_check`). A
prioridade do alerta depende dos achados de vídeo e dados clínicos. Com os exemplos da
demonstração, o alerta fica `critical` por R1 (vídeo `high`) e R3 (vídeo e dados clínicos
anômalos), sem R4, porque o áudio não é anômalo.

Com o teste aprovado, o evento substitui o arquivo da pasta de recebidos:

```powershell
Copy-Item "output/thamy/P0001_audio.json" "data/fase4_recebidos/P0001_audio.json" -Force
py -m pytest
py -m pytest modules/audio_thamy/integration_checks
```

## Testes

| Suíte | Onde roda | O que cobre |
|---|---|---|
| `integration_checks/test_monitoring_event.py` | raiz do projeto | conversor, contrato v1 e consolidação com o vídeo e o clínico da demonstração |
| `integration_checks/test_wav_pipeline_to_event.py` | raiz do projeto | pipeline da Thamy com o WAV simulado, Azure simulado, conversão e validação do evento |
| `tests/` do módulo | `modules/audio_thamy` | testes originais do módulo de áudio, com Azure simulado |

O teste do WAV executa o pipeline da Thamy em um subprocesso dentro de `modules/audio_thamy`,
porque o pacote `src` do módulo tem o mesmo nome do `src` do projeto. A leitura do WAV e os
atributos acústicos são calculados com librosa; as chamadas ao Azure Speech e ao Azure AI
Language são substituídas pela transcrição e pela análise de texto do resultado enviado. O
teste é pulado se as dependências do módulo não estiverem instaladas.

Os testes do módulo rodam de dentro da pasta dele, para que `src` aponte para o código da
Thamy:

```powershell
Push-Location "modules/audio_thamy"
py -m pip install -r requirements.txt
py -m pytest tests
Pop-Location
```

Para esses testes bastam `librosa`, `soundfile`, `azure-cognitiveservices-speech`,
`azure-ai-textanalytics`, `azure-core` e `python-dotenv`. O pacote `datasets` só é usado por
`src/download_coswara.py`.

## Nova transcrição com Azure

Para processar uma nova gravação, preencha `modules/audio_thamy/.env` com as quatro variáveis
Azure listadas em `.env.example` e, dentro de `modules/audio_thamy`, rode
`py -m src.audio_pipeline`. O resultado é gravado em `outputs/audio_pipeline_result.json`, no
próprio módulo. Em seguida, repita a conversão com esse arquivo, informando o paciente
fictício, o ID da gravação e o horário correspondentes. Os arquivos exploratórios do Coswara
não fazem parte desta pasta.
