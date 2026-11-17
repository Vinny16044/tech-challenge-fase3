# Tech Challenge - Fase 3

## Assistente Virtual Médico com Fine-tuning de LLM e LangChain

**FIAP POS TECH - IADT (IA para Devs)**

Continuação das Fases 1 e 2: após a automação de análises de exames (modelos preditivos + SHAP)
e da interpretação via LLM, o hospital agora conta com um **assistente virtual médico** treinado
com os dados próprios da instituição, orquestrado com LangChain/LangGraph.

> ⚠️ Este sistema é uma ferramenta de **apoio à decisão clínica**. Ele **nunca prescreve
> diretamente, sem validação humana** — a decisão final é sempre do profissional de saúde.

## Equipe

| Integrante | Frente |
|---|---|
| Vinicius | Frente 1 — Fine-tuning da LLM (esta parte) |
| Thamy | Frente 2 — Dados (dataset de fine-tuning + base de prontuários) |
| Paola | Frente 3 — LangChain / LangGraph |
| Paola | Relatório técnico, diagrama, validação |
| Thamy | Vídeo e evidências |

## Início rápido (manage.py)

Não decore em qual shell cada coisa roda — o `manage.py` roteia sozinho
(comandos de GPU são executados dentro do WSL automaticamente). Tudo a partir
do **PowerShell**, na raiz do projeto:

```powershell
python manage.py doctor     # diagnóstico: o que está ok e o que falta
python manage.py setup      # instala dependências (Windows + WSL)
python manage.py db         # cria a base de prontuários no MySQL
python manage.py dataset    # gera o dataset de fine-tuning
python manage.py train      # treina no WSL (QLoRA na GPU)
python manage.py evaluate   # compara base vs. fine-tunado
python manage.py export     # exporta o modelo para GGUF
python manage.py ollama     # instala o modelo no Ollama (medico-fase3)
python manage.py ask "Quais exames estão pendentes?" -p P0001
python manage.py validate   # suíte de validação da Frente 3
```

Pré-requisitos: WSL2 com Ubuntu + driver NVIDIA (ver `docs/setup_wsl2.md`),
MySQL (local ou `docker compose up -d`) e [Ollama](https://ollama.com) no Windows.
Se a sua distro não se chama `Ubuntu-24.04`, defina `WSL_DISTRO` no `.env`.

## Frente 1 — Pipeline de Fine-tuning

- **Modelo base:** `unsloth/Llama-3.2-3B-Instruct` (4-bit)
- **Técnica:** QLoRA (LoRA sobre modelo quantizado em 4 bits) via [Unsloth](https://unsloth.ai)
- **Hardware alvo:** RTX 3070 (8 GB VRAM), rodando em **WSL2** — ver `docs/setup_wsl2.md`
- **Dados:** protocolos internos do hospital (sintéticos), FAQ de médicos e modelos de
  laudos/receitas, em formato instrução/resposta (`data/processed/*.jsonl`).
  A Frente 2 complementa com MedQuAD/PubMedQA convertidos pelo mesmo script.

### Estrutura

```
tech-challenge-fase3/
├── configs/finetune.yaml            # Hiperparâmetros do treino (edite aqui)
├── data/
│   ├── synthetic/                   # Protocolos/FAQ sintéticos do hospital (fonte)
│   ├── processed/                   # JSONL instrução/resposta gerado (entra no treino)
│   └── eval/                        # Perguntas de avaliação antes/depois
├── src/finetune/
│   ├── prompts.py                   # System prompt + guardrails (portado da Fase 2)
│   ├── prepare_dataset.py           # Curadoria/anonimização -> JSONL de treino
│   ├── train.py                     # Treino QLoRA com Unsloth
│   ├── evaluate.py                  # Comparação base vs. fine-tunado
│   └── inference.py                 # Chat local com o modelo + logging de auditoria
├── models/                          # Adapters LoRA salvos (gitignored)
├── logs/                            # Logs de auditoria (JSONL)
└── docs/setup_wsl2.md               # Setup do ambiente WSL2 + CUDA
```

### Como executar (dentro do WSL2)

```bash
# 1. Ambiente
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-finetune.txt

# 2. Gerar o dataset de treino a partir dos dados sintéticos
python -m src.finetune.prepare_dataset

# 3. Treinar (QLoRA na GPU)
python -m src.finetune.train

# 4. Avaliar: mesmas perguntas no modelo base e no fine-tunado
python -m src.finetune.evaluate

# 5. Conversar com o assistente (com guardrails e log de auditoria)
python -m src.finetune.inference --question "Qual o protocolo de triagem para nódulo mamário?"
python -m src.finetune.inference            # modo chat interativo
```

Os passos 2 (e os testes do 5 em modo `--offline`) rodam em qualquer máquina;
os passos 3–5 exigem GPU NVIDIA (WSL2 ou Colab).

### Saídas

- `models/lora-llama32-3b-medico/` — adapter LoRA (e opcionalmente o modelo mesclado)
- `results/eval_before_after.jsonl` — respostas base vs. fine-tunado por pergunta
- `logs/audit-YYYYMMDD.jsonl` — trilha de auditoria de cada interação

## Frente 2 — Dados (Thammy)

- **Dataset de fine-tuning**: além dos dados sintéticos do hospital, o
  `src/data/convert_medquad.py` converte o [MedQuAD](https://github.com/abachaa/MedQuAD)
  (subconjunto CancerGov, oncologia) para `data/raw/medquad_cancer.csv`, com curadoria de
  tamanho e fonte por par. O `prepare_dataset` mistura tudo: exemplos externos (em inglês)
  recebem um system prompt neutro em inglês, e os exemplos PT do hospital são repetidos
  (`--synthetic-repeat`, padrão 3x) para o português continuar dominante.
- **Base de prontuários (MySQL)**: `src/data/build_prontuarios.py` cria as tabelas
  `pacientes` (dados SEER anonimizados — sem raça/estado civil, id sintético),
  `exames` (pré-tratamento do PROT-ONCO-002, ~18% pendentes, sintéticos) e
  `alertas` (pendências >30 dias, PROT-ONCO-005) — é o banco que o LangChain consulta.

```bash
pip install -r requirements-data.txt

# MySQL: configure DATABASE_URL no .env (ver .env.example).
# Sem MySQL instalado: docker compose up -d
python -m src.data.build_prontuarios

# Dataset externo (opcional, já versionado em data/raw/medquad_cancer.csv):
git clone --depth 1 https://github.com/abachaa/MedQuAD.git /tmp/MedQuAD
python -m src.data.convert_medquad --medquad-dir /tmp/MedQuAD --limit 400
```

## Integração com as demais frentes

- A Frente 3 (LangChain) consome o modelo por carregamento direto do adapter
  (`FastLanguageModel.from_pretrained` + `PeftModel`) ou via **Ollama** após exportar GGUF
  (`python -m src.finetune.train --export-gguf`).
- O `prompts.SYSTEM_ASSISTENTE` é o mesmo usado no grafo LangGraph — fonte única de guardrails.

## Licença

MIT — Projeto acadêmico FIAP POS TECH 2026

---

# Fase 4 — Integração multimodal (áudio, vídeo e dados clínicos)

Protótipo de monitoramento hospitalar que consolida os achados dos módulos de **áudio**
(Thamy), **vídeo** e **detecção de anomalias em dados clínicos** (Vini). A arquitetura e a
integração (Paola) recebem os JSONs dos módulos, validam, consolidam por paciente e janela de
tempo, aplicam regras de alerta determinísticas e produzem um relatório estruturado com
auditoria e revisão humana.

> ⚠️ Os resultados são **achados para avaliação profissional**. Não são diagnóstico nem
> conduta. Todo alerta nasce em `pending_medical_review`. Os dados são fictícios ou vêm de
> datasets públicos desidentificados (MIMIC-IV Demo e UR Fall Detection).

**Relatório técnico da Fase 4:**
[docs/Relatorio_Tecnico_Fase4.docx](docs/Relatorio_Tecnico_Fase4.docx) — fluxo multimodal,
modelos aplicados em cada tipo de dado, resultados, exemplos de anomalias e limitações.
Diagrama do fluxo: [docs/diagrama_fase4.png](docs/diagrama_fase4.png).

A arquitetura está descrita em [docs/fase4_arquitetura.md](docs/fase4_arquitetura.md):
diagrama do fluxo, validação e agrupamento, regras de prioridade, resumo offline e com
Ollama, auditoria em JSONL e evidências dos testes locais.

Os módulos de vídeo e de anomalias clínicas estão em
[docs/fase4_video_e_anomalias.md](docs/fase4_video_e_anomalias.md): datasets, métodos,
resultados, exemplos de anomalias e limitações.

## Estrutura da Fase 4

```
src/contracts/event_v1.py         # Contrato JSON v1, versão 1.1 (Pydantic) — fonte da verdade
src/contracts/review.py           # Decisão de revisão humana
src/contracts/generate_schema.py  # Gera schemas/event_v1.schema.json
src/monitoring/workflow.py        # Grafo LangGraph + CLI
src/monitoring/nodes.py           # Nós: validar, consolidar, alertar, revisar, resumir, relatório
src/monitoring/rules.py           # Regras de prioridade do protótipo (R0–R5)
src/monitoring/summarizer.py      # Resumo determinístico + LLM opcional com fallback
src/monitoring/audit.py           # Auditoria JSONL
schemas/event_v1.schema.json      # JSON Schema gerado
data/fase4/                       # Demonstração de referência do P0001 (áudio, vídeo, clínico)
data/fase4_recebidos/             # Amostras do teste conjunto: saídas reais dos três módulos
data/fase4_revisao/               # Exemplo de decisão de revisão humana
data/fase4_vini/                  # Saídas de vídeo (70 clipes, com a validação) e de um paciente clínico
modules/audio_thamy/              # Módulo de áudio da Thamy e conversor para o contrato v1
src/video_analysis/               # Vídeo (Vini): OpenCV, YOLOv8-pose, YOLOv8, regras de movimento, relatório
src/anomaly_detection/            # Anomalias clínicas (Vini): sinais vitais, prescrições, movimentação
src/fase4_samples.py              # Gera as amostras de vídeo e clínico para a integração
src/api/fase4.py                  # Painel web da Fase 4 (rotas); página em src/api/static/fase4.html
docs/fase4_video_e_anomalias.md   # Vídeo e anomalias: datasets, métodos, resultados, limitações
docs/fase4_relatorio_video.md     # Relatório automático da análise de vídeo
docs/fase4_relatorio_clinico.md   # Relatório automático das anomalias clínicas
docs/fase4_arquitetura.md         # Arquitetura, fluxo e evidências dos testes
docs/fase4_contrato_eventos_v1.md # Campos, valores aceitos, evidências, IDs estáveis
docs/fase4_regras_alerta.md       # Regras, revisão, resumo e auditoria
tests/                            # Testes (pytest) com dados fictícios
```

A Fase 4 segue o padrão do grafo da Fase 3 (`src/graph/`): `StateGraph` com estado
`TypedDict`, nós em uma classe, auditoria JSONL e execução offline.

## Fluxo LangGraph

```
validate_events ──(nenhum evento aceito)──────────────────────────────┐
      │                                                               ▼
      └─► consolidate ─► evaluate_alerts ─► apply_reviews ─► summarize ─► build_report
```

1. **validate_events**: lê o JSON bruto, valida contra o contrato v1, rejeita JSON malformado,
   descarta reenvios idênticos e rejeita conflitos de `event_id`.
2. **consolidate**: agrupa por `patient_id` + janela (padrão 15 min), registra fontes recebidas
   e ausentes e preserva `event_id`, `source` e `origin` (arquivo) de cada evento.
3. **evaluate_alerts**: aplica as regras R0–R5 e registra motivos e evidências.
4. **apply_reviews**: registra decisões humanas informadas (revisor + horário).
5. **summarize**: resumo determinístico; LLM (Ollama) só com `--llm`, com fallback.
6. **build_report**: relatório estruturado + linha final de auditoria.

## Como executar (PowerShell, na raiz do projeto)

Os comandos usam o inicializador `py` do Python para Windows.

```powershell
py -m pip install -r requirements-fase4.txt          # se ainda não tiver as dependências
py -m src.monitoring.workflow --input data/fase4 --summary-only
py -m src.monitoring.workflow --input data/fase4 --output output/relatorio_P0001.json
py -m src.monitoring.workflow --input data/fase4 --review-file data/fase4_revisao/P0001_decisao_exemplo.json
py -m src.monitoring.workflow --input data/fase4 --window-minutes 30
py -m src.monitoring.workflow --input data/fase4 --llm --model llama3.2:3b   # opcional
py -m src.contracts.generate_schema                  # regenera o JSON Schema
py -m pytest                                         # testes
```

Auditoria: `logs/audit-fase4-AAAAMMDD.jsonl` (ou `--audit-log <arquivo>`).

### Saída consolidada (demonstração `data/fase4`, `--summary-only`)

```
Paciente P0001 | janela 2026-11-12T13:00:00Z a 2026-11-12T13:15:00Z (UTC).
Fontes recebidas: audio, video, clinical. Fontes ausentes: nenhuma.
3 evento(s) recebido(s), 3 com anomalia.
- [audio] vocal_distress (ANOMALIA, prioridade do módulo: medium): Vocalizações repetidas ... [audio-P0001-rec0007-seg0042]
- [video] bed_exit_attempt (ANOMALIA, prioridade do módulo: high): Movimento de tentativa de sair do leito ... [video-P0001-cam12-clip0315]
- [clinical] vital_signs (ANOMALIA, prioridade do módulo: medium): Frequência cardíaca e saturação ... [clinical-P0001-obs5521]
Alerta ALT-8935b56efce5: prioridade critical (R1 PRIORIDADE_MODULO; R3 CORROBORACAO_2; R4 CORROBORACAO_3). Status: pending_medical_review.
Achados automatizados para avaliação profissional. Não constituem diagnóstico nem indicação de conduta; a decisão é do profissional de saúde.
```

### Saída consolidada (demonstração `data/fase4`, JSON — trecho)

Relatório completo: `docs/fase4_exemplo_relatorio_P0001.json`.

```json
{
  "report_version": "1.0",
  "contract_version": "1.1",
  "rules_version": "prototipo-1",
  "config": { "window_minutes": 15, "summary_mode": "offline", "model": null },
  "intake": { "events_received": 3, "accepted": 3, "rejected": 0, "duplicates": 0 },
  "sources_received": ["audio", "video", "clinical"],
  "sources_missing": [],
  "alerts": [
    {
      "alert_id": "ALT-8935b56efce5",
      "patient_id": "P0001",
      "window_start": "2026-11-12T13:00:00Z",
      "priority": "critical",
      "review_status": "pending_medical_review"
    }
  ],
  "groups": [
    {
      "group_id": "P0001@2026-11-12T13:00:00Z",
      "status": "alert_pending_review",
      "sources_received": ["audio", "video", "clinical"],
      "sources_missing": [],
      "events": [
        { "event_id": "audio-P0001-rec0007-seg0042", "module": "audio", "origin": "data/fase4/P0001_audio.json", "...": "..." }
      ],
      "alert": {
        "alert_id": "ALT-8935b56efce5",
        "priority": "critical",
        "priority_reasons": [
          { "rule_id": "R1", "priority_after": "high", "event_ids": ["video-P0001-cam12-clip0315"] },
          { "rule_id": "R3", "priority_before": "high", "priority_after": "critical" },
          { "rule_id": "R4", "priority_before": "critical", "priority_after": "critical" }
        ],
        "evidence": [ { "event_id": "audio-P0001-rec0007-seg0042", "type": "audio_segment", "uri": "audio://leito-12/rec0007.wav", "start_s": 120.0, "end_s": 160.0 } ],
        "review": { "status": "pending_medical_review", "reviewer_id": null, "decided_at": null }
      },
      "summary": { "method": "deterministic", "text": "Paciente P0001 | janela ..." }
    }
  ]
}
```

## Situação da integração

- **Áudio (Thamy):** integrado. O resultado da consulta simulada foi convertido para o
  contrato v1 por `modules/audio_thamy/to_monitoring_event.py` e está em
  `data/fase4_recebidos/P0001_audio.json`, como `audio_check` (busca por palavras-chave, sem
  anomalia). Nos testes, o WAV simulado é processado localmente e os serviços Azure são
  simulados. Detalhes em `modules/audio_thamy/INTEGRAR_COM_FASE4.md`.
- **Vídeo e dados clínicos (Vini):** integrados. `data/fase4_recebidos/P0001_video.json` e
  `P0001_clinical.json` são saídas dos módulos, geradas por `py -m src.fase4_samples`: uma
  queda detectada num clipe do UR Fall Detection Dataset, e sinais vitais críticos mais uma
  mudança de prescrição de um paciente do MIMIC-IV Demo. O `P0001` do teste conjunto é
  composto: vídeo e dados clínicos vêm de datasets diferentes, com horários deslocados
  para a mesma janela. Detalhes em `docs/fase4_video_e_anomalias.md`.

Como as saídas do Vini responderam aos pontos em aberto:

- **Tipos de evento:** o contrato passou à versão 1.1, com `prescription_change`,
  `patient_transfer` e a evidência `prescription`. Eventos 1.0 continuam válidos.
- **Um evento por conjunto:** o detector clínico emite um `vital_signs` por observação, com
  uma evidência por sinal alterado, e só no início, no agravamento ou na persistência de um
  episódio.
- **`event_id` estável:** montado com a origem do dado (internação e horário, ou clipe,
  pessoa e instante), então o reenvio repete o ID.
- **Fuso:** os horários do MIMIC-IV são tratados como UTC; o vídeo usa o início da gravação
  informado em `--recorded-at`.

Para toda a equipe: calibrar as prioridades sugeridas pelos módulos e as regras R1–R4, que
são regras de protótipo, e definir os esquemas de `uri` (onde os artefatos ficam guardados e
quem pode acessá-los). Com um evento clínico por hora e um vídeo de segundos, a janela de
15 minutos só junta as modalidades quando os relógios estão alinhados.

## Fase 4 — Guia rápido para a equipe (teste conjunto)

**1. Instale as dependências** (PowerShell, na raiz do projeto):

```powershell
py -m pip install -r requirements-fase4.txt
```

**2. Substitua o seu arquivo** em `data/fase4_recebidos/`, mantendo o nome exato:

| Pessoa | Arquivo | Situação |
|---|---|---|
| Thamy | `data/fase4_recebidos/P0001_audio.json` | substituído pelo áudio convertido |
| Vini | `data/fase4_recebidos/P0001_video.json` | substituído pela saída do módulo de vídeo |
| Vini | `data/fase4_recebidos/P0001_clinical.json` | substituído pela saída do módulo clínico |

A pasta `data/fase4/` é a demonstração de referência e não deve ser alterada.

**3. Siga o contrato v1:**

- Especificação: `docs/fase4_contrato_eventos_v1.md` (campos, `event_type` por módulo,
  evidências, `event_id` estável e horário com fuso)
- JSON Schema: `schemas/event_v1.schema.json`

No **primeiro teste conjunto**, as três amostras devem usar o paciente fictício `P0001` e a
mesma janela fictícia dos exemplos: **12/11/2026, entre 10:00:00 e 10:14:59 `-03:00`**
(13:00–13:15 UTC). Se o seu módulo precisar de outro campo ou `event_type`, avise junto com
a amostra.

**4. Rode o fluxo com as amostras recebidas:**

```powershell
py -m src.monitoring.workflow --input "data\fase4_recebidos" --summary-only
```

Resultado esperado: **um grupo** (`P0001 | janela 2026-11-12T13:00:00Z a 2026-11-12T13:15:00Z`)
com `Fontes recebidas: audio, video, clinical` e `Fontes ausentes: nenhuma`, e a última linha
com `4 aceitos, 0 rejeitados` (um de áudio, um de vídeo e dois clínicos). Se algum evento for rejeitado, rode sem `--summary-only` e veja
`intake.rejected_events` (arquivo de origem, motivo e erros de cada campo).

**5. Rode a suíte de testes:**

```powershell
py -m pytest
```

`tests/test_fase4_recebidos.py` confere as amostras da pasta de recebidos: contrato v1, módulo
compatível com o nome do arquivo, `P0001` e consolidação das três fontes num só grupo.

## Fase 4 — Painel web

O mesmo servidor do chat da Fase 3 serve um painel da Fase 4, com quatro abas:

- **Alerta integrado:** roda o fluxo sobre `data/fase4_recebidos/`, mostra os eventos das
  três modalidades, as regras que definiram a prioridade e o resumo, e registra a revisão
  humana (confirmar ou descartar) com revisor, horário e auditoria.
- **Vídeo:** os 70 clipes, com o vídeo anotado (esqueleto, objetos e avisos), os achados
  de cada um e o resultado da validação; há filtro para os erros do detector.
- **Dados clínicos:** gráfico dos sinais vitais do paciente de exemplo, com os avisos
  marcados, e um replay que reapresenta a internação observação a observação.
- **Áudio:** gravação, transcrição, sentimento, termos sinalizados e atributos acústicos.

```powershell
py -m pip install -r requirements-fase4.txt -r requirements-fase4-video-clinico.txt -r requirements-web.txt
py -m uvicorn src.api.server:app --port 8000     # ou: python manage.py web
```

Abra `http://localhost:8000/fase4`. O painel lê as saídas já geradas: os vídeos anotados
vêm de `output/video/` (rode `py -m src.video_analysis.run --crop right --annotate`) e o
gráfico precisa do MIMIC-IV Demo baixado. Sem eles, mostra só os resultados versionados.
As decisões de revisão ficam em `output/fase4_painel/revisoes.json`.

## Fase 4 — Vídeo e anomalias clínicas (Vini)

```powershell
py -m pip install -r requirements-fase4.txt -r requirements-fase4-video-clinico.txt
py -m src.anomaly_detection.data_loader --download   # MIMIC-IV Demo, ≈ 6 MB
py -m src.video_analysis.preprocessing --download --all  # 70 clipes do URFD, ≈ 93 MB
py -m src.anomaly_detection.run                      # sinais vitais, prescrições, movimentação
py -m src.anomaly_detection.run --stream 30057454    # replay de uma internação em tempo real
py -m src.anomaly_detection.evaluate                 # avaliação com anomalias injetadas
py -m src.video_analysis.run --crop right --annotate # poses, objetos, eventos, vídeos anotados
py -m src.fase4_samples                              # amostras para a integração
```

As saídas ficam em `output/clinical/` e `output/video/` (fora do Git). Resultados na
execução de 16/11/2026:

- **Vídeo:** limiares ajustados em 8 clipes e validados nos outros 62 do dataset, sem
  reajuste: 22 de 26 quedas detectadas (84,6%) e 3 falsos alarmes em 36 atividades normais
  (especificidade de 91,7%), a ≈ 39 quadros/s em CPU. Três das quatro quedas não detectadas
  partem da posição sentada.
- **Sinais vitais:** 19.805 observações de 100 pacientes, 16,2% anômalas, 1.871 eventos
  (3,7 por paciente-dia) no perfil de UTI.
- **Prescrições e movimentação:** 566 alterações sinalizadas em 136 internações e 9 padrões
  de movimentação.
- **Anomalias injetadas:** sensibilidade de 100% com episódios completos e 74% com
  episódios discretos, com 3,0% de falsos positivos.
