# Módulo de Análise de Áudio — FIAP Pós-Tech | Fase 4

## 1. Visão geral

Este projeto implementa um módulo de processamento de áudio para integração a um assistente clínico multimodal desenvolvido no Tech Challenge da FIAP.

O módulo combina técnicas de processamento digital de sinais, reconhecimento automático de fala e análise de linguagem natural.

O objetivo é transformar gravações de áudio em informações estruturadas que possam apoiar fluxos de revisão humana e integração com outros componentes do sistema.

**O módulo não realiza diagnósticos médicos.**

## 2. Funcionalidades implementadas

- Validação de arquivos de áudio.
- Extração de características acústicas, incluindo RMS, ZCR, centroide espectral e MFCC.
- Identificação de intervalos de atividade sonora utilizando limiar de RMS.
- Transcrição de fala em português brasileiro com Azure AI Speech.
- Análise textual com Azure AI Language.
- Extração de frases-chave, entidades e sentimento.
- Identificação demonstrativa de termos predefinidos que exigem revisão humana.
- Consolidação dos resultados em JSON.
- Registro estruturado de erros.
- Testes automatizados com pytest e simulações dos serviços Azure.

## 3. Tecnologias

| Tecnologia | Utilização |
|---|---|
| Python | Implementação principal |
| Librosa | Processamento e características acústicas |
| NumPy | Operações numéricas |
| Pandas | Organização de dados |
| Matplotlib | Visualização |
| Azure AI Speech | Transcrição |
| Azure AI Language | Análise textual |
| Pytest | Testes automatizados |
| Python-dotenv | Variáveis de ambiente |

## 4. Dataset

O projeto utiliza uma amostra do dataset Coswara, contendo gravações de respiração, tosse, contagem e vocalizações.

Foram selecionados 15 arquivos WAV para exploração acústica.

Também foi utilizada uma gravação simulada de consulta em português brasileiro para validar o fluxo de transcrição e análise textual.

Fontes:

- [Coswara — projeto original](https://github.com/iiscleap/Coswara-Data)
- [Coswara — Hugging Face](https://huggingface.co/datasets/szzs1693/coswara-data)

O Coswara é utilizado para exploração de sinais acústicos, não para validação diagnóstica.

## 5. Configuração do ambiente

Recomenda-se utilizar uma versão do Python compatível com as dependências do projeto. O ambiente de desenvolvimento validado utiliza Python 3.13.14.

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se a ativação do ambiente for bloqueada:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. Configuração do Azure

Crie um arquivo `.env` na raiz do projeto, utilizando `.env.example` como referência.

Variáveis necessárias:

```env
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_LANGUAGE_KEY=
AZURE_LANGUAGE_ENDPOINT=
```

As credenciais devem ser obtidas nos respectivos recursos do Azure.

**Nunca compartilhe o arquivo `.env` ou suas chaves de acesso.**

## 7. Execução

Na raiz do projeto, execute:

```powershell
python -m src.audio_pipeline
```

O pipeline realiza as seguintes etapas:

1. Validação do áudio.
2. Transcrição da fala.
3. Análise textual.
4. Geração de indicadores demonstrativos.
5. Extração de características acústicas.
6. Consolidação e gravação do resultado.

O arquivo principal é salvo em:

`outputs/audio_pipeline_result.json`

Em caso de falha, um registro estruturado é salvo em:

`outputs/audio_pipeline_error.json`

## 8. Estrutura do resultado

O JSON de sucesso contém os seguintes campos:

| Campo | Descrição |
|---|---|
| `module` | Identificação do módulo |
| `schema_version` | Versão do contrato |
| `analysis_id` | Identificador único da análise |
| `status` | Estado da execução |
| `processed_at` | Data e hora de processamento |
| `audio_metadata` | Metadados do arquivo |
| `transcription` | Texto transcrito |
| `text_analysis` | Análise textual |
| `acoustic_features` | Características acústicas |
| `alerts` | Alertas demonstrativos |
| `limitations` | Limitações de interpretação |

O contrato atual utiliza a versão `1.0.0`.

## 9. Testes automatizados

Execute:

```powershell
python -m pytest tests/ -v
```

Na última validação, foram aprovados 25 testes, cobrindo:

- Validação de arquivos.
- Transcrição com serviços simulados.
- Análise textual com serviços simulados.
- Contrato JSON.
- Tratamento de erros.
- Integração do pipeline.

Os testes automatizados utilizam simulações para evitar chamadas desnecessárias ao Azure.

## 10. Resultado da validação

A execução de ponta a ponta com serviços Azure reais apresentou:

| Indicador | Resultado |
|---|---|
| Status | `success` |
| Duração do áudio | 19,817 segundos |
| Taxa de amostragem | 16.000 Hz |
| RMS médio | 0,021608 |
| ZCR médio | 0,158796 |
| Intervalos de atividade sonora | 12 |
| Alertas demonstrativos | 1 |
| Testes automatizados | 25 aprovados |

Esses valores correspondem ao arquivo de consulta simulada utilizado na validação e não representam métricas gerais de desempenho do sistema.

## 11. Limitações

O módulo possui limitações importantes:

- Os intervalos detectados por RMS representam atividade sonora, não identificação validada de tosse ou respiração.
- A transcrição automática pode apresentar erros e exige revisão.
- A análise de sentimento não corresponde a avaliação clínica.
- Os alertas são baseados em termos predefinidos, não em diagnóstico.
- O dataset Coswara não representa consultas hospitalares reais.
- Não foi validado um classificador clínico de eventos respiratórios.

O módulo deve ser utilizado como componente de apoio ao processamento e à revisão humana.

## 12. Integração

O resultado estruturado pode ser consumido por outros módulos do assistente clínico.

A integração deverá considerar a versão do contrato JSON, os estados de sucesso e erro, a necessidade de revisão humana e as limitações clínicas descritas.

## 13. Segurança e privacidade

- Não versionar credenciais.
- Não utilizar gravações identificáveis de pacientes sem autorização e base legal adequada.
- Evitar registrar informações pessoais em logs.
- Aplicar controles de acesso e retenção aos arquivos de áudio e resultados.
- Garantir revisão humana antes de qualquer utilização clínica.