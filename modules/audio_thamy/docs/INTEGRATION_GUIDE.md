# Guia de Integração — Módulo de Análise de Áudio

**Projeto:** FIAP Pós-Tech — AI for DEVS  
**Componente:** Audio Analysis  
**Versão do contrato:** 1.0.0  
**Status:** Implementado e validado em ambiente de desenvolvimento
**Responsável:** Thamy Santos

## 1. Objetivo

Este documento descreve como integrar o módulo de análise de áudio ao assistente clínico multimodal desenvolvido para o Tech Challenge da FIAP.

O módulo recebe um arquivo de áudio, realiza o processamento acústico, transcreve a fala utilizando Azure AI Speech, analisa o texto com Azure AI Language e produz um resultado estruturado em JSON.

A integração permite que outros componentes do sistema utilizem informações extraídas de gravações simuladas, mantendo as limitações clínicas e a necessidade de revisão humana.

## 2. Arquitetura

O processamento segue o fluxo:

```text
Arquivo WAV
    |
    v
Validação do áudio
    |
    v
Azure AI Speech
    |
    v
Transcrição
    |
    v
Azure AI Language
    |
    v
Análise textual
    |
    v
Extração de características acústicas
    |
    v
Geração de alertas demonstrativos
    |
    v
Resultado JSON
    |
    v
Assistente clínico multimodal
```

Os alertas são gerados a partir de termos predefinidos na transcrição e não constituem avaliação diagnóstica.

## 3. Interface de entrada

O módulo recebe um caminho para um arquivo de áudio.

**Formato utilizado na validação:**

- WAV
- Mono
- Taxa de amostragem de 16.000 Hz
- Idioma de transcrição: português brasileiro (`pt-BR`)

O pipeline utiliza a função:

```python
from src.audio_pipeline import run_pipeline

result = run_pipeline(
    audio_path="data/raw/consulta_simulada.wav",
    output_path="outputs/audio_pipeline_result.json",
    error_path="outputs/audio_pipeline_error.json"
)
```

O módulo depende de credenciais válidas para Azure AI Speech e Azure AI Language.

Os caminhos devem ser ajustados conforme o diretório de execução e a estrutura do projeto integrado.

## 4. Interface de saída

Em caso de sucesso, o pipeline retorna um dicionário Python e grava o resultado no arquivo:

`outputs/audio_pipeline_result.json`

O contrato possui a seguinte estrutura:

| Campo | Tipo | Descrição |
|---|---|---|
| `module` | string | Identificação do módulo |
| `schema_version` | string | Versão do contrato |
| `analysis_id` | string | Identificador único da execução |
| `status` | string | Estado da execução |
| `processed_at` | string | Data e hora em UTC |
| `audio_metadata` | object | Informações do arquivo |
| `transcription` | object | Texto reconhecido |
| `text_analysis` | object | Resultado da análise textual |
| `acoustic_features` | object | Indicadores acústicos |
| `alerts` | array | Alertas demonstrativos |
| `limitations` | object | Limitações do processamento |

**Exemplo completo:** `docs/audio_result_example.json`

O exemplo utiliza dados fictícios e deve ser tratado como uma referência de formato, não como resultado de uma análise clínica.

## 5. Campos utilizados na integração

### 5.1 Transcrição

```python
transcription = result["transcription"]["text"]
```

Contém o texto reconhecido pelo Azure Speech.

O texto pode apresentar erros de reconhecimento e deve ser revisado antes de qualquer utilização clínica.

### 5.2 Análise textual

```python
text_analysis = result["text_analysis"]

sentiment = text_analysis["sentiment"]["label"]
key_phrases = text_analysis["key_phrases"]
entities = text_analysis["entities"]
```

O Azure AI Language fornece indicadores linguísticos.

A classificação de sentimento não representa o estado psicológico ou a condição médica do paciente.

### 5.3 Características acústicas

```python
features = result["acoustic_features"]

rms = features["rms_mean"]
zcr = features["zcr_mean"]
mfcc = features["mfcc_mean"]
events = features["detected_events"]
```

Os indicadores permitem explorar propriedades físicas do sinal.

Os eventos identificados por limiar de RMS representam intervalos de atividade sonora. Não são detecções clinicamente validadas de tosse, respiração ou alterações de saúde.

### 5.4 Alertas

```python
alerts = result["alerts"]

for alert in alerts:
    if alert["requires_human_review"]:
        print(alert["detected_terms"])
```

Os alertas indicam termos predefinidos encontrados na transcrição.

Não devem gerar automaticamente diagnósticos, tratamentos ou decisões clínicas.

## 6. Tratamento de erros

Se ocorrer uma falha, o pipeline registra um JSON de erro em:

`outputs/audio_pipeline_error.json`

A estrutura inclui:

```json
{
  "status": "error",
  "error": {
    "stage": "transcription",
    "type": "RuntimeError",
    "message": "Falha demonstrativa de transcrição"
  }
}
```

Esse é um exemplo parcial e ilustrativo da estrutura de erro.

Na implementação atual, a função `run_pipeline()` propaga a exceção após registrar a falha.

Portanto, o componente integrador deve tratar exceções:

```python
from src.audio_pipeline import run_pipeline

try:
    result = run_pipeline(
        audio_path="data/raw/consulta_simulada.wav"
    )
except Exception as exc:
    print(f"Falha no módulo de áudio: {type(exc).__name__}")
    result = None

if result is not None:
    print(result["status"])
```

Em produção, informações sensíveis não devem ser expostas em logs.

O arquivo de resultado anterior não deve ser utilizado como substituto de uma execução que falhou.

## 7. Exemplo de integração

```python
from src.audio_pipeline import run_pipeline


def process_audio_for_assistant(audio_path):
    try:
        result = run_pipeline(audio_path=audio_path)

    except Exception as exc:
        return {
            "status": "error",
            "source": "audio_analysis",
            "error_type": type(exc).__name__
        }

    return {
        "status": result["status"],
        "source": result["module"],
        "analysis_id": result["analysis_id"],
        "transcription": result["transcription"]["text"],
        "key_phrases": result["text_analysis"]["key_phrases"],
        "alerts": result["alerts"],
        "requires_human_review": (
            result["limitations"]["transcription_requires_review"]
            or any(
                alert["requires_human_review"]
                for alert in result["alerts"]
            )
        )
    }
```

Esse exemplo apresenta um adaptador simplificado para o assistente clínico.

A conexão com LangChain, banco de dados, workflows e interface deverá ser implementada e testada pela equipe no projeto principal.

## 8. Testes e validação

O módulo possui testes automatizados com pytest.

Para executar:

```powershell
python -m pytest tests/ -q
```

**Último resultado registrado:** 25 testes aprovados.

Também foi realizada uma execução real utilizando Azure Speech e Azure AI Language.

Resultados da validação:

| Indicador | Resultado |
|---|---|
| Status | success |
| Duração do áudio | 19,817 s |
| Taxa de amostragem | 16.000 Hz |
| Eventos de atividade sonora | 12 |
| Alertas demonstrativos | 1 |
| Testes automatizados | 25 aprovados |

Os resultados se referem a uma gravação simulada específica.

## 9. Requisitos de integração

Antes de conectar o módulo ao assistente principal, a equipe deverá:

1. Instalar as dependências descritas no `requirements.txt`.
2. Configurar as variáveis de ambiente Azure.
3. Garantir acesso ao arquivo de áudio.
4. Executar o pipeline a partir do ambiente correto.
5. Validar `schema_version` e `status` antes de consumir o resultado.
6. Tratar exceções e falhas de processamento.
7. Preservar a necessidade de revisão humana.
8. Definir como os resultados serão armazenados e associados ao contexto da aplicação.

O módulo atualmente disponibiliza uma interface Python e um arquivo JSON. Não há uma API HTTP implementada nesta etapa.

## 10. Segurança e limitações

O módulo é um protótipo acadêmico.

Não realiza diagnósticos, recomendações terapêuticas ou decisões clínicas autônomas.

A integração deve considerar:

- Proteção de credenciais e dados.
- Revisão humana das transcrições.
- Limitação dos indicadores acústicos.
- Validação dos resultados antes de uso.
- Tratamento de falhas sem reutilizar resultados antigos.
- Proteção de informações sensíveis em logs e arquivos.

