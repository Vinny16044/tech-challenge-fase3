# Relatório automático — anomalias em dados clínicos

> Achados automatizados para avaliação profissional. Não constituem diagnóstico.

Dataset: [MIMIC-IV Clinical Database Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/). Perfil de limiares: `icu`.

## Sinais vitais

- 19805 observações de 100 pacientes em 140 internações em UTI (505.0 pacientes-dia).
- Valores implausíveis descartados como artefato: 78.
- Isolation Forest treinado com 18948 observações completas.
- Observações anômalas: 3203 (16.2%).
- Eventos emitidos: 1871 (3.71 por paciente-dia); 1332 observações anômalas suprimidas por pertencerem a um episódio já avisado.

Observações anômalas por camada (uma observação pode acionar mais de uma):

| desvio_individual | faixa_referencia | tendencia | padrao_multivariado |
|---|---|---|---|
| 1503 | 1481 | 895 | 379 |

Eventos por prioridade sugerida:

| critical | high | medium | low |
|---|---|---|---|
| 107 | 314 | 298 | 1152 |

## Prescrições

566 alterações sinalizadas em 136 internações.

| dose_change | vasoactive_start | atypical_dose | order_burst |
|---|---|---|---|
| 372 | 102 | 61 | 31 |

## Movimentação entre unidades

9 padrões sinalizados.

| icu_return | frequent_transfers |
|---|---|
| 7 | 2 |

## Exemplos de anomalias detectadas

- `vital_signs` · **critical** · paciente P10004422 · 2111-01-19T03:50:00+00:00: Sinais vitais fora da faixa de referência: FC 127 bpm, FR 26 irpm, SpO2 95 %, PAS 78 mmHg. Escore parcial inspirado no NEWS2: 9. Desvio do padrão individual: FC +39 bpm em relação à mediana recente (88); SpO2 -5 % em relação à mediana recente (100). Piora rápida: escore subiu 9 pontos nas últimas observações. Motivo do aviso: agravamento.
- `prescription_change` · **critical** · paciente P10006053 · 2111-11-14T02:00:00+00:00: Associação de novo vasoativo (PHENYLEPHrine) a esquema que já incluía norepinephrine, vasopressin. Alteração sinalizada para revisão; pode ser ajuste planejado.
- `prescription_change` · **critical** · paciente P10006053 · 2111-11-14T02:00:00+00:00: Associação de novo vasoativo (Vasopressin) a esquema que já incluía norepinephrine. Alteração sinalizada para revisão; pode ser ajuste planejado.
- `vital_signs` · **critical** · paciente P10006053 · 2111-11-15T06:00:00+00:00: Sinais vitais fora da faixa de referência: FC 96 bpm, FR 25 irpm, SpO2 90 %, PAS 92 mmHg. Escore parcial inspirado no NEWS2: 9. Desvio do padrão individual: SpO2 -7 % em relação à mediana recente (97). Piora rápida: escore subiu 6 pontos nas últimas observações. Motivo do aviso: início do episódio.
- `vital_signs` · **critical** · paciente P10017492 · 2116-07-05T06:52:00+00:00: Sinais vitais fora da faixa de referência: FR 35 irpm, SpO2 85 %, PAS 70 mmHg. Escore parcial inspirado no NEWS2: 9. Combinação de sinais atípica para a população (Isolation Forest). Motivo do aviso: início do episódio.
- `prescription_change` · **critical** · paciente P10021487 · 2116-12-03T16:00:00+00:00: Associação de novo vasoativo (NORepinephrine) a esquema que já incluía phenylephrine. Alteração sinalizada para revisão; pode ser ajuste planejado.
- `patient_transfer` · **high** · paciente P10017492 · 2116-06-27T17:35:34+00:00: Retorno à terapia intensiva (Trauma SICU (TSICU)) 2 h depois da saída para PACU. Padrão de movimentação sinalizado para revisão.
- `patient_transfer` · **high** · paciente P10031757 · 2137-10-15T17:29:21+00:00: Retorno à terapia intensiva (Surgical Intensive Care Unit (SICU)) 24 h depois da saída para Transplant. Padrão de movimentação sinalizado para revisão.
- `patient_transfer` · **high** · paciente P10020740 · 2150-03-19T04:41:33+00:00: Retorno à terapia intensiva (Medical/Surgical Intensive Care Unit (MICU/SICU)) 2 h depois da saída para Medicine. Padrão de movimentação sinalizado para revisão.
