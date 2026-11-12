# Regras de alerta — protótipo (versão `prototipo-1`)

> Regras **do protótipo**, criadas para demonstrar a integração. Não foram validadas
> clinicamente e não substituem protocolos da instituição. Todo resultado é um **achado para
> avaliação profissional**; nenhum alerta é entregue como diagnóstico ou conduta.

Implementação: `src/monitoring/rules.py`. Testes: `tests/test_alert_rules.py`.

## Consolidação

- Eventos aceitos são agrupados por **paciente** e **janela de tempo** fixa (padrão 15 min,
  configurável com `--window-minutes`). As janelas são alinhadas em UTC
  (ex.: 13:00–13:15Z), então o agrupamento não depende da ordem de chegada.
- Limitação conhecida: eventos próximos mas em lados opostos de uma borda (ex.: 13:14 e
  13:16) caem em janelas diferentes.
- Cada grupo informa `sources_received` e `sources_missing`.

## Regras (aplicadas em ordem; só contam eventos com `anomaly_detected = true`)

| ID | Nome | Condição | Efeito |
|---|---|---|---|
| R0 | SEM_ANOMALIA | nenhum evento anômalo no grupo | sem alerta (`status = no_alert`) |
| R1 | PRIORIDADE_MODULO | sempre que há anomalia | prioridade inicial = maior `priority` sugerida pelos módulos |
| R2 | EVENTO_SENSIVEL | `fall_detected` ou `call_for_help` anômalo | no mínimo `high` |
| R3 | CORROBORACAO_2 | anomalias em 2+ modalidades diferentes | sobe um nível (máx. `critical`) |
| R4 | CORROBORACAO_3 | anomalias nas 3 modalidades | `critical` |
| R5 | AVALIACAO_PARCIAL | alguma modalidade sem eventos na janela | informativa; não altera a prioridade |

Cada regra disparada vira um item em `priority_reasons` com `rule_id`, descrição,
`priority_before`, `priority_after` e os `event_ids` que a motivaram. O alerta também lista
todas as evidências dos eventos anômalos, cada uma com `event_id` e `module`.

## Identificação e revisão humana

- `alert_id = "ALT-" + sha256(patient_id | window_start | event_ids ordenados)[:12]`.
  Mesmos eventos → mesmo alerta, em qualquer execução. Se chegar um evento anômalo novo na
  janela, o alerta muda de ID e precisa de nova revisão.
- Todo alerta nasce com `review.status = "pending_medical_review"`.
- Uma decisão (`--review-file`, modelo em `src/contracts/review.py`) exige `alert_id`,
  `reviewer_id`, `decision` (`confirmed` | `dismissed`) e `decided_at` com fuso;
  `reviewer_name` e `notes` são opcionais. Resultado: `reviewed_confirmed` ou
  `reviewed_dismissed`.
- São rejeitadas e auditadas: decisão inválida, `alert_id` inexistente e segunda decisão para
  um alerta já revisado.

## Resumo (LLM)

O resumo determinístico é sempre gerado e serve de referência. Com `--llm`, o Ollama recebe
apenas os achados estruturados (sem URIs de evidência) e só reescreve o texto; ele não altera
prioridade, regras nem revisão. Se o modelo falhar ou estiver indisponível, o fluxo usa o
texto determinístico (`method = deterministic_fallback`, com o motivo).

## Auditoria (`logs/audit-fase4-AAAAMMDD.jsonl`)

Toda linha tem `ts`, `run_id` e `etapa`:

| etapa | Principais campos |
|---|---|
| `evento_aceito` | origin, event_id, patient_id, module, event_type, timestamp, content_hash |
| `evento_rejeitado` | origin, reason (`invalid_json`, `invalid_payload_type`, `schema_validation_error`, `event_id_conflict`), event_id quando houver, errors |
| `evento_duplicado_ignorado` | origin, event_id, first_origin |
| `alerta_gerado` / `sem_alerta` | alert_id, group_id, patient_id, janela, priority, rules, event_ids, evidências (event_id + uri) |
| `revisao_registrada` / `revisao_rejeitada` | alert_id, reviewer_id, decision, decided_at, event_ids, reason |
| `resumo_gerado` | group_id, alert_id, method, model, fallback_reason |
| `execucao_concluida` | contagens, group_ids, alert_ids |
