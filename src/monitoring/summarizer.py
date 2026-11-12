"""
Resumo dos achados consolidados.

O resumo determinístico é SEMPRE gerado e é a referência. O LLM (opcional)
apenas reescreve, em linguagem natural, os achados estruturados já
calculados — ele não altera prioridade, regras, evidências ou status de
revisão. Se o LLM estiver indisponível ou falhar, o fluxo segue com o
resumo determinístico.

Responsável: Paola
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from src.monitoring.rules import DISCLAIMER

SummaryLLM = Callable[[str], str]

LLM_SYSTEM_PROMPT = (
    "Você resume achados de monitoramento hospitalar para a equipe de saúde. "
    "Use SOMENTE os dados estruturados fornecidos. Não faça diagnóstico, não "
    "sugira tratamento ou medicação e não altere a prioridade informada. "
    "Cite os event_id dos achados. Responda em português, em até 6 frases, e "
    "termine lembrando que os achados dependem de avaliação profissional."
)


def deterministic_summary(group: dict[str, Any]) -> str:
    received = ", ".join(group["sources_received"]) or "nenhuma"
    missing = ", ".join(group["sources_missing"]) or "nenhuma"
    events = group["events"]
    anomalous = [event for event in events if event["anomaly_detected"]]

    lines = [
        f"Paciente {group['patient_id']} | janela {group['window_start']} a "
        f"{group['window_end']} (UTC).",
        f"Fontes recebidas: {received}. Fontes ausentes: {missing}.",
        f"{len(events)} evento(s) recebido(s), {len(anomalous)} com anomalia.",
    ]

    for event in events:
        flag = "ANOMALIA" if event["anomaly_detected"] else "sem anomalia"
        lines.append(
            f"- [{event['module']}] {event['event_type']} ({flag}, prioridade do "
            f"módulo: {event['priority']}): {event['finding']} "
            f"[{event['event_id']}]"
        )

    alert = group.get("alert")
    if alert:
        rules = "; ".join(
            f"{reason['rule_id']} {reason['rule']}" for reason in alert["priority_reasons"]
        )
        lines.append(
            f"Alerta {alert['alert_id']}: prioridade {alert['priority']} "
            f"({rules}). Status: {alert['review']['status']}."
        )
    else:
        lines.append("Nenhum alerta gerado: não há eventos com anomalia nesta janela.")

    lines.append(DISCLAIMER)
    return "\n".join(lines)


def structured_findings(group: dict[str, Any]) -> dict[str, Any]:
    """Recorte estruturado enviado ao LLM (sem dados além do relatório)."""
    alert = group.get("alert")
    return {
        "patient_id": group["patient_id"],
        "window_start": group["window_start"],
        "window_end": group["window_end"],
        "sources_received": group["sources_received"],
        "sources_missing": group["sources_missing"],
        "events": [
            {
                key: event[key]
                for key in (
                    "event_id",
                    "module",
                    "event_type",
                    "finding",
                    "anomaly_detected",
                    "priority",
                )
            }
            for event in group["events"]
        ],
        "alert": (
            {
                "alert_id": alert["alert_id"],
                "priority": alert["priority"],
                "rules": [reason["rule"] for reason in alert["priority_reasons"]],
                "review_status": alert["review"]["status"],
            }
            if alert
            else None
        ),
    }


def summarize(
    group: dict[str, Any],
    llm: SummaryLLM | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    deterministic = deterministic_summary(group)
    result: dict[str, Any] = {
        "method": "deterministic",
        "model": None,
        "text": deterministic,
        "deterministic_text": deterministic,
        "fallback_reason": None,
    }

    if llm is None:
        return result

    prompt = (
        "Achados estruturados (JSON):\n"
        + json.dumps(structured_findings(group), ensure_ascii=False, indent=2)
    )
    try:
        text = (llm(prompt) or "").strip()
        if not text:
            raise ValueError("LLM retornou texto vazio")
    except Exception as exc:  # LLM indisponível não pode derrubar o fluxo
        result["method"] = "deterministic_fallback"
        result["fallback_reason"] = f"{type(exc).__name__}: {exc}"[:300]
        return result

    result.update({"method": "llm", "model": model_name, "text": text})
    return result


def build_ollama_summarizer(model_name: str = "llama3.2:3b") -> SummaryLLM:
    """
    Cria o resumidor via Ollama (mesmo padrão da Fase 3). Import tardio: o
    modo offline não depende de langchain-ollama nem do servidor Ollama.
    Erros de conexão/modelo ausente surgem na chamada e viram fallback.
    """

    def _call(prompt: str) -> str:
        from langchain_ollama import ChatOllama

        client = ChatOllama(
            model=model_name,
            temperature=0.0,
            num_predict=300,
            validate_model_on_init=True,
        )
        response = client.invoke([("system", LLM_SYSTEM_PROMPT), ("human", prompt)])
        content = response.content
        return content.strip() if isinstance(content, str) else str(content).strip()

    return _call
