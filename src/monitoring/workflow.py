"""
Workflow LangGraph da Fase 4 — integração de áudio, vídeo e dados clínicos.

Uso offline (PowerShell, na raiz do projeto):
    py -m src.monitoring.workflow --input data/fase4 --output output/relatorio_P0001.json

Responsável: Paola (arquitetura e integração)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.monitoring.nodes import MonitoringGraphNodes
from src.monitoring.state import MonitoringState, RawInput
from src.monitoring.summarizer import SummaryLLM, build_ollama_summarizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOW_MINUTES = 15


class MonitoringIntegrationGraph:
    """Grafo: valida -> consolida -> alerta -> revisão -> resumo -> relatório."""

    def __init__(
        self,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        audit_path: str | Path | None = None,
        llm: SummaryLLM | None = None,
        model_name: str | None = None,
    ) -> None:
        if window_minutes < 1 or window_minutes > 24 * 60:
            raise ValueError("window_minutes deve estar entre 1 e 1440")
        self.window_minutes = window_minutes
        self.nodes = MonitoringGraphNodes(
            audit_path=str(audit_path) if audit_path else None,
            llm=llm,
            model_name=model_name,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MonitoringState)

        builder.add_node("validate_events", self.nodes.validate_events)
        builder.add_node("consolidate", self.nodes.consolidate)
        builder.add_node("evaluate_alerts", self.nodes.evaluate_alerts)
        builder.add_node("apply_reviews", self.nodes.apply_reviews)
        builder.add_node("summarize", self.nodes.summarize)
        builder.add_node("build_report", self.nodes.build_report)

        builder.add_edge(START, "validate_events")
        builder.add_conditional_edges(
            "validate_events",
            self.nodes.route_after_validation,
            {"consolidate": "consolidate", "build_report": "build_report"},
        )
        builder.add_edge("consolidate", "evaluate_alerts")
        builder.add_edge("evaluate_alerts", "apply_reviews")
        builder.add_edge("apply_reviews", "summarize")
        builder.add_edge("summarize", "build_report")
        builder.add_edge("build_report", END)

        return builder.compile()

    def invoke(
        self,
        raw_inputs: list[RawInput],
        review_decisions: list[Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Executa o fluxo e devolve o relatório consolidado."""
        state = self.graph.invoke(
            {
                "run_id": run_id or f"RUN-{uuid.uuid4().hex[:12]}",
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "window_minutes": self.window_minutes,
                "raw_inputs": list(raw_inputs),
                "review_decisions": list(review_decisions or []),
            }
        )
        return state["report"]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_inputs(paths: list[str | Path]) -> list[RawInput]:
    """
    Lê arquivos JSON (ou todos os *.json de uma pasta, em ordem alfabética).
    O conteúdo vai bruto para o grafo, que é quem valida — inclusive JSON
    malformado, que vira rejeição auditada em vez de erro de execução.
    """
    inputs: list[RawInput] = []
    for raw_path in paths:
        path = Path(raw_path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            # utf-8-sig: aceita arquivos salvos pelo PowerShell/Notepad com BOM.
            content = file.read_text(encoding="utf-8-sig")
            inputs.append({"origin": _display_path(file), "content": content})
    return inputs


def load_review_decisions(path: str | Path) -> list[Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else [data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fase 4 — consolidação de eventos de áudio, vídeo e dados clínicos (LangGraph)."
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=[str(ROOT / "data" / "fase4")],
        help="Arquivos JSON ou pastas com *.json. Padrão: data/fase4",
    )
    parser.add_argument(
        "--window-minutes",
        "-w",
        type=int,
        default=DEFAULT_WINDOW_MINUTES,
        help="Tamanho da janela de consolidação em minutos (padrão: 15).",
    )
    parser.add_argument("--review-file", help="JSON com decisão(ões) de revisão humana.")
    parser.add_argument("--output", "-o", help="Salva o relatório JSON neste caminho.")
    parser.add_argument("--audit-log", help="Caminho do JSONL de auditoria.")
    parser.add_argument("--run-id", help="ID da execução (padrão: gerado).")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Tenta resumir com o LLM via Ollama (padrão: offline/determinístico).",
    )
    parser.add_argument("--model", default="llama3.2:3b", help="Modelo do Ollama.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Imprime só os resumos em texto, em vez do JSON completo.",
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")  # console Windows sem UTF-8
    except AttributeError:
        pass

    graph = MonitoringIntegrationGraph(
        window_minutes=args.window_minutes,
        audit_path=args.audit_log,
        llm=build_ollama_summarizer(args.model) if args.llm else None,
        model_name=args.model if args.llm else None,
    )
    report = graph.invoke(
        load_inputs(args.input),
        review_decisions=load_review_decisions(args.review_file) if args.review_file else None,
        run_id=args.run_id,
    )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    if args.summary_only:
        for group in report["groups"]:
            print(group["summary"]["text"])
            print()
        intake = report["intake"]
        print(
            f"Eventos: {intake['events_received']} recebidos, {intake['accepted']} aceitos, "
            f"{intake['rejected']} rejeitados, {intake['duplicates']} duplicados."
        )
    else:
        print(rendered)

    if args.output:
        print(f"\nRelatório salvo em: {args.output}", file=sys.stderr)
    print(f"Auditoria (JSONL): {report['audit_log']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
