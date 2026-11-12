"""
Trilha de auditoria em JSONL (uma linha por evento de auditoria).

Mantém o padrão da Fase 3 (`logs/audit-YYYYMMDD.jsonl`, campo `etapa`), mas
com arquivo próprio da Fase 4 e um `run_id` em todas as linhas, para rastrear
cada resultado até os eventos de entrada.

Responsável: Paola
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT / "logs"


def default_audit_path() -> Path:
    return DEFAULT_LOG_DIR / f"audit-fase4-{dt.date.today():%Y%m%d}.jsonl"


class AuditLogger:
    def __init__(self, path: Path | str | None = None, run_id: str = "") -> None:
        self.path = Path(path) if path else default_audit_path()
        self.run_id = run_id

    def log(self, etapa: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": self.run_id,
            "etapa": etapa,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry


def read_audit(path: Path | str) -> list[dict[str, Any]]:
    """Lê o JSONL de auditoria (útil para testes e inspeção)."""
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
