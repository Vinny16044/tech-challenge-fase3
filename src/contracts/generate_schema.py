"""
Gera `schemas/event_v1.schema.json` a partir do modelo Pydantic.

Uso (PowerShell, na raiz do projeto):
    py -m src.contracts.generate_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from src.contracts.event_v1 import json_schema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "event_v1.schema.json"


def render_schema() -> str:
    return json.dumps(json_schema(), ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(render_schema(), encoding="utf-8")
    print(f"Schema gerado em: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
