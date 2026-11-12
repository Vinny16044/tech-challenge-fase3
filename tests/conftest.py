"""Fixtures da Fase 4. Todos os dados são fictícios."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLES_DIR = ROOT / "data" / "fase4"
REVIEW_EXAMPLE = ROOT / "data" / "fase4_revisao" / "P0001_decisao_exemplo.json"


def _load(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def audio_event() -> dict:
    return _load("P0001_audio.json")


@pytest.fixture
def video_event() -> dict:
    return _load("P0001_video.json")


@pytest.fixture
def clinical_event() -> dict:
    return _load("P0001_clinical.json")


@pytest.fixture
def make_event(audio_event):
    """Cria variações de um evento fictício a partir do exemplo de áudio."""

    def _make(**overrides) -> dict:
        event = copy.deepcopy(audio_event)
        event.update(overrides)
        return event

    return _make


@pytest.fixture
def audit_path(tmp_path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture
def graph(audit_path):
    from src.monitoring.workflow import MonitoringIntegrationGraph

    return MonitoringIntegrationGraph(window_minutes=15, audit_path=audit_path)


def as_inputs(*events: dict) -> list[dict]:
    return [{"origin": f"teste/{index}.json", "content": event} for index, event in enumerate(events)]
