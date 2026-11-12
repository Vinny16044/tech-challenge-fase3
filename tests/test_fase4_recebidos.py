"""
Teste de aceitação das amostras em data/fase4_recebidos/.

A pasta começa com cópias dos exemplos fictícios. Quando a Thamy (áudio) e o
Vini (vídeo e dados clínicos) substituírem os arquivos, este teste confere o
contrato v1 e o primeiro teste conjunto: P0001, mesma janela, três fontes.
"""

from __future__ import annotations

import json

import pytest

from src.contracts import MonitoringEventV1
from src.monitoring.workflow import MonitoringIntegrationGraph, load_inputs

from conftest import ROOT

RECEIVED_DIR = ROOT / "data" / "fase4_recebidos"
EXPECTED_FILES = {
    "P0001_audio.json": "audio",  # Thamy
    "P0001_video.json": "video",  # Vini
    "P0001_clinical.json": "clinical",  # Vini
}


@pytest.mark.parametrize("filename, module", sorted(EXPECTED_FILES.items()))
def test_received_sample_follows_contract(filename, module):
    path = RECEIVED_DIR / filename
    assert path.exists(), f"arquivo esperado não encontrado: {path}"

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        event = MonitoringEventV1.model_validate(item)
        assert event.module == module, f"{filename} deveria conter eventos de '{module}'"
        assert event.patient_id == "P0001", "no primeiro teste conjunto, use P0001"


def test_received_samples_consolidate_into_one_group(tmp_path):
    report = MonitoringIntegrationGraph(audit_path=tmp_path / "audit.jsonl").invoke(
        load_inputs([RECEIVED_DIR])
    )

    assert report["intake"]["rejected_events"] == []
    assert report["intake"]["accepted"] == report["intake"]["events_received"] >= 3
    assert len(report["groups"]) == 1, (
        "as amostras caíram em grupos diferentes: "
        + ", ".join(group["group_id"] for group in report["groups"])
    )
    group = report["groups"][0]
    assert group["patient_id"] == "P0001"
    assert group["sources_received"] == ["audio", "video", "clinical"]
    assert group["sources_missing"] == []
