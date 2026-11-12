"""Fluxo LangGraph completo da Fase 4 com dados fictícios."""

from __future__ import annotations

import copy
import json

import pytest

from src.monitoring.audit import read_audit
from src.monitoring.workflow import MonitoringIntegrationGraph, load_inputs

from conftest import EXAMPLES_DIR, REVIEW_EXAMPLE


def inputs(*events):
    return [{"origin": f"teste/{i}.json", "content": e} for i, e in enumerate(events)]


def etapas(audit_path):
    return [entry["etapa"] for entry in read_audit(audit_path)]


# --------------------------------------------------------------------------- #
# Fluxo completo
# --------------------------------------------------------------------------- #
def test_full_flow_with_three_modalities(graph, audit_path):
    report = graph.invoke(load_inputs([EXAMPLES_DIR]), run_id="RUN-teste")

    assert report["intake"]["accepted"] == 3
    assert report["intake"]["rejected"] == 0
    assert report["sources_received"] == ["audio", "video", "clinical"]
    assert report["sources_missing"] == []
    assert len(report["groups"]) == 1

    group = report["groups"][0]
    assert group["patient_id"] == "P0001"
    assert group["window_start"] == "2026-11-12T13:00:00Z"
    assert group["window_end"] == "2026-11-12T13:15:00Z"
    assert group["status"] == "alert_pending_review"

    # Origem e ID de cada evento preservados no resultado.
    origins = {e["event_id"]: e["origin"] for e in group["events"]}
    assert origins == {
        "audio-P0001-rec0007-seg0042": "data/fase4/P0001_audio.json",
        "video-P0001-cam12-clip0315": "data/fase4/P0001_video.json",
        "clinical-P0001-obs5521": "data/fase4/P0001_clinical.json",
    }
    assert all(e["source"]["system"].startswith("fase4-") for e in group["events"])

    alert = group["alert"]
    assert alert["priority"] == "critical"
    assert [r["rule_id"] for r in alert["priority_reasons"]] == ["R1", "R3", "R4"]
    assert set(alert["event_ids"]) == set(origins)
    assert {item["event_id"] for item in alert["evidence"]} == set(origins)
    assert alert["review"]["status"] == "pending_medical_review"

    assert group["summary"]["method"] == "deterministic"
    assert "ALT-" in group["summary"]["text"]
    assert "avaliação profissional" in group["summary"]["text"]
    assert report["alerts"] == [
        {
            "alert_id": alert["alert_id"],
            "patient_id": "P0001",
            "window_start": "2026-11-12T13:00:00Z",
            "priority": "critical",
            "review_status": "pending_medical_review",
        }
    ]

    audit = read_audit(audit_path)
    assert all(entry["run_id"] == "RUN-teste" for entry in audit)
    assert etapas(audit_path).count("evento_aceito") == 3
    generated = [e for e in audit if e["etapa"] == "alerta_gerado"]
    assert generated[0]["alert_id"] == alert["alert_id"]
    assert set(generated[0]["event_ids"]) == set(origins)
    assert etapas(audit_path)[-1] == "execucao_concluida"


def test_report_is_json_serializable(graph):
    report = graph.invoke(load_inputs([EXAMPLES_DIR]))
    json.dumps(report, ensure_ascii=False)


def test_alert_id_is_stable_across_runs_and_input_order(audio_event, video_event, clinical_event, audit_path):
    g = MonitoringIntegrationGraph(audit_path=audit_path)
    a = g.invoke(inputs(audio_event, video_event, clinical_event))
    b = g.invoke(inputs(clinical_event, audio_event, video_event))
    assert a["alerts"][0]["alert_id"] == b["alerts"][0]["alert_id"]
    assert a["run_id"] != b["run_id"]


# --------------------------------------------------------------------------- #
# JSON inválido / evento inválido
# --------------------------------------------------------------------------- #
def test_malformed_json_is_rejected_and_flow_continues(graph, audit_path, audio_event):
    report = graph.invoke(
        [
            {"origin": "teste/quebrado.json", "content": '{"schema_version": "1.0", "event_id": '},
            {"origin": "teste/audio.json", "content": json.dumps(audio_event)},
        ]
    )
    assert report["intake"]["accepted"] == 1
    rejected = report["intake"]["rejected_events"]
    assert rejected[0]["origin"] == "teste/quebrado.json"
    assert rejected[0]["reason"] == "invalid_json"
    assert report["groups"][0]["sources_received"] == ["audio"]

    audit = read_audit(audit_path)
    rejection = next(e for e in audit if e["etapa"] == "evento_rejeitado")
    assert rejection["reason"] == "invalid_json"
    assert rejection["origin"] == "teste/quebrado.json"


def test_schema_invalid_event_is_rejected_with_details(graph, make_event):
    bad = make_event(event_id="audio-P0001-sem-fuso", timestamp="2026-11-12T10:03:20")
    report = graph.invoke(inputs(bad))
    assert report["intake"]["accepted"] == 0
    item = report["intake"]["rejected_events"][0]
    assert item["reason"] == "schema_validation_error"
    assert item["event_id"] == "audio-P0001-sem-fuso"
    assert any(err["loc"] == "timestamp" for err in item["errors"])
    assert report["groups"] == []
    assert report["alerts"] == []


def test_non_object_payload_is_rejected(graph):
    report = graph.invoke([{"origin": "teste/numero.json", "content": "42"}])
    assert report["intake"]["rejected_events"][0]["reason"] == "invalid_payload_type"


def test_batch_file_with_list_of_events(graph, audio_event, video_event):
    report = graph.invoke([{"origin": "teste/lote.json", "content": json.dumps([audio_event, video_event])}])
    assert report["intake"]["accepted"] == 2
    assert {e["origin"] for e in report["groups"][0]["events"]} == {"teste/lote.json#0", "teste/lote.json#1"}


def test_empty_input_produces_empty_report(graph):
    report = graph.invoke([])
    assert report["intake"]["events_received"] == 0
    assert report["groups"] == []
    assert report["sources_missing"] == ["audio", "video", "clinical"]


# --------------------------------------------------------------------------- #
# Duplicados
# --------------------------------------------------------------------------- #
def test_identical_resend_is_ignored_as_duplicate(graph, audit_path, audio_event):
    report = graph.invoke(inputs(audio_event, copy.deepcopy(audio_event)))
    assert report["intake"]["accepted"] == 1
    assert report["intake"]["duplicates"] == 1
    dup = report["intake"]["duplicate_events"][0]
    assert dup["event_id"] == audio_event["event_id"]
    assert dup["first_origin"] == "teste/0.json"
    assert len(report["groups"][0]["events"]) == 1
    assert "evento_duplicado_ignorado" in etapas(audit_path)


def test_same_event_id_with_different_content_is_conflict(graph, audio_event):
    changed = copy.deepcopy(audio_event)
    changed["finding"] = "Conteúdo diferente com o mesmo ID (fictício)."
    report = graph.invoke(inputs(audio_event, changed))
    assert report["intake"]["accepted"] == 1
    assert report["intake"]["rejected_events"][0]["reason"] == "event_id_conflict"
    assert report["groups"][0]["events"][0]["finding"] == audio_event["finding"]


# --------------------------------------------------------------------------- #
# Pacientes e janelas diferentes
# --------------------------------------------------------------------------- #
def test_different_patients_are_not_mixed(graph, audio_event, video_event):
    other = copy.deepcopy(video_event)
    other.update(patient_id="P0002", event_id="video-P0002-cam03-clip0001")
    report = graph.invoke(inputs(audio_event, other))
    assert [g["patient_id"] for g in report["groups"]] == ["P0001", "P0002"]
    assert all(len(g["events"]) == 1 for g in report["groups"])
    # Cada paciente tem só uma modalidade: sem corroboração entre pacientes.
    assert all("R3" not in [r["rule_id"] for r in g["alert"]["priority_reasons"]] for g in report["groups"])


def test_different_windows_are_not_mixed(graph, audio_event, clinical_event):
    later = copy.deepcopy(clinical_event)
    later["timestamp"] = "2026-11-12T10:48:00-03:00"
    report = graph.invoke(inputs(audio_event, later))
    assert [g["window_start"] for g in report["groups"]] == ["2026-11-12T13:00:00Z", "2026-11-12T13:45:00Z"]
    assert all(g["alert"]["priority"] == "medium" for g in report["groups"])


def test_window_size_is_configurable(audio_event, clinical_event, audit_path):
    later = copy.deepcopy(clinical_event)
    later["timestamp"] = "2026-11-12T10:48:00-03:00"
    report = MonitoringIntegrationGraph(window_minutes=60, audit_path=audit_path).invoke(inputs(audio_event, later))
    assert len(report["groups"]) == 1
    assert report["config"]["window_minutes"] == 60
    assert report["groups"][0]["alert"]["priority"] == "high"  # R3: duas modalidades


def test_invalid_window_size():
    with pytest.raises(ValueError):
        MonitoringIntegrationGraph(window_minutes=0)


# --------------------------------------------------------------------------- #
# Modalidade ausente / sem anomalia
# --------------------------------------------------------------------------- #
def test_missing_modality_is_reported(graph, audio_event, clinical_event):
    report = graph.invoke(inputs(audio_event, clinical_event))
    group = report["groups"][0]
    assert group["sources_received"] == ["audio", "clinical"]
    assert group["sources_missing"] == ["video"]
    assert report["sources_missing"] == ["video"]
    r5 = next(r for r in group["alert"]["priority_reasons"] if r["rule_id"] == "R5")
    assert "video" in r5["description"]
    assert "Fontes ausentes: video" in group["summary"]["text"]


def test_events_without_anomaly_produce_no_alert(graph, audit_path, make_event):
    normal = make_event(
        event_id="audio-P0001-rec0008-seg0001",
        event_type="audio_check",
        finding="Ambiente sonoro dentro do padrão (fictício).",
        anomaly_detected=False,
        priority="low",
    )
    report = graph.invoke(inputs(normal))
    group = report["groups"][0]
    assert group["alert"] is None
    assert group["status"] == "no_alert"
    assert report["alerts"] == []
    assert "Nenhum alerta gerado" in group["summary"]["text"]
    assert "sem_alerta" in etapas(audit_path)
    assert "alerta_gerado" not in etapas(audit_path)


# --------------------------------------------------------------------------- #
# Revisão humana
# --------------------------------------------------------------------------- #
def test_alert_stays_pending_without_decision(graph):
    report = graph.invoke(load_inputs([EXAMPLES_DIR]))
    review = report["groups"][0]["alert"]["review"]
    assert review == {
        "status": "pending_medical_review",
        "reviewer_id": None,
        "reviewer_name": None,
        "decision": None,
        "decided_at": None,
        "notes": None,
    }
    assert report["review_decisions"] == []


def test_review_decision_is_recorded_with_reviewer_and_time(graph, audit_path):
    decision = json.loads(REVIEW_EXAMPLE.read_text(encoding="utf-8"))
    report = graph.invoke(load_inputs([EXAMPLES_DIR]), review_decisions=[decision])

    alert = report["groups"][0]["alert"]
    assert alert["alert_id"] == decision["alert_id"], "exemplo de decisão desatualizado"
    assert alert["review"]["status"] == "reviewed_confirmed"
    assert alert["review"]["reviewer_id"] == "CRM-FICTICIO-0001"
    assert alert["review"]["decided_at"] == "2026-11-12T10:20:00-03:00"
    assert report["groups"][0]["status"] == "alert_reviewed"
    assert report["review_decisions"][0]["outcome"] == "recorded"

    recorded = next(e for e in read_audit(audit_path) if e["etapa"] == "revisao_registrada")
    assert recorded["alert_id"] == decision["alert_id"]
    assert recorded["reviewer_id"] == "CRM-FICTICIO-0001"
    assert set(recorded["event_ids"]) == set(alert["event_ids"])


def test_invalid_or_unknown_review_decisions_are_rejected(graph, audit_path):
    decisions = [
        {"alert_id": "ALT-000000000000", "reviewer_id": "CRM-X", "decision": "confirmed",
         "decided_at": "2026-11-12T10:20:00-03:00"},
        {"alert_id": "ALT-8935b56efce5", "reviewer_id": "CRM-X", "decision": "confirmed",
         "decided_at": "2026-11-12T10:20:00"},  # sem fuso
    ]
    report = graph.invoke(load_inputs([EXAMPLES_DIR]), review_decisions=decisions)
    reasons = [r["reason"] for r in report["review_decisions"]]
    assert reasons == ["alert_not_found", "invalid_review_decision"]
    assert report["groups"][0]["alert"]["review"]["status"] == "pending_medical_review"
    assert etapas(audit_path).count("revisao_rejeitada") == 2


def test_second_decision_for_same_alert_is_rejected(graph):
    decision = json.loads(REVIEW_EXAMPLE.read_text(encoding="utf-8"))
    second = dict(decision, decision="dismissed", reviewer_id="CRM-FICTICIO-0002")
    report = graph.invoke(load_inputs([EXAMPLES_DIR]), review_decisions=[decision, second])
    assert [r["outcome"] for r in report["review_decisions"]] == ["recorded", "rejected"]
    assert report["review_decisions"][1]["reason"] == "alert_already_reviewed"
    assert report["groups"][0]["alert"]["review"]["decision"] == "confirmed"


# --------------------------------------------------------------------------- #
# Resumo com LLM (simulado) e fallback
# --------------------------------------------------------------------------- #
def test_llm_summary_uses_only_structured_findings(audit_path):
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Resumo fictício gerado pelo LLM."

    report = MonitoringIntegrationGraph(audit_path=audit_path, llm=fake_llm, model_name="fake").invoke(
        load_inputs([EXAMPLES_DIR])
    )
    summary = report["groups"][0]["summary"]
    assert summary["method"] == "llm"
    assert summary["text"] == "Resumo fictício gerado pelo LLM."
    assert "ALT-" in summary["deterministic_text"]
    assert report["groups"][0]["alert"]["priority"] == "critical"  # LLM não altera regras
    assert "audio-P0001-rec0007-seg0042" in prompts[0]
    assert "audio://" not in prompts[0]  # evidências brutas não vão ao LLM


def test_unavailable_llm_falls_back_to_deterministic(audit_path):
    def broken_llm(prompt: str) -> str:
        raise ConnectionError("Ollama indisponível (simulado)")

    report = MonitoringIntegrationGraph(audit_path=audit_path, llm=broken_llm, model_name="fake").invoke(
        load_inputs([EXAMPLES_DIR])
    )
    summary = report["groups"][0]["summary"]
    assert summary["method"] == "deterministic_fallback"
    assert "ConnectionError" in summary["fallback_reason"]
    assert summary["text"] == summary["deterministic_text"]
    assert report["groups"][0]["alert"]["review"]["status"] == "pending_medical_review"
    resumo = next(e for e in read_audit(audit_path) if e["etapa"] == "resumo_gerado")
    assert resumo["method"] == "deterministic_fallback"
