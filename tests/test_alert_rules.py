"""Regras determinísticas do protótipo (R0–R5)."""

from __future__ import annotations

from src.monitoring.rules import build_alert_id, evaluate_group


def _event(event_id, module, priority="medium", anomaly=True, event_type=None):
    default_type = {"audio": "vocal_distress", "video": "agitation", "clinical": "vital_signs"}
    return {
        "event_id": event_id,
        "module": module,
        "event_type": event_type or default_type[module],
        "anomaly_detected": anomaly,
        "priority": priority if anomaly else "low",
        "evidence": [{"type": "x", "uri": f"{module}://ficticio/{event_id}"}],
    }


def _group(*events, missing=None):
    received = sorted({e["module"] for e in events})
    return {
        "patient_id": "P0001",
        "window_start": "2026-11-12T13:00:00Z",
        "window_end": "2026-11-12T13:15:00Z",
        "events": list(events),
        "sources_missing": missing if missing is not None else [
            m for m in ("audio", "video", "clinical") if m not in received
        ],
    }


def _rules(alert):
    return [r["rule_id"] for r in alert["priority_reasons"]]


def test_r0_no_anomaly_means_no_alert():
    group = _group(_event("audio-a1", "audio", anomaly=False), _event("video-v1", "video", anomaly=False))
    assert evaluate_group(group) is None


def test_r1_single_modality_keeps_module_priority():
    alert = evaluate_group(_group(_event("audio-a1", "audio", "medium")))
    assert alert["priority"] == "medium"
    assert _rules(alert) == ["R1", "R5"]


def test_r2_sensitive_event_raises_to_high():
    alert = evaluate_group(_group(_event("video-v1", "video", "low", event_type="fall_detected")))
    assert alert["priority"] == "high"
    assert "R2" in _rules(alert)


def test_r3_two_modalities_raise_one_level():
    alert = evaluate_group(_group(_event("audio-a1", "audio", "low"), _event("clinical-c1", "clinical", "medium")))
    assert alert["priority"] == "high"
    r3 = next(r for r in alert["priority_reasons"] if r["rule_id"] == "R3")
    assert r3["priority_before"] == "medium" and r3["priority_after"] == "high"
    assert set(r3["event_ids"]) == {"audio-a1", "clinical-c1"}


def test_r4_three_modalities_is_critical():
    alert = evaluate_group(
        _group(_event("audio-a1", "audio", "low"), _event("video-v1", "video", "low"), _event("clinical-c1", "clinical", "low"))
    )
    assert alert["priority"] == "critical"
    assert _rules(alert) == ["R1", "R3", "R4"]


def test_non_anomalous_modality_does_not_count_as_corroboration():
    alert = evaluate_group(_group(_event("audio-a1", "audio", "medium"), _event("video-v1", "video", anomaly=False)))
    assert alert["priority"] == "medium"
    assert alert["event_ids"] == ["audio-a1"]
    assert alert["anomalous_modalities"] == ["audio"]


def test_alert_starts_pending_and_lists_evidence():
    alert = evaluate_group(_group(_event("audio-a1", "audio")))
    assert alert["review"]["status"] == "pending_medical_review"
    assert alert["review"]["reviewer_id"] is None
    assert alert["evidence"][0]["event_id"] == "audio-a1"
    assert alert["evidence"][0]["module"] == "audio"
    assert "avaliação profissional" in alert["disclaimer"]


def test_alert_id_is_deterministic_and_order_independent():
    a = build_alert_id("P0001", "2026-11-12T13:00:00Z", ["b", "a"])
    b = build_alert_id("P0001", "2026-11-12T13:00:00Z", ["a", "b"])
    assert a == b and a.startswith("ALT-")
    assert a != build_alert_id("P0002", "2026-11-12T13:00:00Z", ["a", "b"])
