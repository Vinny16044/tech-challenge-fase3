"""Sinais vitais: faixas, desvio individual, episódios, modelo e carga dos dados."""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta

import pytest

pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.anomaly_detection import vital_signs  # noqa: E402
from src.anomaly_detection.data_loader import load_vitals  # noqa: E402
from src.anomaly_detection.events import EventContext  # noqa: E402
from src.anomaly_detection.vital_signs import (  # noqa: E402
    LAYER_BASELINE,
    LAYER_MULTIVARIATE,
    LAYER_RANGE,
    LAYER_TREND,
    PopulationModel,
    VitalConfig,
    VitalSignsMonitor,
    analyze_frame,
    band_points,
    carry_forward,
)
from src.contracts import MonitoringEventV1  # noqa: E402

START = datetime(2026, 1, 1, 8, 0)
NORMAL = {"hr": 76, "rr": 16, "spo2": 98, "sbp": 124, "temp_c": 36.8}
META = {"subject_id": 4242, "hadm_id": 1, "stay_id": 77}


def feed(monitor, observations, key="77", step_minutes=60):
    return [
        monitor.update(key, START + timedelta(minutes=step_minutes * i), values, meta=META)
        for i, values in enumerate(observations)
    ]


@pytest.mark.parametrize(
    "signal, value, points",
    [
        ("hr", 40, 3), ("hr", 45, 1), ("hr", 76, 0), ("hr", 100, 1), ("hr", 120, 2), ("hr", 131, 3),
        ("rr", 8, 3), ("rr", 16, 0), ("rr", 22, 2), ("rr", 25, 3),
        ("spo2", 91, 3), ("spo2", 93, 2), ("spo2", 95, 1), ("spo2", 97, 0),
        ("sbp", 90, 3), ("sbp", 100, 2), ("sbp", 110, 1), ("sbp", 130, 0), ("sbp", 220, 3),
        ("temp_c", 35.0, 3), ("temp_c", 36.8, 0), ("temp_c", 38.5, 1), ("temp_c", 39.5, 2),
    ],
)
def test_reference_bands(signal, value, points):
    assert band_points(signal, value) == points


def test_stable_patient_raises_no_anomaly():
    findings = feed(VitalSignsMonitor(), [NORMAL] * 12)
    assert not any(f.anomaly or f.emit for f in findings)
    assert all(f.priority == "low" for f in findings)


def test_abrupt_deterioration_is_flagged_by_range_and_baseline():
    shock = {"hr": 134, "rr": 27, "spo2": 88, "sbp": 78, "temp_c": 36.9}
    finding = feed(VitalSignsMonitor(), [NORMAL] * 10 + [shock])[-1]

    assert finding.anomaly and finding.emit
    assert finding.emit_reason == "início do episódio"
    assert finding.score == 12
    assert set(finding.red_signals) == {"hr", "rr", "spo2", "sbp"}
    assert {LAYER_RANGE, LAYER_BASELINE, LAYER_TREND} <= set(finding.layers)
    assert {d["signal"] for d in finding.deviations} >= {"hr", "sbp", "spo2"}
    assert finding.priority == "critical"


def test_change_within_range_is_low_priority_anomaly():
    """FC sobe 30 bpm mas continua na faixa: só o desvio individual dispara."""
    calm = dict(NORMAL, hr=58)
    finding = feed(VitalSignsMonitor(), [calm] * 10 + [dict(calm, hr=89)])[-1]
    assert finding.layers == [LAYER_BASELINE]
    assert finding.priority == "low" and finding.anomaly


def test_spo2_rise_is_not_a_deviation():
    low = dict(NORMAL, spo2=94)
    finding = feed(VitalSignsMonitor(), [low] * 10 + [dict(low, spo2=100)])[-1]
    assert finding.deviations == []


def test_episode_emits_on_onset_escalation_and_persistence():
    config = VitalConfig(realert_minutes=120)
    moderate = {"hr": 118, "rr": 22, "spo2": 95, "sbp": 108, "temp_c": 36.9}  # escore 6
    severe = {"hr": 135, "rr": 27, "spo2": 90, "sbp": 85, "temp_c": 36.9}
    findings = feed(VitalSignsMonitor(config), [NORMAL] * 8 + [moderate] * 4 + [severe] + [NORMAL] * 3 + [moderate])

    emitted = [(i, f.emit_reason) for i, f in enumerate(findings) if f.emit]
    assert emitted[0] == (8, "início do episódio")
    assert (10, "persistência") in emitted  # 2 h depois do primeiro aviso
    assert (12, "agravamento") in emitted
    assert findings[9].anomaly and not findings[9].emit  # suprimida: mesmo episódio
    assert emitted[-1] == (16, "início do episódio")  # novo episódio após 2 observações normais


def test_stale_measurement_is_not_carried():
    monitor = VitalSignsMonitor(VitalConfig(carry_minutes=120))
    monitor.update("1", START, {"sbp": 80, "hr": 80})
    recent = monitor.update("1", START + timedelta(minutes=60), {"hr": 80})
    stale = monitor.update("1", START + timedelta(minutes=200), {"hr": 80})
    assert recent.points["sbp"] == 3
    assert "sbp" not in stale.current


def test_icu_profile_is_less_sensitive_than_ward():
    one_red = dict(NORMAL, rr=26)  # um único sinal com 3 pontos
    ward = feed(VitalSignsMonitor(VitalConfig.ward()), [NORMAL] * 3 + [one_red])[-1]
    icu = feed(VitalSignsMonitor(VitalConfig.icu()), [NORMAL] * 3 + [one_red])[-1]
    assert ward.anomaly and LAYER_RANGE in ward.layers
    assert not icu.anomaly


def test_event_follows_contract_and_has_stable_id():
    shock = {"hr": 134, "rr": 27, "spo2": 88, "sbp": 78, "temp_c": 36.9}
    finding = feed(VitalSignsMonitor(), [NORMAL] * 10 + [shock])[-1]
    event = vital_signs.to_event(finding)
    validated = MonitoringEventV1.model_validate(event)

    assert event == vital_signs.to_event(finding)
    assert event["event_id"] == "clinical-P4242-vs-77-20260101T180000"
    assert validated.event_type == "vital_signs" and validated.anomaly_detected
    assert validated.timestamp.tzinfo is not None
    metrics = {item["metric"]: item for item in event["evidence"]}
    assert metrics["heart_rate"]["value"] == 134 and metrics["heart_rate"]["unit"] == "bpm"
    assert metrics["news2_partial_score"]["value"] == 12
    assert "diagn" not in event["finding"].lower()


def test_event_context_shifts_time_and_renames_patient():
    finding = feed(VitalSignsMonitor(), [NORMAL] * 10 + [dict(NORMAL, hr=140, sbp=80)])[-1]
    context = EventContext(patient_alias={"4242": "P0001"}, time_shift=timedelta(days=10))
    event = vital_signs.to_event(finding, context)
    assert event["patient_id"] == "P0001"
    assert event["timestamp"].startswith("2026-01-11T18:00:00")


def test_normal_observation_becomes_clinical_check():
    finding = feed(VitalSignsMonitor(), [NORMAL])[-1]
    event = vital_signs.to_event(finding)
    assert event["event_type"] == "clinical_check"
    assert event["anomaly_detected"] is False and event["priority"] == "low"
    MonitoringEventV1.model_validate(event)


def test_population_model_flags_atypical_combination():
    pytest.importorskip("sklearn")
    from src.anomaly_detection.evaluate import synthetic_stays

    normal, _ = synthetic_stays(patients=40, hours=48, episode_share=0.0, seed=3)
    model = PopulationModel().fit(carry_forward(normal, 120))
    assert model.score({"hr": 78, "rr": 16, "spo2": 98, "sbp": 124}) > 0
    assert model.score({"hr": 150, "rr": 34, "spo2": 82, "sbp": 70}) < 0
    assert model.score({"hr": 78, "rr": 16}) is None  # falta sinal

    finding = VitalSignsMonitor(VitalConfig.icu(), model).update(
        "9", START, {"hr": 128, "rr": 23, "spo2": 93, "sbp": 96}
    )
    assert LAYER_MULTIVARIATE in finding.layers


def test_batch_analysis_matches_online_monitor():
    pytest.importorskip("sklearn")
    from src.anomaly_detection.evaluate import synthetic_stays

    frame, _ = synthetic_stays(patients=6, hours=36, episode_share=1.0, seed=5)
    config = VitalConfig.ward()
    model = PopulationModel().fit(carry_forward(frame, config.carry_minutes))
    batch = analyze_frame(frame, config, model)

    monitor = VitalSignsMonitor(config, model)
    online = [
        monitor.update(str(row.stay_id), row.charttime.to_pydatetime(), {s: getattr(row, s) for s in vital_signs.SCORED_SIGNALS})
        for row in frame.sort_values(["stay_id", "charttime"]).itertuples(index=False)
    ]
    assert [(f.anomaly, f.priority, f.emit) for f in batch] == [(f.anomaly, f.priority, f.emit) for f in online]


def test_injected_episodes_are_detected():
    pytest.importorskip("sklearn")
    from src.anomaly_detection.evaluate import benchmark

    results = benchmark(patients=40, hours=48, seed=11)
    for profile in results["profiles"].values():
        assert profile["sensitivity"] >= 0.9
        assert profile["false_positive_rate"] < 0.1


def test_loader_prefers_arterial_pressure_converts_and_discards(tmp_path):
    rows = [
        # subject, hadm, stay, charttime, itemid, valuenum
        (1, 10, 100, "2150-01-01 10:00:00", 220045, 80),  # FC
        (1, 10, 100, "2150-01-01 10:00:00", 220179, 120),  # PAS não invasiva
        (1, 10, 100, "2150-01-01 10:00:00", 220050, 111),  # PAS invasiva: tem preferência
        (1, 10, 100, "2150-01-01 10:00:00", 223761, 98.6),  # °F
        (1, 10, 100, "2150-01-01 11:00:00", 220045, 0),  # artefato
        (1, 10, 100, "2150-01-01 11:00:00", 220277, 97),
        (1, 10, 100, "2150-01-01 11:00:00", 220052, 801),  # PAM implausível
        (1, 10, 100, "2150-01-01 12:00:00", 999999, 5),  # item que não interessa
    ]
    frame = pd.DataFrame(rows, columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"])
    with gzip.open(tmp_path / "chartevents.csv.gz", "wt", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)

    vitals = load_vitals(tmp_path)
    assert len(vitals) == 2
    first, second = vitals.iloc[0], vitals.iloc[1]
    assert first["sbp"] == 111 and first["hr"] == 80
    assert first["temp_c"] == pytest.approx(37.0, abs=0.01)
    assert pd.isna(second["hr"]) and pd.isna(second["map"]) and second["spo2"] == 97
    assert vitals.attrs["discarded"]["hr"] == 1 and vitals.attrs["discarded"]["map"] == 1


def test_loader_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        load_vitals(tmp_path)
