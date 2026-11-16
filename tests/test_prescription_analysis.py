"""Prescrições e movimentação entre unidades. Dados fictícios."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.anomaly_detection import movement_patterns, prescription_analysis  # noqa: E402
from src.anomaly_detection.movement_patterns import analyze_transfers, transfer_to_event  # noqa: E402
from src.anomaly_detection.prescription_analysis import (  # noqa: E402
    PrescriptionConfig,
    analyze_prescriptions,
    parse_dose,
    prepare,
)
from src.contracts import MonitoringEventV1  # noqa: E402


def order(drug, start, dose, *, hadm=500, unit="mg", route="PO", per_day=1.0, kind="MAIN", subject=42):
    return {
        "subject_id": subject,
        "hadm_id": hadm,
        "starttime": pd.Timestamp(start),
        "stoptime": pd.NaT,
        "drug_type": kind,
        "drug": drug,
        "dose_val_rx": dose,
        "dose_unit_rx": unit,
        "doses_per_24_hrs": per_day,
        "route": route,
    }


def findings_of(orders, kind=None, config=None):
    found = analyze_prescriptions(pd.DataFrame(orders), config)
    return [f for f in found if kind is None or f.kind == kind]


@pytest.mark.parametrize(
    "raw, expected",
    [("25", 25.0), ("25,000", 25000.0), ("1-2", 2.0), ("0.5-1", 1.0), (40, 40.0)],
)
def test_parse_dose(raw, expected):
    assert parse_dose(raw) == expected


@pytest.mark.parametrize("raw", ["MD to order daily dose", None, float("nan")])
def test_parse_dose_without_number(raw):
    assert math.isnan(parse_dose(raw))


def test_prepare_drops_diluents_and_unusable_rows():
    orders = prepare(
        pd.DataFrame(
            [
                order("Metoprolol Tartrate", "2150-01-01 08:00", "25"),
                order("0.9% Sodium Chloride", "2150-01-01 08:00", "1000", unit="mL", route="IV"),
                order("Sodium Chloride 0.9%  Flush", "2150-01-01 08:00", "3", unit="mL", route="IV"),
                order("5% Dextrose", "2150-01-01 08:00", "250", unit="mL", kind="BASE", route="IV"),
            ]
        )
    )
    assert orders["drug"].tolist() == ["Metoprolol Tartrate"]
    assert orders["route_group"].tolist() == ["PO"]


def test_abrupt_dose_increase_is_flagged():
    found = findings_of(
        [
            order("Metoprolol Tartrate", "2150-01-01 08:00", "25", per_day=2),
            order("Metoprolol Tartrate", "2150-01-02 08:00", "100", per_day=2),
        ],
        "dose_change",
    )
    assert len(found) == 1
    assert found[0].extra["ratio"] == 4.0 and found[0].priority == "low"
    assert "Aumento abrupto" in found[0].detail and "50 para 200 mg" in found[0].detail


def test_gradual_titration_is_not_flagged():
    found = findings_of(
        [
            order("Metoprolol Tartrate", "2150-01-01 08:00", "25"),
            order("Metoprolol Tartrate", "2150-01-02 08:00", "50"),
            order("Metoprolol Tartrate", "2150-01-03 08:00", "75"),
        ],
        "dose_change",
    )
    assert found == []


def test_high_alert_drug_raises_priority_and_reduction_is_described():
    found = findings_of(
        [
            order("Heparin", "2150-01-01 08:00", "25,000", unit="UNIT", route="IV"),
            order("Heparin", "2150-01-01 20:00", "5000", unit="UNIT", route="IV"),
        ],
        "dose_change",
    )
    assert len(found) == 1
    assert found[0].priority == "high"  # alta vigilância (medium) + mudança de 5x
    assert "Redução abrupta" in found[0].detail


def test_protocol_titrated_drug_needs_larger_change():
    doubled = [order("Insulin", "2150-01-01 08:00", "4", unit="UNIT", route="SC"), order("Insulin", "2150-01-01 12:00", "12", unit="UNIT", route="SC")]
    huge = doubled[:1] + [order("Insulin", "2150-01-01 12:00", "30", unit="UNIT", route="SC")]
    assert findings_of(doubled, "dose_change") == []
    assert len(findings_of(huge, "dose_change")) == 1


def test_different_route_or_admission_is_not_compared():
    found = findings_of(
        [
            order("Furosemide", "2150-01-01 08:00", "20", route="PO"),
            order("Furosemide", "2150-01-01 12:00", "120", route="IV"),
            order("Furosemide", "2150-01-01 12:00", "200", route="PO", hadm=501),
        ],
        "dose_change",
    )
    assert found == []


def test_vasoactive_start_and_second_agent():
    found = findings_of(
        [
            order("NORepinephrine", "2150-01-01 08:00", "8", route="IV DRIP"),
            order("NORepinephrine", "2150-01-01 14:00", "8", route="IV DRIP"),  # mesmo agente: não repete
            order("Vasopressin", "2150-01-01 18:00", "40", unit="UNIT", route="IV DRIP"),
            order("Racepinephrine", "2150-01-01 19:00", "0.5", unit="mL", route="IH"),  # inalatório
        ],
        "vasoactive_start",
    )
    assert [(f.drug, f.priority) for f in found] == [("NORepinephrine", "high"), ("Vasopressin", "critical")]
    assert "norepinephrine" in found[1].detail


def test_atypical_dose_against_population():
    population = [order("Acetaminophen", f"2150-01-{1 + i % 28:02d} 08:00", "650", hadm=600 + i) for i in range(25)]
    outlier = order("Acetaminophen", "2150-02-01 08:00", "4000", hadm=700)
    found = findings_of(population + [outlier], "atypical_dose")
    assert len(found) == 1
    assert found[0].hadm_id == 700 and found[0].priority == "medium"
    assert "6.2x a mediana" in found[0].detail

    few = findings_of(population[:5] + [outlier], "atypical_dose")
    assert few == []  # sem pedidos suficientes para ter um padrão


def test_order_burst_after_quiet_days():
    quiet = [order(f"Droga {i}", pd.Timestamp("2150-01-01 08:00") + pd.Timedelta(hours=8 * i), "10") for i in range(9)]
    burst = [order(f"Nova {i}", "2150-01-05 09:00", "10") for i in range(14)]
    found = findings_of(quiet + burst, "order_burst")
    assert len(found) == 1
    assert found[0].value == 14 and found[0].unit == "pedidos/6h"

    admission_only = [order(f"Droga {i}", "2150-01-01 08:30", "10", hadm=800) for i in range(20)]
    assert findings_of(admission_only, "order_burst") == []


def test_duplicated_identical_orders_become_one_finding():
    orders = [order("NORepinephrine", "2150-01-01 08:00", "8", route="IV DRIP")] * 3
    assert len(findings_of(orders, "vasoactive_start")) == 1


def test_prescription_event_follows_contract():
    finding = findings_of(
        [
            order("Heparin", "2150-01-01 08:00", "5000", unit="UNIT", route="IV"),
            order("Heparin", "2150-01-01 20:00", "25000", unit="UNIT", route="IV"),
        ],
        "dose_change",
    )[0]
    event = prescription_analysis.to_event(finding)
    validated = MonitoringEventV1.model_validate(event)

    assert event == prescription_analysis.to_event(finding)
    assert validated.schema_version == "1.1"
    assert validated.event_type == "prescription_change"
    assert event["patient_id"] == "P0042"
    item = event["evidence"][0]
    assert item["type"] == "prescription" and item["metric"] == "Heparin"
    assert item["value"] == 25000 and item["unit"] == "unit"
    assert "pode ser ajuste planejado" in event["finding"]


# --------------------------------------------------------------------------
# Movimentação entre unidades
# --------------------------------------------------------------------------

MICU = "Medical Intensive Care Unit (MICU)"


def transfer(unit, start, *, hadm=900, subject=7, kind="transfer"):
    return {"subject_id": subject, "hadm_id": hadm, "eventtype": kind, "careunit": unit, "intime": pd.Timestamp(start), "outtime": pd.NaT}


def test_icu_return_within_limit_is_flagged():
    frame = pd.DataFrame(
        [
            transfer("Emergency Department", "2150-01-01 00:00", kind="ED"),
            transfer(MICU, "2150-01-01 06:00", kind="admit"),
            transfer("Medicine", "2150-01-03 10:00"),
            transfer("Medicine", "2150-01-03 18:00"),  # troca de leito na mesma unidade
            transfer(MICU, "2150-01-04 20:00"),
        ]
    )
    found = analyze_transfers(frame)
    assert [f.kind for f in found] == ["icu_return"]
    assert found[0].value == 34.0 and found[0].priority == "high"

    event = transfer_to_event(found[0])
    validated = MonitoringEventV1.model_validate(event)
    assert validated.event_type == "patient_transfer"
    assert event["evidence"][0]["metric"] == "hours_since_icu_discharge"


def test_icu_return_ignores_late_return_and_brief_passage():
    late = pd.DataFrame(
        [transfer(MICU, "2150-01-01 06:00"), transfer("Medicine", "2150-01-02 06:00"), transfer(MICU, "2150-01-08 06:00")]
    )
    brief = pd.DataFrame(
        [transfer(MICU, "2150-01-01 06:00"), transfer("Medicine", "2150-01-02 06:00"), transfer(MICU, "2150-01-02 06:10")]
    )
    assert analyze_transfers(late) == []
    assert analyze_transfers(brief) == []


def test_frequent_transfers_are_reported_once():
    frame = pd.DataFrame(
        [
            transfer("Medicine", "2150-01-01 00:00"),
            transfer("Cardiac Surgery", "2150-01-01 05:00"),
            transfer("PACU", "2150-01-01 09:00"),
            transfer("Med/Surg", "2150-01-01 15:00"),
            transfer("Neurology", "2150-01-01 20:00"),
        ]
    )
    found = analyze_transfers(frame)
    assert [f.kind for f in found] == ["frequent_transfers"]
    assert found[0].priority == "low"
    assert movement_patterns.TransferConfig().frequent_count == 3
