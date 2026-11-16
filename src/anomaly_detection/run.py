"""
Detecção de anomalias clínicas de ponta a ponta (MIMIC-IV Clinical Database Demo).

    py -m src.anomaly_detection.data_loader --download      # uma vez, ≈ 6 MB
    py -m src.anomaly_detection.run                         # todos os pacientes
    py -m src.anomaly_detection.run --subject 10023117      # um paciente
    py -m src.anomaly_detection.run --stream 30057454       # replay em "tempo real"

Gera em `output/clinical/`:
- `events.json`  eventos no contrato da Fase 4 (sinais vitais, prescrições, movimentações);
- `report.json` e `report.md`  taxas de aviso, exemplos e limitações.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.anomaly_detection import movement_patterns, prescription_analysis, vital_signs
from src.anomaly_detection.data_loader import (
    DATASET_NAME,
    DATASET_URL,
    RAW_DIR,
    ROOT,
    load_prescriptions,
    load_transfers,
    load_vitals,
)
from src.anomaly_detection.events import EventContext, save_json
from src.anomaly_detection.vital_signs import PROFILES, SCORED_SIGNALS, PopulationModel, VitalSignsMonitor, carry_forward

OUTPUT_DIR = ROOT / "output" / "clinical"
PRIORITY_ORDER = ("critical", "high", "medium", "low")


def analyze(raw_dir: Path = RAW_DIR, profile: str = "icu") -> dict[str, Any]:
    """Roda os três detectores sobre o dataset inteiro e devolve achados e métricas."""
    config = PROFILES[profile]()
    vitals = load_vitals(raw_dir)
    model = PopulationModel().fit(carry_forward(vitals, config.carry_minutes))
    vital_findings = vital_signs.analyze_frame(vitals, config, model)
    prescription_findings = prescription_analysis.analyze_prescriptions(load_prescriptions(raw_dir))
    transfer_findings = movement_patterns.analyze_transfers(load_transfers(raw_dir))

    stay_days = (
        vitals.groupby("stay_id")["charttime"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400).sum()
    )
    anomalous = [f for f in vital_findings if f.anomaly]
    emitted = [f for f in vital_findings if f.emit]
    metrics = {
        "dataset": {"name": DATASET_NAME, "url": DATASET_URL},
        "profile": profile,
        "vital_signs": {
            "observations": len(vital_findings),
            "patients": int(vitals["subject_id"].nunique()),
            "icu_stays": int(vitals["stay_id"].nunique()),
            "patient_days": round(float(stay_days), 1),
            "discarded_implausible": vitals.attrs.get("discarded", {}),
            "model_training_rows": model.trained_rows,
            "anomalous_observations": len(anomalous),
            "anomalous_share": round(len(anomalous) / max(len(vital_findings), 1), 4),
            "events_emitted": len(emitted),
            "events_per_patient_day": round(len(emitted) / max(float(stay_days), 1e-9), 2),
            "suppressed_by_episode": len(anomalous) - len(emitted),
            "by_layer": dict(Counter(layer for f in anomalous for layer in f.layers)),
            "by_priority": dict(Counter(f.priority for f in emitted)),
            "by_reason": dict(Counter(f.emit_reason for f in emitted)),
        },
        "prescriptions": {
            "findings": len(prescription_findings),
            "by_kind": dict(Counter(f.kind for f in prescription_findings)),
            "by_priority": dict(Counter(f.priority for f in prescription_findings)),
            "admissions_with_findings": len({f.hadm_id for f in prescription_findings}),
        },
        "transfers": {
            "findings": len(transfer_findings),
            "by_kind": dict(Counter(f.kind for f in transfer_findings)),
        },
    }
    return {
        "vital_findings": vital_findings,
        "prescription_findings": prescription_findings,
        "transfer_findings": transfer_findings,
        "metrics": metrics,
        "model": model,
        "config": config,
        "vitals": vitals,
    }


def build_events(result: dict[str, Any], context: EventContext | None = None, subjects: set[int] | None = None) -> list[dict[str, Any]]:
    """Eventos do contrato para os achados (opcionalmente só de alguns pacientes)."""
    context = context or EventContext()

    def wanted(subject_id: int) -> bool:
        return subjects is None or int(subject_id) in subjects

    events = [
        vital_signs.to_event(f, context) for f in result["vital_findings"] if f.emit and wanted(f.meta["subject_id"])
    ]
    events += [
        prescription_analysis.to_event(f, context) for f in result["prescription_findings"] if wanted(f.subject_id)
    ]
    events += [
        movement_patterns.transfer_to_event(f, context) for f in result["transfer_findings"] if wanted(f.subject_id)
    ]
    return sorted(events, key=lambda e: (e["timestamp"], e["event_id"]))


def render_markdown(metrics: dict[str, Any], events: list[dict[str, Any]]) -> str:
    """Relatório em Markdown com as taxas de aviso e exemplos de anomalias."""
    vitals, rx, transfers = metrics["vital_signs"], metrics["prescriptions"], metrics["transfers"]

    def table(counts: dict[str, int], order: tuple[str, ...] | None = None) -> list[str]:
        keys = [k for k in (order or sorted(counts, key=lambda k: -counts[k])) if k in counts]
        return ["| " + " | ".join(str(k) for k in keys) + " |", "|" + "---|" * len(keys), "| " + " | ".join(str(counts[k]) for k in keys) + " |"]

    lines = [
        "# Relatório automático — anomalias em dados clínicos",
        "",
        "> Achados automatizados para avaliação profissional. Não constituem diagnóstico.",
        "",
        f"Dataset: [{metrics['dataset']['name']}]({metrics['dataset']['url']}). Perfil de limiares: `{metrics['profile']}`.",
        "",
        "## Sinais vitais",
        "",
        f"- {vitals['observations']} observações de {vitals['patients']} pacientes em {vitals['icu_stays']} internações "
        f"em UTI ({vitals['patient_days']} pacientes-dia).",
        f"- Valores implausíveis descartados como artefato: {sum(vitals['discarded_implausible'].values())}.",
        f"- Isolation Forest treinado com {vitals['model_training_rows']} observações completas.",
        f"- Observações anômalas: {vitals['anomalous_observations']} ({vitals['anomalous_share'] * 100:.1f}%).",
        f"- Eventos emitidos: {vitals['events_emitted']} ({vitals['events_per_patient_day']} por paciente-dia); "
        f"{vitals['suppressed_by_episode']} observações anômalas suprimidas por pertencerem a um episódio já avisado.",
        "",
        "Observações anômalas por camada (uma observação pode acionar mais de uma):",
        "",
        *table(vitals["by_layer"]),
        "",
        "Eventos por prioridade sugerida:",
        "",
        *table(vitals["by_priority"], PRIORITY_ORDER),
        "",
        "## Prescrições",
        "",
        f"{rx['findings']} alterações sinalizadas em {rx['admissions_with_findings']} internações.",
        "",
        *table(rx["by_kind"]),
        "",
        "## Movimentação entre unidades",
        "",
        f"{transfers['findings']} padrões sinalizados.",
        "",
        *(table(transfers["by_kind"]) if transfers["by_kind"] else ["Nenhum."]),
        "",
        "## Exemplos de anomalias detectadas",
        "",
    ]
    shown: Counter = Counter()
    for event in sorted(events, key=lambda e: PRIORITY_ORDER.index(e["priority"])):
        key = (event["event_type"], event["source"]["system"])
        if shown[key] >= 3 or not event["anomaly_detected"]:
            continue
        shown[key] += 1
        lines.append(
            f"- `{event['event_type']}` · **{event['priority']}** · paciente {event['patient_id']} · "
            f"{event['timestamp']}: {event['finding']}"
        )
    return "\n".join(lines) + "\n"


def stream(result: dict[str, Any], stay_id: int, delay: float, context: EventContext) -> None:
    """Reapresenta uma internação observação a observação, como num monitor em tempo real."""
    vitals = result["vitals"]
    stay = vitals[vitals["stay_id"] == stay_id]
    if stay.empty:
        raise SystemExit(f"internação {stay_id} não encontrada")
    monitor = VitalSignsMonitor(result["config"], result["model"])
    print(f"Internação {stay_id}: {len(stay)} observações. Perfil {result['metrics']['profile']}.\n")
    for row in stay.itertuples(index=False):
        values = {signal: getattr(row, signal) for signal in SCORED_SIGNALS}
        finding = monitor.update(
            str(stay_id),
            row.charttime.to_pydatetime(),
            values,
            meta={"subject_id": int(row.subject_id), "hadm_id": int(row.hadm_id), "stay_id": int(stay_id)},
        )
        shown = "  ".join(
            f"{vital_signs.LABELS[s]} {finding.measured[s]:g}" for s in SCORED_SIGNALS if s in finding.measured
        )
        print(f"{row.charttime}  {shown:<52} escore {finding.score}")
        if finding.emit:
            event = vital_signs.to_event(finding, context)
            print(f"    >>> ALERTA [{event['priority'].upper()}] {event['event_id']}\n        {event['finding']}")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Anomalias em dados clínicos — Fase 4")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="icu", help="limiares: icu (padrão) ou ward")
    parser.add_argument("--subject", type=int, nargs="*", help="subject_id do MIMIC-IV; padrão: todos")
    parser.add_argument("--stream", type=int, metavar="STAY_ID", help="replay de uma internação em tempo real")
    parser.add_argument("--delay", type=float, default=0.05, help="pausa entre observações no replay (s)")
    args = parser.parse_args()

    started = time.time()
    result = analyze(args.raw_dir, args.profile)
    context = EventContext()
    if args.stream is not None:
        stream(result, args.stream, args.delay, context)
        return

    subjects = set(args.subject) if args.subject else None
    events = build_events(result, context, subjects)
    metrics = result["metrics"]
    metrics["events_written"] = len(events)
    metrics["subjects_filter"] = sorted(subjects) if subjects else None
    save_json(events, args.output / "events.json")
    save_json(metrics, args.output / "report.json")
    (args.output / "report.md").write_text(render_markdown(metrics, events), encoding="utf-8")

    vitals = metrics["vital_signs"]
    print(
        f"{DATASET_NAME}: {vitals['observations']} observações, {vitals['anomalous_share'] * 100:.1f}% anômalas, "
        f"{vitals['events_emitted']} eventos de sinais vitais ({vitals['events_per_patient_day']}/paciente-dia)."
    )
    print(f"Prescrições: {metrics['prescriptions']['findings']} alterações. Movimentação: {metrics['transfers']['findings']}.")
    print(f"{len(events)} evento(s) em {args.output / 'events.json'} ({time.time() - started:.1f} s).")


if __name__ == "__main__":
    main()
