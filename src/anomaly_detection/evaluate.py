"""
Avaliação do detector de sinais vitais com anomalias injetadas.

O MIMIC-IV não traz rótulo de "anomalia", então a sensibilidade não pode ser
medida nele. Aqui são geradas séries sintéticas de pacientes estáveis e, em
parte delas, é injetado um episódio conhecido (taquicardia com hipotensão,
dessaturação, febre com taquicardia, bradicardia ou piora gradual). Mede-se:

- sensibilidade por episódio: houve aviso durante o episódio?
- atraso: quantas observações até o primeiro aviso;
- falsos positivos: observações normais marcadas como anômalas.

É um teste de funcionamento, não uma validação clínica: os episódios
sintéticos são mais limpos que os reais.

    py -m src.anomaly_detection.evaluate
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection.events import save_json
from src.anomaly_detection.vital_signs import (
    PROFILES,
    SCORED_SIGNALS,
    PopulationModel,
    VitalConfig,
    analyze_frame,
    carry_forward,
)

BASELINE = {"hr": (78, 9, 4), "rr": (16, 2, 1.2), "spo2": (97.5, 1.0, 0.7), "sbp": (124, 11, 6), "temp_c": (36.8, 0.25, 0.15)}

# Efeito do episódio no auge, somado ao padrão do paciente.
EPISODES: dict[str, dict[str, float]] = {
    "taquicardia_hipotensao": {"hr": 48, "sbp": -40, "rr": 8},
    "dessaturacao": {"spo2": -9, "rr": 10, "hr": 18},
    "febre_taquicardia": {"temp_c": 2.4, "hr": 34, "rr": 6},
    "bradicardia": {"hr": -42, "sbp": -22},
    "piora_gradual": {"hr": 36, "sbp": -32, "spo2": -6, "rr": 9},
}
LIMITS = {"hr": (25, 220), "rr": (4, 60), "spo2": (60, 100), "sbp": (50, 240), "temp_c": (33, 41.5)}


def synthetic_stays(
    patients: int, hours: int, episode_share: float, seed: int, intensity: float = 1.0
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Séries horárias de pacientes estáveis, parte delas com um episódio injetado.

    `intensity` multiplica o efeito do episódio: 1.0 é o episódio completo,
    valores menores simulam alterações mais discretas.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1)
    rows, episodes = [], []
    kinds = list(EPISODES)
    for patient in range(patients):
        stay_id = 900000 + patient
        series = {}
        for signal, (mean, between, within) in BASELINE.items():
            level = rng.normal(mean, between)
            noise = np.zeros(hours)
            for step in range(1, hours):  # ruído autocorrelacionado
                noise[step] = 0.6 * noise[step - 1] + rng.normal(0, within)
            series[signal] = level + noise

        if rng.random() < episode_share:
            kind = kinds[patient % len(kinds)]
            length = int(rng.integers(4, 7))
            begin = int(rng.integers(12, hours - length - 2))
            ramp = np.ones(length)
            if kind == "piora_gradual":
                ramp = np.linspace(0.25, 1.0, length)
            for signal, effect in EPISODES[kind].items():
                series[signal][begin : begin + length] += effect * intensity * ramp
            episodes.append({"stay_id": stay_id, "kind": kind, "begin": begin, "end": begin + length - 1})

        for step in range(hours):
            row = {"subject_id": stay_id, "hadm_id": stay_id, "stay_id": stay_id, "charttime": start + timedelta(hours=step)}
            for signal in SCORED_SIGNALS:
                low, high = LIMITS[signal]
                value = float(np.clip(series[signal][step], low, high))
                row[signal] = round(value, 1) if signal == "temp_c" else round(value)
            # A temperatura é medida a cada 4 h, como na prática.
            if step % 4:
                row["temp_c"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows), episodes


def benchmark(
    patients: int = 200, hours: int = 72, episode_share: float = 0.5, seed: int = 7, intensity: float = 1.0
) -> dict[str, Any]:
    """Roda os dois perfis de limiar sobre as mesmas séries e resume o desempenho."""
    frame, episodes = synthetic_stays(patients, hours, episode_share, seed, intensity)
    normal, _ = synthetic_stays(patients, hours, 0.0, seed + 1)  # treino só com pacientes estáveis
    in_episode = {
        (episode["stay_id"], step)
        for episode in episodes
        for step in range(episode["begin"], episode["end"] + 1)
    }
    # As 3 h seguintes ao episódio não contam como falso positivo: o padrão individual ainda está se refazendo.
    recovering = {(e["stay_id"], step) for e in episodes for step in range(e["end"] + 1, e["end"] + 4)}

    results: dict[str, Any] = {
        "patients": patients,
        "hours_per_patient": hours,
        "episodes": len(episodes),
        "seed": seed,
        "intensity": intensity,
        "profiles": {},
    }
    for name, factory in PROFILES.items():
        config: VitalConfig = factory()
        model = PopulationModel().fit(carry_forward(normal, config.carry_minutes))
        findings = analyze_frame(frame, config, model)
        first_hour = frame.groupby("stay_id")["charttime"].transform("min")
        steps = ((frame["charttime"] - first_hour).dt.total_seconds() // 3600).astype(int).tolist()
        flagged = {(int(f.meta["stay_id"]), step) for f, step in zip(findings, steps) if f.anomaly}

        by_kind: dict[str, list[bool]] = {}
        delays = []
        for episode in episodes:
            hits = [s for s in range(episode["begin"], episode["end"] + 1) if (episode["stay_id"], s) in flagged]
            by_kind.setdefault(episode["kind"], []).append(bool(hits))
            if hits:
                delays.append(hits[0] - episode["begin"])
        detected = sum(sum(values) for values in by_kind.values())
        normal_points = [key for key in zip(frame["stay_id"].tolist(), steps) if key not in in_episode and key not in recovering]
        false_alarms = sum(1 for key in normal_points if key in flagged)
        results["profiles"][name] = {
            "sensitivity": round(detected / len(episodes), 3),
            "sensitivity_by_kind": {kind: round(sum(v) / len(v), 3) for kind, v in sorted(by_kind.items())},
            "mean_delay_observations": round(float(np.mean(delays)), 2) if delays else None,
            "false_positive_rate": round(false_alarms / len(normal_points), 4),
            "false_alarms_per_patient_day": round(false_alarms / (len(normal_points) / 24), 3),
            "events_emitted": sum(1 for f in findings if f.emit),
        }
    return results


def render(results: dict[str, Any]) -> str:
    lines = [
        f"Séries sintéticas: {results['patients']} pacientes x {results['hours_per_patient']} h, "
        f"{results['episodes']} episódios injetados com intensidade {results['intensity']:g} "
        f"(semente {results['seed']}).",
        "",
        "| Perfil | Sensibilidade | Atraso médio (obs.) | Falsos positivos | Falsos alarmes/paciente-dia |",
        "|---|---|---|---|---|",
    ]
    for name, data in results["profiles"].items():
        lines.append(
            f"| {name} | {data['sensitivity']:.3f} | {data['mean_delay_observations']} | "
            f"{data['false_positive_rate'] * 100:.2f}% | {data['false_alarms_per_patient_day']} |"
        )
    lines += ["", "Sensibilidade por tipo de episódio:", ""]
    kinds = list(next(iter(results["profiles"].values()))["sensitivity_by_kind"])
    lines.append("| Perfil | " + " | ".join(kinds) + " |")
    lines.append("|---|" + "---|" * len(kinds))
    for name, data in results["profiles"].items():
        lines.append(f"| {name} | " + " | ".join(f"{data['sensitivity_by_kind'][k]:.2f}" for k in kinds) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação com anomalias injetadas")
    parser.add_argument("--patients", type=int, default=200)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--intensity", type=float, nargs="+", default=[1.0, 0.6, 0.4],
        help="intensidade dos episódios (1.0 = completo); aceita vários valores",
    )
    parser.add_argument("--output", type=Path, default=None, help="salva o resultado em JSON")
    args = parser.parse_args()
    results = [benchmark(args.patients, args.hours, seed=args.seed, intensity=value) for value in args.intensity]
    for result in results:
        print(render(result))
    if args.output:
        save_json(results, args.output)


if __name__ == "__main__":
    main()
