"""
Carga dos dados clínicos: MIMIC-IV Clinical Database Demo 2.2 (PhysioNet).

Fonte: https://physionet.org/content/mimic-iv-demo/2.2/ — 100 pacientes
desidentificados, acesso aberto (Open Database License v1.0). As datas já vêm
deslocadas para o futuro pela desidentificação do próprio MIMIC.

Baixar os arquivos (≈ 6 MB) para `data/clinical/raw/`:
    py -m src.anomaly_detection.data_loader --download
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "clinical" / "raw"

DATASET_NAME = "MIMIC-IV Clinical Database Demo 2.2"
DATASET_URL = "https://physionet.org/content/mimic-iv-demo/2.2/"
BASE_URL = "https://physionet.org/files/mimic-iv-demo/2.2/"
FILES: dict[str, str] = {
    "chartevents.csv.gz": "icu/chartevents.csv.gz",
    "icustays.csv.gz": "icu/icustays.csv.gz",
    "prescriptions.csv.gz": "hosp/prescriptions.csv.gz",
    "transfers.csv.gz": "hosp/transfers.csv.gz",
}

# itemid do MIMIC-IV (icu/d_items) -> sinal.
VITAL_ITEMS: dict[int, str] = {
    220045: "hr",  # Heart Rate (bpm)
    220210: "rr",  # Respiratory Rate (insp/min)
    220277: "spo2",  # O2 saturation pulseoxymetry (%)
    220179: "nbp_sbp",  # Non Invasive Blood Pressure systolic
    220180: "nbp_dbp",
    220181: "nbp_map",
    220050: "abp_sbp",  # Arterial Blood Pressure systolic (linha arterial)
    220051: "abp_dbp",
    220052: "abp_map",
    223761: "temp_f",  # Temperature Fahrenheit
    223762: "temp_c",  # Temperature Celsius
}

SIGNALS: tuple[str, ...] = ("hr", "rr", "spo2", "sbp", "dbp", "map", "temp_c")

# Fora destas faixas o valor é tratado como artefato de sensor ou erro de
# digitação (ex.: PAM 801 mmHg, FC 0) e descartado antes da análise.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "hr": (20, 300),
    "rr": (2, 80),
    "spo2": (40, 100),
    "sbp": (30, 300),
    "dbp": (10, 200),
    "map": (20, 250),
    "temp_c": (25, 45),
}


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    """Baixa as tabelas usadas pelo módulo. Não baixa de novo o que já existe."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, remote in FILES.items():
        target = raw_dir / name
        if force or not target.exists():
            print(f"Baixando {BASE_URL}{remote} ...")
            urllib.request.urlretrieve(BASE_URL + remote, target)  # noqa: S310
        saved.append(target)
    return saved


def _require(raw_dir: Path, name: str) -> Path:
    path = Path(raw_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode: py -m src.anomaly_detection.data_loader --download"
        )
    return path


def load_vitals(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Sinais vitais em formato largo: uma linha por internação em UTI e horário.

    Colunas: subject_id, hadm_id, stay_id, charttime, hr, rr, spo2, sbp, dbp,
    map, temp_c. A pressão invasiva (linha arterial) tem preferência sobre a
    não invasiva quando as duas existem no mesmo horário. `attrs["discarded"]`
    traz quantos valores implausíveis foram descartados por sinal.
    """
    events = pd.read_csv(
        _require(raw_dir, "chartevents.csv.gz"),
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
    )
    events = events[events["itemid"].isin(VITAL_ITEMS)].dropna(subset=["valuenum"])
    events["signal"] = events["itemid"].map(VITAL_ITEMS)
    events["charttime"] = pd.to_datetime(events["charttime"])

    wide = events.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "charttime"],
        columns="signal",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    for column in VITAL_ITEMS.values():
        if column not in wide:
            wide[column] = float("nan")

    for part in ("sbp", "dbp", "map"):
        wide[part] = wide[f"abp_{part}"].fillna(wide[f"nbp_{part}"])
    wide["temp_c"] = wide["temp_c"].fillna((wide["temp_f"] - 32.0) / 1.8).round(2)

    discarded: dict[str, int] = {}
    for signal, (low, high) in PLAUSIBLE.items():
        bad = wide[signal].notna() & ~wide[signal].between(low, high)
        discarded[signal] = int(bad.sum())
        wide.loc[bad, signal] = float("nan")

    wide = wide[["subject_id", "hadm_id", "stay_id", "charttime", *SIGNALS]]
    wide = wide.dropna(subset=list(SIGNALS), how="all")
    wide = wide.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    wide.attrs["discarded"] = discarded
    return wide


def load_prescriptions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Prescrições (hosp/prescriptions) com horários convertidos."""
    frame = pd.read_csv(_require(raw_dir, "prescriptions.csv.gz"), low_memory=False)
    for column in ("starttime", "stoptime"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.dropna(subset=["starttime", "drug"]).reset_index(drop=True)


def load_transfers(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Movimentações entre unidades (hosp/transfers)."""
    frame = pd.read_csv(_require(raw_dir, "transfers.csv.gz"))
    for column in ("intime", "outtime"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.dropna(subset=["intime"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Dados clínicos: {DATASET_NAME}")
    parser.add_argument("--download", action="store_true", help="baixa as tabelas do PhysioNet")
    parser.add_argument("--force", action="store_true", help="baixa de novo mesmo se já existir")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    if args.download:
        for path in download(args.raw_dir, force=args.force):
            print(f"ok: {path} ({path.stat().st_size / 1e6:.2f} MB)")
    vitals = load_vitals(args.raw_dir)
    print(
        f"{DATASET_NAME}: {len(vitals)} observações de sinais vitais, "
        f"{vitals['stay_id'].nunique()} internações em UTI, "
        f"{vitals['subject_id'].nunique()} pacientes."
    )
    print(f"Valores implausíveis descartados: {vitals.attrs['discarded']}")


if __name__ == "__main__":
    main()
