"""Execução offline pela linha de comando, como na demonstração do README."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from conftest import REVIEW_EXAMPLE, ROOT


def run_cli(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "src.monitoring.workflow", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
    )


def test_cli_offline_demo_writes_report_and_audit(tmp_path):
    output = tmp_path / "relatorio.json"
    audit = tmp_path / "audit.jsonl"
    process = run_cli("--input", "data/fase4", "--output", str(output), "--audit-log", str(audit), "--run-id", "RUN-cli")

    assert process.returncode == 0, process.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(process.stdout) == report
    assert report["run_id"] == "RUN-cli"
    assert report["config"]["summary_mode"] == "offline"
    assert report["alerts"][0]["review_status"] == "pending_medical_review"
    assert audit.exists()


def test_cli_with_review_file(tmp_path):
    process = run_cli(
        "--input", "data/fase4/P0001_audio.json", "data/fase4/P0001_video.json", "data/fase4/P0001_clinical.json",
        "--review-file", str(REVIEW_EXAMPLE),
        "--audit-log", str(tmp_path / "audit.jsonl"),
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["groups"][0]["alert"]["review"]["status"] == "reviewed_confirmed"
