"""Painel da Fase 4 (`src/api/fase4.py`): rotas, revisão humana e entrega de mídia."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import fase4  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Cliente com as pastas de saída redirecionadas: o teste não toca nas revisões reais."""
    monkeypatch.setattr(fase4, "PANEL_DIR", tmp_path / "painel")
    monkeypatch.setattr(fase4, "REVIEWS_FILE", tmp_path / "painel" / "revisoes.json")
    monkeypatch.setattr(fase4, "AUDIT_FILE", tmp_path / "audit.jsonl")
    monkeypatch.setattr(fase4, "_clinical_cache", None)
    app = FastAPI()
    app.include_router(fase4.router)
    return TestClient(app)


def test_page_is_served(client):
    response = client.get("/fase4")
    assert response.status_code == 200
    assert "Monitoramento multimodal" in response.text
    for tab in ("alerta", "video", "clinico", "audio"):
        assert f'data-tab="{tab}"' in response.text


def test_status_lists_the_received_samples(client):
    state = client.get("/api/fase4/status").json()
    assert state["received_samples"] == ["P0001_audio.json", "P0001_clinical.json", "P0001_video.json"]
    assert state["reviews"] == 0


def test_alert_comes_from_the_integration_flow(client):
    data = client.get("/api/fase4/alertas").json()
    assert data["intake"] == {"events_received": 4, "accepted": 4, "rejected": 0, "duplicates": 0}
    assert data["contract_version"] == "1.1"
    (group,) = data["groups"]
    assert group["sources_received"] == ["audio", "video", "clinical"]
    alert = group["alert"]
    assert alert["priority"] == "critical"
    assert alert["review"]["status"] == "pending_medical_review"
    assert {reason["rule_id"] for reason in alert["priority_reasons"]} >= {"R1", "R2", "R3"}
    assert "diagnóstico" in data["disclaimer"]


def test_review_is_recorded_once_and_audited(client, tmp_path):
    alert_id = client.get("/api/fase4/alertas").json()["groups"][0]["alert"]["alert_id"]
    payload = {"alert_id": alert_id, "decision": "confirmed", "reviewer_id": "CRM-FICTICIO-0001", "notes": "teste"}

    recorded = client.post("/api/fase4/revisao", json=payload)
    assert recorded.status_code == 200
    assert recorded.json()["recorded"]["status"] == "reviewed_confirmed"

    review = client.get("/api/fase4/alertas").json()["groups"][0]["alert"]["review"]
    assert review["status"] == "reviewed_confirmed"
    assert review["reviewer_id"] == "CRM-FICTICIO-0001" and review["notes"] == "teste"
    assert review["decided_at"]  # horário com fuso, preenchido pelo servidor

    again = client.post("/api/fase4/revisao", json={**payload, "decision": "dismissed"})
    assert again.status_code == 409 and "alert_already_reviewed" in again.json()["detail"]
    assert len(json.loads((tmp_path / "painel" / "revisoes.json").read_text(encoding="utf-8"))) == 1

    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "revisao_registrada" in audit and "revisao_rejeitada" in audit

    assert client.delete("/api/fase4/revisao").json() == {"removed": 1}
    assert client.get("/api/fase4/alertas").json()["groups"][0]["alert"]["review"]["status"] == "pending_medical_review"


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"alert_id": "ALT-000000000000", "decision": "confirmed", "reviewer_id": "CRM-1"}, 409),  # alerta inexistente
        ({"alert_id": "errado", "decision": "confirmed", "reviewer_id": "CRM-1"}, 422),
        ({"alert_id": "ALT-000000000000", "decision": "talvez", "reviewer_id": "CRM-1"}, 422),
        ({"alert_id": "ALT-000000000000", "decision": "confirmed", "reviewer_id": ""}, 422),
    ],
)
def test_invalid_reviews_are_refused(client, payload, status):
    assert client.post("/api/fase4/revisao", json=payload).status_code == status
    assert client.get("/api/fase4/status").json()["reviews"] == 0


def test_videos_fall_back_to_the_versioned_summary(client, tmp_path, monkeypatch):
    monkeypatch.setattr(fase4, "VIDEO_OUTPUT", tmp_path / "sem-videos")
    data = client.get("/api/fase4/videos").json()
    assert data["source"] == "versioned"
    assert len(data["videos"]) == 70
    assert set(data["evaluation"]["by_split"]) == {"ajuste", "validacao"}
    first = data["videos"][0]
    assert first["video_url"] is None and first["keyframes"] == []
    assert {"name", "clip", "split", "dataset_label", "predicted_label", "events"} <= set(first)


def test_local_video_run_exposes_media(client, tmp_path, monkeypatch):
    output = tmp_path / "video"
    clip = output / "fall-01-cam0"
    clip.mkdir(parents=True)
    (clip / "anotado.webm").write_bytes(b"webm")
    (clip / "evento_1_fall_detected.jpg").write_bytes(b"jpg")
    (output / "cache").mkdir()
    (output / "cache" / "fall-01-cam0.json").write_text("{}", encoding="utf-8")
    report = {
        "evaluation": None,
        "videos": [
            {
                "video": {"name": "fall-01-cam0.mp4", "duration_s": 5.3, "fps": 30.0},
                "description": "queda", "split": "ajuste", "dataset_label": "fall", "predicted_label": "fall",
                "processed_frames": 160, "person_coverage": 0.8, "models": {"pose": "p", "objects": "o"},
                "objects": [], "tracks": [], "events": [], "annotated_video": "anotado.webm",
                "keyframes": ["evento_1_fall_detected.jpg", "sumiu.jpg"],
            }
        ],
    }
    (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(fase4, "VIDEO_OUTPUT", output)

    (video,) = client.get("/api/fase4/videos").json()["videos"]
    assert video["video_url"] == "/media/video/fall-01-cam0/anotado.webm"
    assert video["keyframes"] == ["/media/video/fall-01-cam0/evento_1_fall_detected.jpg"]  # só o que existe

    served = client.get(video["video_url"])
    assert served.status_code == 200 and served.headers["content-type"] == "video/webm"
    assert client.get(video["keyframes"][0]).headers["content-type"] == "image/jpeg"

    # Só mídia, e só de dentro da pasta de saída.
    assert client.get("/media/video/cache/fall-01-cam0.json").status_code == 404
    assert client.get("/media/video/fall-01-cam0/nao-existe.jpg").status_code == 404
    assert client.get("/media/video/..%2F..%2FREADME.md/x.jpg").status_code == 404
    assert client.get("/media/audio/..%2F..%2FREADME.md").status_code == 404


def test_clinical_events_are_served_even_without_the_series(client, monkeypatch):
    def unavailable(subject):
        raise FileNotFoundError("chartevents.csv.gz não encontrado")

    monkeypatch.setattr(fase4, "_clinical_series", unavailable)
    data = client.get("/api/fase4/clinico").json()
    assert data["patient_id"].startswith("P")
    assert {"vital_signs", "prescription_change"} <= {event["event_type"] for event in data["events"]}
    assert data["series"]["points"] == []
    assert "chartevents" in data["series"]["unavailable"]


def test_clinical_series_is_computed_once(client, monkeypatch):
    calls = []

    def series(subject):
        calls.append(subject)
        return {"profile": "icu", "points": [{"t": "2141-05-22T20:18:00", "hr": 114.0, "emit": False}]}

    monkeypatch.setattr(fase4, "_clinical_series", series)
    first = client.get("/api/fase4/clinico").json()
    client.get("/api/fase4/clinico")
    assert calls == [int(first["patient_id"].lstrip("P"))]
    assert first["series"]["points"][0]["hr"] == 114.0


def test_audio_result_and_recording(client):
    data = client.get("/api/fase4/audio").json()
    assert data["transcription"]["provider"] == "azure_speech"
    assert data["text_analysis"]["sentiment"]["label"] == "negative"
    assert "mfcc_mean" not in data["acoustic"]
    assert data["event"]["event_type"] == "audio_check" and data["event"]["anomaly_detected"] is False
    recording = client.get(data["audio_url"])
    assert recording.status_code == 200 and recording.headers["content-type"] == "audio/wav"


def test_panel_is_registered_in_the_main_server():
    from src.api.server import app

    paths = set(app.openapi()["paths"])
    assert {"/api/ask", "/api/fase4/alertas", "/api/fase4/revisao", "/api/fase4/videos"} <= paths
    main = TestClient(app)
    assert main.get("/fase4").status_code == 200
    assert 'href="/fase4"' in main.get("/").text  # o chat da Fase 3 aponta para o painel
