"""
Extração: do vídeo às observações de pose e aos objetos da cena.

É a única etapa que precisa dos modelos YOLOv8. O resultado é um dicionário
serializável (`Extraction.to_dict`) que pode ser salvo em JSON, o que permite
ajustar e testar as regras de movimento sem rodar a inferência de novo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.video_analysis.objects import (
    OBJECT_MODEL,
    DetectedObject,
    ObjectDetector,
    assign_supports,
    stable_objects,
)
from src.video_analysis.pose import POSE_MODEL, Observation, PoseEstimator, Tracker
from src.video_analysis.preprocessing import ROOT, VideoInfo, iter_frames, probe

MODELS_DIR = ROOT / "models"


def resolve_weights(name: str) -> str:
    """Usa `models/<nome>` se existir; senão deixa a Ultralytics baixar o peso."""
    local = MODELS_DIR / name
    return str(local) if local.exists() else name


@dataclass
class Extraction:
    video: VideoInfo
    observations: list[Observation]
    objects: list[DetectedObject]
    processed_frames: list[int] = field(default_factory=list)
    stride: int = 1
    pose_model: str = Path(POSE_MODEL).stem
    object_model: str = Path(OBJECT_MODEL).stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "stride": self.stride,
            "pose_model": self.pose_model,
            "object_model": self.object_model,
            "processed_frames": self.processed_frames,
            "objects": [item.to_dict() for item in self.objects],
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Extraction":
        return cls(
            video=VideoInfo(**data["video"]),
            observations=[Observation.from_dict(item) for item in data["observations"]],
            objects=[DetectedObject.from_dict(item) for item in data["objects"]],
            processed_frames=list(data.get("processed_frames", [])),
            stride=int(data.get("stride", 1)),
            pose_model=data.get("pose_model", Path(POSE_MODEL).stem),
            object_model=data.get("object_model", Path(OBJECT_MODEL).stem),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Extraction":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def extract(
    path: Path,
    *,
    pose: PoseEstimator | None = None,
    detector: ObjectDetector | None = None,
    stride: int = 1,
    crop: str | None = None,
    object_every: int = 10,
    min_side: int = 480,
) -> Extraction:
    """
    Roda pose em cada quadro processado e detecção de objetos a cada `object_every`.

    Os objetos são consolidados no fim do clipe e só então cada observação
    recebe a sobreposição com a área de apoio.
    """
    pose = pose or PoseEstimator(resolve_weights(POSE_MODEL))
    detector = detector or ObjectDetector(resolve_weights(OBJECT_MODEL))
    tracker = Tracker()
    observations: list[Observation] = []
    sampled_objects: list[list[DetectedObject]] = []
    processed: list[int] = []

    for step, (frame_index, time_s, frame, scale) in enumerate(
        iter_frames(path, stride=stride, crop=crop, min_side=min_side)
    ):
        processed.append(frame_index)
        observations.extend(tracker.update(pose(frame, frame_index, time_s, scale)))
        if step % object_every == 0:
            sampled_objects.append(detector(frame, scale))

    objects = stable_objects(sampled_objects)
    assign_supports(observations, objects)
    return Extraction(
        video=probe(path, crop=crop),
        observations=observations,
        objects=objects,
        processed_frames=processed,
        stride=stride,
        pose_model=pose.name,
        object_model=detector.name,
    )
