"""
Estimativa de postura com YOLOv8-pose e rastreamento simples de pessoas.

O YOLOv8-pose devolve, para cada pessoa, a caixa e 17 articulações no padrão
COCO — o mesmo conjunto usado pelo OpenPose (modelo COCO), sem a instalação à
parte que o OpenPose exige. As métricas de postura deste módulo dependem só
das articulações, então outra fonte de pose pode ser usada no lugar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
INDEX = {name: position for position, name in enumerate(KEYPOINTS)}

# Ligações desenhadas no esqueleto.
SKELETON: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
)

POSE_MODEL = "yolov8n-pose.pt"


@dataclass
class Observation:
    """Uma pessoa em um quadro processado. Coordenadas no quadro original."""

    frame_index: int
    time_s: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    keypoints: list[list[float]]  # 17 x [x, y, confiança]
    track_id: int = -1
    support_label: str | None = None
    support_overlap: float = 0.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def point(self, name: str, min_conf: float = 0.0) -> tuple[float, float] | None:
        x, y, conf = self.keypoints[INDEX[name]]
        return (x, y) if conf >= min_conf else None

    def midpoint(self, left: str, right: str, min_conf: float) -> tuple[float, float] | None:
        """Ponto médio de um par de articulações; aceita só um lado se o outro falhar."""
        points = [p for p in (self.point(left, min_conf), self.point(right, min_conf)) if p]
        if not points:
            return None
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_s": round(self.time_s, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 3),
            "keypoints": [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in self.keypoints],
            "track_id": self.track_id,
            "support_label": self.support_label,
            "support_overlap": round(self.support_overlap, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(
            frame_index=int(data["frame_index"]),
            time_s=float(data["time_s"]),
            bbox=tuple(data["bbox"]),
            confidence=float(data["confidence"]),
            keypoints=[list(point) for point in data["keypoints"]],
            track_id=int(data.get("track_id", -1)),
            support_label=data.get("support_label"),
            support_overlap=float(data.get("support_overlap", 0.0)),
        )


def trunk_angle(observation: Observation, min_conf: float = 0.35) -> float | None:
    """
    Inclinação do tronco em graus: 0 = vertical, 90 = horizontal.

    Mede o segmento entre o centro dos quadris e o centro dos ombros. Devolve
    None quando ombros ou quadris não foram localizados com confiança.
    """
    shoulders = observation.midpoint("left_shoulder", "right_shoulder", min_conf)
    hips = observation.midpoint("left_hip", "right_hip", min_conf)
    if shoulders is None or hips is None:
        return None
    dx, dy = shoulders[0] - hips[0], shoulders[1] - hips[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


class PoseEstimator:
    """YOLOv8-pose (Ultralytics). O modelo é carregado no primeiro uso."""

    def __init__(self, weights: str | Path = POSE_MODEL, conf: float = 0.3, imgsz: int = 640, device: str | None = None):
        self.weights = str(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None

    @property
    def name(self) -> str:
        return Path(self.weights).stem

    def __call__(self, frame, frame_index: int, time_s: float, scale: float = 1.0) -> list[Observation]:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        result = self._model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy() / scale
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy().copy()
        keypoints[:, :, :2] /= scale
        return [
            Observation(
                frame_index=frame_index,
                time_s=time_s,
                bbox=tuple(float(v) for v in box),
                confidence=float(score),
                keypoints=[[float(x), float(y), float(c)] for x, y, c in points],
            )
            for box, score, points in zip(boxes, scores, keypoints)
        ]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    last: Observation
    missed: int = 0
    observations: list[Observation] = field(default_factory=list)


class Tracker:
    """
    Rastreador guloso: liga cada detecção à trilha mais próxima.

    Usa IoU e, quando a caixa muda muito de forma (queda), a distância entre
    os centros. Basta para cenas com poucas pessoas e câmera fixa.
    """

    def __init__(self, max_missed: int = 45, min_iou: float = 0.1, max_distance: float = 1.5):
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_distance = max_distance  # em alturas da última caixa
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Observation]) -> list[Observation]:
        candidates = []
        for track in self._tracks:
            if track.missed > self.max_missed:
                continue
            for index, detection in enumerate(detections):
                overlap = iou(track.last.bbox, detection.bbox)
                size = max(track.last.height, track.last.width, 1.0)
                distance = math.dist(track.last.center, detection.center) / size
                if overlap >= self.min_iou or distance <= self.max_distance:
                    candidates.append((overlap - 0.1 * distance, track, index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track, index in sorted(candidates, key=lambda item: -item[0]):
            if track.track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track.track_id)
            used_detections.add(index)
            detections[index].track_id = track.track_id
            track.last, track.missed = detections[index], 0
            track.observations.append(detections[index])

        for track in self._tracks:
            if track.track_id not in used_tracks:
                track.missed += 1
        for index, detection in enumerate(detections):
            if index not in used_detections:
                detection.track_id = self._next_id
                self._tracks.append(_Track(self._next_id, detection, observations=[detection]))
                self._next_id += 1
        return detections


def group_tracks(observations: list[Observation]) -> dict[int, list[Observation]]:
    """Observações agrupadas por pessoa, em ordem de tempo."""
    tracks: dict[int, list[Observation]] = {}
    for observation in sorted(observations, key=lambda o: (o.track_id, o.frame_index)):
        tracks.setdefault(observation.track_id, []).append(observation)
    return tracks
