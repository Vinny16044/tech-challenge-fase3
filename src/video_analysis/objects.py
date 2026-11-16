"""
Detecção de objetos e áreas de apoio com YOLOv8.

O detector localiza os móveis da cena (cama, sofá, cadeira). Cama e sofá são
"áreas de apoio": uma pessoa deitada sobre elas está em situação esperada; a
mesma postura fora delas — no chão — é a área crítica que vira achado.

Com câmera fixa os móveis não se movem, e a pessoa deitada costuma encobrir a
cama. Por isso as detecções de todo o clipe são reunidas em objetos estáveis
(`stable_objects`) antes de calcular a sobreposição.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.video_analysis.pose import Observation, iou

OBJECT_MODEL = "yolov8n.pt"

# Classes do COCO que interessam à cena.
SCENE_CLASSES: tuple[str, ...] = ("bed", "couch", "chair", "bench")
LYING_SUPPORTS: frozenset[str] = frozenset({"bed", "couch"})


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    frames: int = 1  # em quantos quadros amostrados o objeto apareceu

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox],
            "frames": self.frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectedObject":
        return cls(data["label"], float(data["confidence"]), tuple(data["bbox"]), int(data.get("frames", 1)))


class ObjectDetector:
    """YOLOv8 (Ultralytics) restrito às classes da cena. Carrega no primeiro uso."""

    def __init__(self, weights: str | Path = OBJECT_MODEL, conf: float = 0.25, imgsz: int = 640, device: str | None = None):
        self.weights = str(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._class_ids: list[int] | None = None

    @property
    def name(self) -> str:
        return Path(self.weights).stem

    def __call__(self, frame, scale: float = 1.0) -> list[DetectedObject]:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
            names = self._model.names
            self._class_ids = [index for index, label in names.items() if label in SCENE_CLASSES]
        result = self._model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, classes=self._class_ids, device=self.device, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy() / scale
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return [
            DetectedObject(self._model.names[int(cls)], float(score), tuple(float(v) for v in box))
            for box, score, cls in zip(boxes, scores, classes)
        ]


def stable_objects(
    per_frame: list[list[DetectedObject]], min_fraction: float = 0.3, merge_iou: float = 0.4
) -> list[DetectedObject]:
    """
    Reúne as detecções de vários quadros em objetos fixos da cena.

    Detecções da mesma classe que se sobrepõem viram um objeto, com a caixa
    mediana. Só fica o que apareceu em pelo menos `min_fraction` dos quadros
    amostrados, o que descarta detecções esporádicas.
    """
    clusters: list[dict[str, Any]] = []
    for detections in per_frame:
        for detection in detections:
            for cluster in clusters:
                if cluster["label"] == detection.label and iou(cluster["bbox"], detection.bbox) >= merge_iou:
                    cluster["items"].append(detection)
                    break
            else:
                clusters.append({"label": detection.label, "bbox": detection.bbox, "items": [detection]})

    sampled = max(len(per_frame), 1)
    objects = []
    for cluster in clusters:
        items = cluster["items"]
        if len(items) / sampled < min_fraction:
            continue
        ordered = [sorted(item.bbox[i] for item in items) for i in range(4)]
        median = tuple(values[len(values) // 2] for values in ordered)
        objects.append(
            DetectedObject(
                label=cluster["label"],
                confidence=sum(item.confidence for item in items) / len(items),
                bbox=median,
                frames=len(items),
            )
        )
    return sorted(objects, key=lambda item: (-item.frames, item.label))


def overlap_fraction(person: tuple[float, float, float, float], support: tuple[float, float, float, float]) -> float:
    """Fração da caixa da pessoa que está dentro da caixa do objeto."""
    x1, y1 = max(person[0], support[0]), max(person[1], support[1])
    x2, y2 = min(person[2], support[2]), min(person[3], support[3])
    area = (person[2] - person[0]) * (person[3] - person[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area if area > 0 else 0.0


def assign_supports(observations: list[Observation], objects: list[DetectedObject]) -> None:
    """Anota em cada observação a área de apoio (cama/sofá) mais sobreposta."""
    supports = [item for item in objects if item.label in LYING_SUPPORTS]
    for observation in observations:
        best_label, best_fraction = None, 0.0
        for support in supports:
            fraction = overlap_fraction(observation.bbox, support.bbox)
            if fraction > best_fraction:
                best_label, best_fraction = support.label, fraction
        observation.support_label = best_label
        observation.support_overlap = best_fraction
