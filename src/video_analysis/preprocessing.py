"""
Leitura e preparação dos vídeos (OpenCV).

Os clipes do UR Fall Detection Dataset trazem o mapa de profundidade e a
imagem RGB lado a lado no mesmo quadro (640x240); `crop="right"` mantém só a
metade RGB. Quadros pequenos são ampliados antes da inferência, porque os
modelos de pose perdem articulações em pessoas com poucos pixels.

Baixar para `data/video/raw/`:
    py -m src.video_analysis.preprocessing --download          # 8 clipes de ajuste, ≈ 10 MB
    py -m src.video_analysis.preprocessing --download --all    # os 70 do dataset, ≈ 93 MB
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "video" / "raw"

DATASET_NAME = "UR Fall Detection Dataset (URFD)"
DATASET_URL = "http://fenix.ur.edu.pl/~mkepski/ds/uf.html"
BASE_URL = "http://fenix.ur.edu.pl/~mkepski/ds/data/"

# Clipes de AJUSTE: 4 quedas e 4 atividades do dia a dia (ADL), usados para
# calibrar os limiares. Os outros 62 do dataset formam o conjunto de VALIDAÇÃO,
# que não participou do ajuste. O rótulo vem do nome do arquivo no dataset.
SAMPLE_CLIPS: dict[str, str] = {
    "fall-01-cam0.mp4": "queda a partir de pé",
    "fall-05-cam0.mp4": "queda a partir de pé",
    "fall-13-cam0.mp4": "queda a partir de pé, com outra pessoa sentada na cena",
    "fall-22-cam0.mp4": "queda a partir da cadeira",
    "adl-01-cam0.mp4": "agachar",
    "adl-12-cam0.mp4": "curvar-se para pegar objeto",
    "adl-28-cam0.mp4": "curvar-se para pegar objeto",
    "adl-36-cam0.mp4": "sentar e deitar na cama",
}


# O dataset completo: 30 quedas e 40 atividades do dia a dia, câmera 0.
ALL_CLIPS: tuple[str, ...] = tuple(
    [f"fall-{number:02d}-cam0.mp4" for number in range(1, 31)]
    + [f"adl-{number:02d}-cam0.mp4" for number in range(1, 41)]
)


def clip_split(name: str) -> str | None:
    """`ajuste` para os clipes usados na calibração, `validacao` para os demais do dataset."""
    base = Path(name).name
    if base in SAMPLE_CLIPS:
        return "ajuste"
    return "validacao" if base in ALL_CLIPS else None


@dataclass(frozen=True)
class VideoInfo:
    name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    crop: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def expected_label(name: str) -> str | None:
    """Rótulo do URFD pelo nome do arquivo: `fall` ou `adl`; None se não for do dataset."""
    lowered = Path(name).name.lower()
    if lowered.startswith("fall-"):
        return "fall"
    if lowered.startswith("adl-"):
        return "adl"
    return None


def download(raw_dir: Path = RAW_DIR, force: bool = False, everything: bool = False) -> list[Path]:
    """Baixa os clipes de ajuste, ou o dataset inteiro com `everything`. Pula o que já existe."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in ALL_CLIPS if everything else SAMPLE_CLIPS:
        target = raw_dir / name
        if force or not target.exists():
            print(f"Baixando {BASE_URL}{name} ...")
            urllib.request.urlretrieve(BASE_URL + name, target)  # noqa: S310
        saved.append(target)
    return saved


def _crop(frame, crop: str | None):
    if crop is None:
        return frame
    width = frame.shape[1]
    if crop == "right":
        return frame[:, width // 2 :]
    if crop == "left":
        return frame[:, : width // 2]
    raise ValueError(f"recorte desconhecido: {crop}")


def probe(path: Path, crop: str | None = None) -> VideoInfo:
    """Metadados do vídeo, já considerando o recorte."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"não foi possível abrir o vídeo: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if crop in ("left", "right"):
        width //= 2
    return VideoInfo(
        name=Path(path).name,
        fps=float(fps),
        frame_count=count,
        width=width,
        height=height,
        duration_s=round(count / fps, 3) if fps else 0.0,
        crop=crop,
    )


def iter_frames(
    path: Path, *, stride: int = 1, crop: str | None = None, min_side: int = 480
) -> Iterator[tuple[int, float, "object", float]]:
    """
    Percorre o vídeo e devolve (índice do quadro, tempo em s, imagem, escala).

    `stride` pula quadros; `min_side` amplia quadros pequenos. `escala` é o
    fator aplicado: coordenadas da inferência divididas por ela voltam ao
    quadro original.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"não foi possível abrir o vídeo: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frame = _crop(frame, crop)
                scale = 1.0
                shortest = min(frame.shape[:2])
                if shortest < min_side:
                    scale = min_side / shortest
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                yield index, index / fps, frame, scale
            index += 1
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Vídeos: {DATASET_NAME}")
    parser.add_argument("--download", action="store_true", help="baixa os 8 clipes de ajuste")
    parser.add_argument("--all", action="store_true", help="com --download, baixa os 70 clipes do dataset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    if args.download:
        download(args.raw_dir, force=args.force, everything=args.all)
    for path in sorted(args.raw_dir.glob("*.mp4")):
        info = probe(path, crop="right")
        print(f"{info.name}: {info.frame_count} quadros, {info.fps:.0f} fps, {info.width}x{info.height}, {info.duration_s} s")


if __name__ == "__main__":
    main()
