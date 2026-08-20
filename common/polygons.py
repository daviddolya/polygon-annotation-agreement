#!/usr/bin/env python3
"""Полигоны COCO: чтение, растеризация, метрики масок (P4b).

Дополняет boxes.py: там бокс, здесь контур. Внутреннее представление —
плоский список [x1, y1, x2, y2, ...] в абсолютных пикселях, как в COCO.
Один объект может состоять из нескольких контуров: объект, разорванный
перекрытием надвое, это два контура и одна маска.

Растеризация делается через PIL, эрозия — MinFilter. Это настоящая
морфологическая эрозия квадратным элементом, так что scipy не нужен.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]


@dataclass
class Poly:
    """Один объект: класс и один или несколько контуров."""
    cls: str
    parts: list[list[float]]
    iscrowd: bool = False

    @property
    def vertices(self) -> int:
        return sum(len(p) // 2 for p in self.parts)

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[i] for p in self.parts for i in range(0, len(p), 2)]
        ys = [p[i] for p in self.parts for i in range(1, len(p), 2)]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


@dataclass
class PolyFrame:
    file_name: str
    width: int
    height: int
    polys: list[Poly] = field(default_factory=list)


def load_coco_polygons(path: str | Path, keep: set[str] | None = None,
                       drop_crowd: bool = True) -> list[PolyFrame]:
    """Читает COCO JSON, оставляя только полигональные аннотации.

    iscrowd=1 хранится как RLE, а не как контур: сопоставлять с ручной
    разметкой некорректно, поэтому по умолчанию отбрасывается.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    frames = {
        img["id"]: PolyFrame(img["file_name"], img["width"], img["height"])
        for img in data["images"]
    }
    for ann in data["annotations"]:
        cls = names.get(ann["category_id"])
        if cls is None or (keep is not None and cls not in keep):
            continue
        crowd = bool(ann.get("iscrowd", 0))
        if crowd and drop_crowd:
            continue
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue          # RLE или пусто — не полигон
        parts = [list(map(float, p)) for p in seg if len(p) >= 6]
        if not parts:
            continue
        frame = frames.get(ann["image_id"])
        if frame is not None:
            frame.polys.append(Poly(cls, parts, crowd))
    return list(frames.values())


def rasterize(poly: Poly, width: int, height: int) -> np.ndarray:
    """Контуры -> булева маска. Несколько частей объединяются в одну маску."""
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for part in poly.parts:
        points = [(part[i], part[i + 1]) for i in range(0, len(part) - 1, 2)]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.array(canvas, dtype=bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice = 2·IoU/(1+IoU). Считается напрямую, чтобы не тащить деление дважды."""
    inter = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    return 2 * inter / total if total else 0.0


def boundary_band(mask: np.ndarray, distance: int) -> np.ndarray:
    """Полоса шириной distance внутри границы маски: mask минус эрозия.

    Эрозия — MinFilter квадратом (2·distance+1). Для оценки границы этого
    достаточно; круглый элемент дал бы отличие в единицы пикселей.
    """
    if distance < 1:
        return mask
    size = 2 * distance + 1
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    eroded = np.array(img.filter(ImageFilter.MinFilter(size)), dtype=np.uint8) > 0
    return mask & ~eroded


def boundary_iou(a: np.ndarray, b: np.ndarray, distance: int) -> float:
    """Boundary IoU (arXiv:2103.16562): IoU, посчитанный только в полосе границы.

    В отличие от mask IoU одинаково строг к крупным и мелким объектам:
    сдвиг контура на несколько пикселей роняет его и там, и там.
    """
    return mask_iou(boundary_band(a, distance), boundary_band(b, distance))


def boundary_distance(width: int, height: int, ratio: float = 0.02) -> int:
    """Ширина полосы границы — доля диагонали кадра, как в статье."""
    return max(1, round(ratio * (width ** 2 + height ** 2) ** 0.5))


def match_polys(mine: list[Poly], ref: list[Poly], masks_mine: list[np.ndarray],
                masks_ref: list[np.ndarray], iou_threshold: float):
    """Жадное сопоставление по убыванию mask IoU, БЕЗ учёта класса.

    Тот же принцип, что в agreement.py для боксов: если требовать совпадения
    метки, ошибка класса распадётся на пропуск и лишний объект сразу,
    и согласие по классам считать будет не на чем.
    """
    candidates = sorted(
        ((mask_iou(masks_mine[i], masks_ref[j]), i, j)
         for i in range(len(mine)) for j in range(len(ref))),
        key=lambda t: -t[0])
    used_mine: set[int] = set()
    used_ref: set[int] = set()
    pairs = []
    for score, i, j in candidates:
        if score < iou_threshold:
            break
        if i in used_mine or j in used_ref:
            continue
        used_mine.add(i)
        used_ref.add(j)
        pairs.append((i, j, score))
    extra = [i for i in range(len(mine)) if i not in used_mine]
    missing = [j for j in range(len(ref)) if j not in used_ref]
    return pairs, extra, missing


def split_pairs(mine: list[Poly], ref: list[Poly], masks_mine: list[np.ndarray],
                masks_ref: list[np.ndarray], extra: list[int], missing: list[int],
                floor: float = 0.1):
    """Разорванные пары: один объект, посчитанный дважды.

    Если контуры разошлись сильнее порога сопоставления, пара не матчится и объект
    попадает в отчёт и как «пропущено», и как «лишнее». Это поведение порога, а не
    ошибка разметки, но по двум отдельным числам его не видно.

    Связывает оставшиеся объекты жадно по убыванию mask IoU, беря только пары выше
    `floor`. Нижняя граница нужна, чтобы мелкий эталонный объект, случайно попавший
    внутрь чужого крупного контура, не был объявлен парой: такие дают IoU около 0.01,
    а настоящие разорванные пары — от 0.2.

    Возвращает `(splits, extra, missing)`, где списки уже без вошедших в пары.
    """
    candidates = sorted(
        ((mask_iou(masks_mine[i], masks_ref[j]), i, j) for i in extra for j in missing),
        key=lambda t: -t[0])
    used_mine: set[int] = set()
    used_ref: set[int] = set()
    splits = []
    for score, i, j in candidates:
        if score < floor:
            break
        if i in used_mine or j in used_ref:
            continue
        used_mine.add(i)
        used_ref.add(j)
        splits.append((i, j, score))
    return (splits,
            [i for i in extra if i not in used_mine],
            [j for j in missing if j not in used_ref])
