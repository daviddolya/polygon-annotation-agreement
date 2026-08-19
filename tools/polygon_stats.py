#!/usr/bin/env python3
"""Насколько полигон отличается от бокса — на своих кадрах (P4b, шаг 1).

Считает по эталонным аннотациям COCO две вещи для каждого класса:

    полигон/бокс   доля площади бокса, которую реально занимает объект.
                   100% значит, что бокс и есть объект; 50% — что половина
                   бокса это фон, и детектор учится на нём как на объекте
    вершин         сколько точек в контуре. Прямая цена разметки: каждая
                   вершина это клик

Зависимостей нет, площадь считается формулой шнуровки (shoelace) по контуру.
Аннотации с iscrowd=1 пропускаются: там RLE-область толпы, а не полигон.

    python3 tools/polygon_stats.py \\
        --ann ../detection-annotation-quality/data/coco/annotation/instances_val2017.json \\
        --frames ../detection-annotation-quality/data/subset/selection.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]


def polygon_area(points: list[float]) -> float:
    """Площадь замкнутого контура по формуле шнуровки.

    points — плоский список [x1, y1, x2, y2, ...], как хранит COCO.
    """
    n = len(points) // 2
    total = 0.0
    for i in range(n):
        x1, y1 = points[2 * i], points[2 * i + 1]
        x2, y2 = points[(2 * i + 2) % (2 * n)], points[(2 * i + 3) % (2 * n)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", type=Path, required=True, help="instances_val2017.json")
    p.add_argument("--frames", type=Path,
                   help="selection.json; без него считается по всем кадрам")
    p.add_argument("--classes", nargs="*", default=CLASSES)
    args = p.parse_args()

    data = json.loads(args.ann.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    keep = set(args.classes)

    if args.frames:
        wanted = set(json.loads(args.frames.read_text(encoding="utf-8"))["files"])
        image_ids = {i["id"] for i in data["images"] if i["file_name"] in wanted}
        if not image_ids:
            raise SystemExit("ни один кадр из selection.json не найден в аннотациях")
    else:
        image_ids = {i["id"] for i in data["images"]}

    ratios = defaultdict(list)
    vertices = defaultdict(list)

    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            continue
        cls = names.get(ann["category_id"])
        if cls not in keep or ann.get("iscrowd", 0) == 1:
            continue
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue
        area = sum(polygon_area(part) for part in seg if len(part) >= 6)
        box_w, box_h = ann["bbox"][2], ann["bbox"][3]
        if area <= 0 or box_w * box_h <= 0:
            continue
        ratios[cls].append(area / (box_w * box_h))
        vertices[cls].append(sum(len(part) // 2 for part in seg))

    print(f"кадров {len(image_ids)}")
    print(f"{'класс':<12}{'объектов':>9}{'полигон/бокс':>14}{'вершин, медиана':>17}")
    every = []
    for cls in args.classes:
        values = ratios[cls]
        if not values:
            continue
        every += values
        median = sorted(vertices[cls])[len(vertices[cls]) // 2]
        print(f"{cls:<12}{len(values):>9}{sum(values) / len(values):>13.0%}{median:>17}")
    if every:
        print(f"{'ВСЕГО':<12}{len(every):>9}{sum(every) / len(every):>13.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
