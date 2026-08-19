#!/usr/bin/env python3
"""Отбор кадров под полигональную разметку (P4b, шаг 2).

Берутся кадры из уже размеченной сотни проекта P2 — те же изображения,
качать нечего. Отбор идёт по двум правилам:

  1. в кадре есть объекты классов, где полигон реально отличается от бокса
     (по замеру шага 1 это person, bicycle, motorcycle: 52-54% площади бокса);
  2. объектов немного. Полигон размечается втрое медленнее бокса, и кадр
     с восемью людьми съедает полчаса без выигрыша для метрики.

Классы и контуры наружу НЕ выводятся: разметка идёт вслепую, иначе согласие
по классам считать не на чем. Распределение печатается только с --stats.

    .venv/bin/python tools/select_polygon_frames.py \\
        --ann ../detection-annotation-quality/data/coco/annotation/instances_val2017.json \\
        --frames-dir ../detection-annotation-quality/data/subset/frames \\
        --subset ../detection-annotation-quality/data/subset/selection.json \\
        --out data/subset --count 25
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

PRIORITY = ["person", "bicycle", "motorcycle"]
CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="каталог с кадрами сотни из P2")
    p.add_argument("--subset", type=Path, required=True,
                   help="selection.json проекта P2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=25)
    p.add_argument("--max-objects", type=int, default=4,
                   help="потолок объектов в кадре: полигоны дороги")
    p.add_argument("--min-area", type=float, default=900.0,
                   help="px^2; контур объекта мельче не разглядеть")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stats", action="store_true",
                   help="напечатать эталонное распределение — только после разметки")
    args = p.parse_args()

    pool_names = set(json.loads(args.subset.read_text(encoding="utf-8"))["files"])
    data = json.loads(args.ann.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    keep = set(CLASSES)

    images = {img["id"]: img for img in data["images"]
              if img["file_name"] in pool_names}
    if not images:
        raise SystemExit("ни один кадр из selection.json не найден в аннотациях")

    per_image: dict[int, Counter] = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in images:
            continue
        cls = names.get(ann["category_id"])
        if cls not in keep or ann.get("iscrowd", 0) == 1:
            continue
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue
        if ann["bbox"][2] * ann["bbox"][3] < args.min_area:
            continue
        per_image.setdefault(ann["image_id"], Counter())[cls] += 1

    pool = [
        img_id for img_id, cnt in per_image.items()
        if sum(cnt.values()) <= args.max_objects
        and any(cnt[c] for c in PRIORITY)
    ]
    if len(pool) < args.count:
        raise SystemExit(
            f"после фильтров осталось {len(pool)} кадров, нужно {args.count}. "
            f"Ослабь --max-objects или --min-area")

    # жадный отбор: на каждом шаге тянем кадр с самым редким пока приоритетным
    # классом, иначе выборка станет наполовину из одних person
    random.Random(args.seed).shuffle(pool)
    selected, got = [], Counter({c: 0 for c in PRIORITY})
    while pool and len(selected) < args.count:
        rare = min(PRIORITY, key=lambda c: got[c])
        pick = next((i for i in pool if per_image[i][rare]), pool[0])
        pool.remove(pick)
        selected.append(pick)
        got.update({c: per_image[pick][c] for c in PRIORITY})

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    file_names = []
    for img_id in selected:
        file_name = images[img_id]["file_name"]
        shutil.copy2(args.frames_dir / file_name, frames_dir / file_name)
        file_names.append(file_name)

    manifest = {
        "source": "подмножество из selection.json проекта P2",
        "task": "polygons",
        "priority_classes": PRIORITY,
        "filters": {"max_objects": args.max_objects, "min_area_px2": args.min_area},
        "seed": args.seed,
        "count": len(file_names),
        "files": sorted(file_names),
    }
    (args.out / "selection_polygons.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"отобрано {len(file_names)} кадров -> {frames_dir}")
    print(f"манифест: {args.out / 'selection_polygons.json'}")
    if args.stats:
        print("эталонное распределение приоритетных классов:", dict(got))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
