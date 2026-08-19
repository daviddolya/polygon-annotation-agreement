#!/usr/bin/env python3
"""Согласованность полигональной разметки с эталонной (P4b, шаг 3).

Главный артефакт этапа. Отвечает на три разных вопроса, которые одним
числом не отвечаются:

    mask IoU       насколько совпадает площадь. Нечувствителен к границе
                   у крупных объектов — см. arXiv:2103.16562
    Boundary IoU   насколько совпадает сама граница. Одинаково строг
                   к крупным и мелким
    Dice           та же площадь в другом виде: Dice = 2*IoU/(1+IoU).
                   Считается ради сопоставимости с чужими отчётами

Плюс согласие по классам (Cohen's kappa) на сопоставленных парах и цена
разметки в вершинах — своя против эталонной.

Сопоставление жадное по убыванию mask IoU, порог 0.5, БЕЗ учёта класса:
иначе ошибка класса распадается на пропуск и лишний объект сразу.

    .venv/bin/python annotation/polygon_agreement.py \
        --mine annotation/my_labels/instances_default.json \
        --reference ../detection-annotation-quality/data/coco/annotation/instances_val2017.json \
        --out reports/polygon_metrics.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from agreement import cohens_kappa  # noqa: E402
from polygons import (CLASSES, boundary_distance, boundary_iou, dice,  # noqa: E402
                      load_coco_polygons, mask_iou, match_polys, rasterize)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("reports/polygon_metrics.json"))
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--boundary-ratio", type=float, default=0.02,
                   help="ширина полосы границы как доля диагонали кадра")
    p.add_argument("--classes", nargs="*", default=CLASSES)
    args = p.parse_args()

    keep = set(args.classes)
    mine_frames = {f.file_name: f for f in load_coco_polygons(args.mine, keep=keep)}
    ref_frames = {f.file_name: f for f in load_coco_polygons(args.reference, keep=keep)}

    absent = sorted(set(mine_frames) - set(ref_frames))
    if absent:
        raise SystemExit(f"нет эталона для {len(absent)} кадров, например {absent[:3]}")
    if not mine_frames:
        raise SystemExit("в своей разметке нет ни одного полигона — проверь формат экспорта")

    totals = Counter()
    ious, dices, bious = [], [], []
    label_pairs: list[tuple[str, str]] = []
    per_class_iou = defaultdict(list)
    per_class_biou = defaultdict(list)
    per_class = defaultdict(Counter)
    verts_mine, verts_ref = defaultdict(list), defaultdict(list)
    examples = defaultdict(list)

    for name, frame in sorted(mine_frames.items()):
        ref = ref_frames[name]
        w, h = frame.width, frame.height
        distance = boundary_distance(w, h, args.boundary_ratio)

        masks_mine = [rasterize(p, w, h) for p in frame.polys]
        masks_ref = [rasterize(p, w, h) for p in ref.polys]
        pairs, extra, missing = match_polys(frame.polys, ref.polys,
                                            masks_mine, masks_ref,
                                            args.iou_threshold)

        totals["mine"] += len(frame.polys)
        totals["reference"] += len(ref.polys)
        totals["matched"] += len(pairs)

        for i, j, score in pairs:
            m, r = frame.polys[i], ref.polys[j]
            d = dice(masks_mine[i], masks_ref[j])
            b = boundary_iou(masks_mine[i], masks_ref[j], distance)
            ious.append(score)
            dices.append(d)
            bious.append(b)
            label_pairs.append((m.cls, r.cls))
            per_class_iou[r.cls].append(score)
            per_class_biou[r.cls].append(b)
            verts_mine[r.cls].append(m.vertices)
            verts_ref[r.cls].append(r.vertices)
            if m.cls == r.cls:
                per_class[r.cls]["matched"] += 1
            else:
                totals["mismatch"] += 1
                per_class[r.cls]["mismatching_label"] += 1
                if len(examples[f"mismatch:{r.cls}->{m.cls}"]) < 5:
                    examples[f"mismatch:{r.cls}->{m.cls}"].append(name)
        for i in extra:
            totals["extra"] += 1
            per_class[frame.polys[i].cls]["extra"] += 1
            if len(examples[f"extra:{frame.polys[i].cls}"]) < 5:
                examples[f"extra:{frame.polys[i].cls}"].append(name)
        for j in missing:
            totals["missing"] += 1
            per_class[ref.polys[j].cls]["missing"] += 1
            if len(examples[f"missing:{ref.polys[j].cls}"]) < 5:
                examples[f"missing:{ref.polys[j].cls}"].append(name)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    errors = totals["missing"] + totals["extra"] + totals["mismatch"]
    kappa = cohens_kappa(label_pairs, args.classes)

    result = {
        "iou_threshold": args.iou_threshold,
        "boundary_ratio": args.boundary_ratio,
        "frames": len(mine_frames),
        "polygons_mine": totals["mine"],
        "polygons_reference": totals["reference"],
        "matched": totals["matched"],
        "mismatching_label": totals["mismatch"],
        "missing_annotation": totals["missing"],
        "extra_annotation": totals["extra"],
        "error_rate": errors / totals["reference"] if totals["reference"] else None,
        "mean_mask_iou": mean(ious),
        "mean_dice": mean(dices),
        "mean_boundary_iou": mean(bious),
        "cohens_kappa": kappa,
        "per_class_mask_iou": {c: (mean(per_class_iou[c]) if per_class_iou[c] else None)
                               for c in args.classes},
        "per_class_boundary_iou": {c: (mean(per_class_biou[c]) if per_class_biou[c] else None)
                                   for c in args.classes},
        "per_class": {c: dict(per_class[c]) for c in args.classes},
        "vertices_mine": {c: (mean(verts_mine[c]) if verts_mine[c] else None)
                          for c in args.classes},
        "vertices_reference": {c: (mean(verts_ref[c]) if verts_ref[c] else None)
                               for c in args.classes},
        "examples": dict(examples),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"кадров {result['frames']}, полигонов у меня {totals['mine']}, "
          f"в эталоне {totals['reference']}")
    print(f"сопоставлено {totals['matched']}, из них с другой меткой {totals['mismatch']}")
    print(f"пропущено {totals['missing']}, лишних {totals['extra']}")
    print(f"mask IoU {mean(ious):.3f} | Dice {mean(dices):.3f} | "
          f"Boundary IoU {mean(bious):.3f} | kappa {kappa:.3f}")
    print("вершин на объект (мои / эталон):")
    for c in args.classes:
        if verts_mine[c]:
            print(f"  {c:<12}{mean(verts_mine[c]):>6.0f} / {mean(verts_ref[c]):.0f}")
    print(f"метрики -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
