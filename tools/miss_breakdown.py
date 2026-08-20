"""Разбор непокрытых эталонных объектов: не увиден вовсе или контур разошёлся.

Сопоставление в polygon_agreement.py даёт одно число «пропущено». Причины за ним
две, и чинятся они по-разному: объект, которого разметчик не заметил, — это
внимание, объект с разошедшимся контуром — это правило инструкции. Скрипт делит
пропуски по признаку пересечения с ближайшим своим полигоном.

    .venv/bin/python tools/miss_breakdown.py --mine ... --reference ...
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))
from polygons import CLASSES, load_coco_polygons, mask_iou, rasterize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='порог сопоставления, тот же, что в polygon_agreement.py')
    args = ap.parse_args()

    mine = load_coco_polygons(args.mine)
    ref = load_coco_polygons(args.reference, keep=set(CLASSES))
    frames = {f.file_name: f for f in mine}

    rows = []
    for rf in ref:
        mf = frames.get(rf.file_name)
        if mf is None:
            continue
        masks_mine = [rasterize(p, rf.width, rf.height) for p in mf.polys]
        for rp in rf.polys:
            rm = rasterize(rp, rf.width, rf.height)
            best = max((mask_iou(rm, m) for m in masks_mine), default=0.0)
            if best < args.threshold:
                rows.append((rf.file_name, rp.cls, int(rm.sum()), best))

    # Нулевое пересечение не годится как признак «не размечен»: мелкий объект,
    # попавший внутрь чужого крупного контура, даёт IoU около 0.01, а размечен
    # он при этом не был. Граница по IoU 0.1 разделяет случаи честнее.
    unseen = [r for r in rows if r[3] < 0.1]
    diverged = [r for r in rows if r[3] >= 0.1]
    print(f'непокрытых эталонных объектов: {len(rows)} (порог {args.threshold})')
    print(f'  объект не размечен, IoU < 0.1: {len(unseen)}')
    print(f'  контур разошёлся, 0.1 <= IoU < {args.threshold}: {len(diverged)}')
    if unseen:
        areas = sorted(r[2] for r in unseen)
        print(f'  площадь неразмеченных: медиана {areas[len(areas) // 2]} px, '
              f'мельче 500 px: {sum(1 for x in areas if x < 500)} из {len(areas)}')
    if unseen:
        print('  классы непойманных:', collections.Counter(r[1] for r in unseen).most_common())

    by_frame = collections.Counter(r[0] for r in rows)
    print('\nкадры с наибольшим числом непокрытых:')
    for name, n in by_frame.most_common(5):
        print(f'  {name}  {n}')

    print('\nсписок, по возрастанию IoU:')
    for name, cls, area, iou in sorted(rows, key=lambda x: x[3]):
        print(f'  {name}  {cls:<11} IoU {iou:.2f}  площадь {area} px')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
