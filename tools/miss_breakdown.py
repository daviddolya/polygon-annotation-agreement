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
from polygons import (CLASSES, load_coco_polygons, match_polys,  # noqa: E402
                      rasterize, split_pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='порог сопоставления, тот же, что в polygon_agreement.py')
    ap.add_argument('--floor', type=float, default=0.1,
                    help='ниже этого IoU объекты не считаются одной разорванной парой')
    args = ap.parse_args()

    mine = load_coco_polygons(args.mine)
    ref = {f.file_name: f for f in load_coco_polygons(args.reference, keep=set(CLASSES))}

    unseen, diverged = [], []
    for mf in mine:
        rf = ref.get(mf.file_name)
        if rf is None:
            continue
        w, h = rf.width, rf.height
        masks_mine = [rasterize(p, w, h) for p in mf.polys]
        masks_ref = [rasterize(p, w, h) for p in rf.polys]
        _, extra, missing = match_polys(mf.polys, rf.polys, masks_mine, masks_ref,
                                        args.threshold)
        # То же разложение, что в polygon_agreement.py: правило живёт в одном месте,
        # иначе два инструмента разойдутся в числах при первой же правке.
        splits, _, thin_missing = split_pairs(mf.polys, rf.polys, masks_mine, masks_ref,
                                              extra, missing, args.floor)
        for j in thin_missing:
            unseen.append((mf.file_name, rf.polys[j].cls, int(masks_ref[j].sum()), 0.0))
        for i, j, score in splits:
            diverged.append((mf.file_name, rf.polys[j].cls, int(masks_ref[j].sum()),
                             score, mf.polys[i].cls))

    rows = unseen + [(a, b, c, d) for a, b, c, d, _ in diverged]
    print(f'непокрытых эталонных объектов: {len(rows)} (порог {args.threshold})')
    print(f'  объект не размечен: {len(unseen)}')
    print(f'  разорванная пара, {args.floor} <= IoU < {args.threshold}: {len(diverged)}')
    if unseen:
        areas = sorted(r[2] for r in unseen)
        print(f'  площадь неразмеченных: медиана {areas[len(areas) // 2]} px, '
              f'мельче 500 px: {sum(1 for x in areas if x < 500)} из {len(areas)}')
        print('  классы неразмеченных:',
              collections.Counter(r[1] for r in unseen).most_common())

    by_frame = collections.Counter(r[0] for r in rows)
    print('\nкадры с наибольшим числом непокрытых:')
    for name, n in by_frame.most_common(5):
        print(f'  {name}  {n}')

    if diverged:
        print('\nразорванные пары:')
        for name, cls, area, score, my_cls in sorted(diverged, key=lambda x: x[3]):
            tag = cls if cls == my_cls else f'{cls} -> {my_cls}'
            print(f'  {name}  {tag:<24} IoU {score:.2f}  площадь {area} px')

    print('\nне размечено, по возрастанию площади:')
    for name, cls, area, _ in sorted(unseen, key=lambda x: x[2]):
        print(f'  {name}  {cls:<11} площадь {area} px')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
