#!/usr/bin/env python3
"""Парные картинки «эталон | своя разметка» для README (P4b, шаг 5).

render_polygons.py накладывает обе разметки на один кадр — так удобно искать
расхождение глазами, но нечитаемо как витрина: контуры лежат друг на друге и
не видно, какое решение принял разметчик. Здесь кадр показан дважды, рядом,
и дефекты подписаны прямо у объектов.

Подписи английские намеренно: картинки идут в README публичного репозитория.

    .venv/bin/python tools/render_pairs.py \
        --mine annotation/my_labels/instances_default.json \
        --reference ../detection-annotation-quality/data/coco/annotation/instances_val2017.json \
        --images data/subset/frames --out reports/pairs
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from polygons import (CLASSES, load_coco_polygons, match_polys,  # noqa: E402
                      rasterize, split_pairs)
from render_polygons import MINE_COLOR, REF_COLOR, draw_poly, load_font  # noqa: E402

MISS_COLOR = (214, 39, 40)      # красный: объект есть у одного и нет у другого
WEAK_COLOR = (230, 159, 0)      # янтарный: пара нашлась, но контур разошёлся
SPLIT_COLOR = (148, 103, 189)   # фиолетовый: пара разорвана порогом
BAR_BG = (255, 255, 255)
BAR_FG = (0, 0, 0)
TITLE_H = 26
FOOT_H = 26
GUTTER = 8
MIN_MARK = 12                   # рамка мельче этого не видна глазом
LABEL_PAD = 2


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def mark_box(poly, width, height):
    """Рамка вокруг объекта, растянутая до различимого глазом размера."""
    x, y, w, h = poly.bbox()
    cx, cy = x + w / 2, y + h / 2
    w, h = max(w, MIN_MARK), max(h, MIN_MARK)
    x0, y0 = max(0, cx - w / 2 - 2), max(0, cy - h / 2 - 2)
    x1, y1 = min(width, cx + w / 2 + 2), min(height, cy + h / 2 + 2)
    return int(x0), int(y0), int(x1), int(y1)


def place_label(draw, text, box, occupied, panel, font):
    """Ищет для подписи свободное место рядом с рамкой.

    На кадрах с толпой мелких объектов подписи неизбежно налезают друг на
    друга, поэтому позиция подбирается перебором, а не берётся фиксированной.
    """
    tw, th = text_size(draw, text, font)
    w, h = tw + 2 * LABEL_PAD, th + 2 * LABEL_PAD + 2
    cands = [(box[0], box[1] - h - 1), (box[2] + 3, box[1]),
             (box[0], box[3] + 3), (box[0] - w - 3, box[1])]
    for step in range(1, 14):
        dy = step * (h + 2)
        cands += [(box[2] + 3, box[1] - dy), (box[2] + 3, box[1] + dy),
                  (box[0] - w - 3, box[1] - dy), (box[0] - w - 3, box[1] + dy)]
    for x, y in cands:
        rect = (int(x), int(y), int(x) + w, int(y) + h)
        if not (panel[0] <= rect[0] and rect[2] <= panel[2]
                and panel[1] <= rect[1] and rect[3] <= panel[3]):
            continue
        if any(overlaps(rect, o) for o in occupied):
            continue
        occupied.append(rect)
        return rect
    # свободного места рядом не нашлось — ищем ближайшее по всей панели,
    # иначе подписи начинают затирать друг друга и кадр нечитаем
    best = None
    step = h + 2
    for y in range(panel[1], panel[3] - h, step):
        for x in range(panel[0], panel[2] - w, max(w // 2, 8)):
            rect = (x, y, x + w, y + h)
            if any(overlaps(rect, o) for o in occupied):
                continue
            d = (x - box[0]) ** 2 + (y - box[1]) ** 2
            if best is None or d < best[0]:
                best = (d, rect)
    rect = best[1] if best else (box[0], max(panel[1], box[1] - h - 1),
                                 box[0] + w, max(panel[1], box[1] - h - 1) + h)
    occupied.append(rect)
    return rect


def mark_defect(draw, poly, text, color, occupied, panel, offset):
    """Первый проход: рамка вокруг объекта.

    Рамки рисуются раньше подписей и целиком, иначе рамка следующего объекта
    ложится поверх уже нарисованной подписи предыдущего и обрезает текст.
    """
    ox, oy = offset
    x0, y0, x1, y1 = mark_box(poly, panel[2] - panel[0], panel[3] - panel[1])
    box = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)
    draw.rectangle(box, outline=color, width=2)
    occupied.append(box)
    return (box, text, color, panel)


def label_defect(draw, item, occupied, font):
    """Второй проход: подпись, поверх всех рамок."""
    box, text, color, panel = item
    rect = place_label(draw, text, box, occupied, panel, font)
    if not overlaps(rect, (box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6)):
        draw.line([((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                   ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)],
                  fill=color, width=1)
    draw.rectangle(rect, fill=color)
    draw.text((rect[0] + LABEL_PAD, rect[1] + LABEL_PAD), text,
              fill=(255, 255, 255), font=font)


def build_pair(base, frame, reference, pairs, extra, missing, low_iou, splits,
               font_small, font_bar):
    w, h = base.width, base.height
    canvas = Image.new("RGB", (w * 2 + GUTTER, TITLE_H + h + FOOT_H), BAR_BG)
    canvas.paste(base, (0, TITLE_H))
    canvas.paste(base, (w + GUTTER, TITLE_H))
    draw = ImageDraw.Draw(canvas)

    draw.text((6, 6), "Reference - COCO val2017", fill=BAR_FG, font=font_bar)
    draw.text((w + GUTTER + 6, 6), "My annotation", fill=BAR_FG, font=font_bar)

    left = (0, TITLE_H, w, TITLE_H + h)
    right = (w + GUTTER, TITLE_H, w * 2 + GUTTER, TITLE_H + h)

    for poly in reference.polys:
        draw_poly(draw, _shift(poly, 0, TITLE_H), REF_COLOR)
    for poly in frame.polys:
        draw_poly(draw, _shift(poly, w + GUTTER, TITLE_H), MINE_COLOR)

    occ_left, occ_right = [], []
    items_left, items_right = [], []
    for j in missing:
        items_left.append(mark_defect(draw, reference.polys[j],
                                      f"missed: {reference.polys[j].cls}",
                                      MISS_COLOR, occ_left, left, (0, TITLE_H)))
    for i in extra:
        items_right.append(mark_defect(draw, frame.polys[i],
                                       f"extra: {frame.polys[i].cls}", MISS_COLOR,
                                       occ_right, right, (w + GUTTER, TITLE_H)))
    for i, j, score in low_iou:
        items_left.append(mark_defect(draw, reference.polys[j], f"IoU {score:.2f}",
                                      WEAK_COLOR, occ_left, left, (0, TITLE_H)))
        items_right.append(mark_defect(draw, frame.polys[i], f"IoU {score:.2f}",
                                       WEAK_COLOR, occ_right, right,
                                       (w + GUTTER, TITLE_H)))
    for i, j, score in splits:
        r_cls, m_cls = reference.polys[j].cls, frame.polys[i].cls
        tag = (f"split: IoU {score:.2f}" if r_cls == m_cls
               else f"split: {r_cls}->{m_cls}, IoU {score:.2f}")
        items_left.append(mark_defect(draw, reference.polys[j], tag, SPLIT_COLOR,
                                      occ_left, left, (0, TITLE_H)))
        items_right.append(mark_defect(draw, frame.polys[i], tag, SPLIT_COLOR,
                                       occ_right, right, (w + GUTTER, TITLE_H)))
    for item in items_left + items_right:
        label_defect(draw, item, occ_left if item[3] is left else occ_right, font_small)
    return canvas, draw


def _shift(poly, dx, dy):
    """Копия объекта со сдвинутыми координатами — панели лежат на общем холсте."""
    class _P:
        pass
    p = _P()
    p.parts = [[c + (dx if k % 2 == 0 else dy) for k, c in enumerate(part)]
               for part in poly.parts]
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mine", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--low-iou", type=float, default=0.8,
                    help="сматченная пара ниже этого считается слабой")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--split-floor", type=float, default=0.1,
                    help="ниже этого IoU объекты не считаются одной разорванной парой")
    ap.add_argument("--keep-order", type=Path, default=None,
                    help="манифест прошлого прогона: сохранить прежние номера файлов")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="куда сложить JSON с составом дефектов для build_readme.py")
    args = ap.parse_args()

    keep = set(CLASSES)
    mine = {f.file_name: f for f in load_coco_polygons(args.mine, keep=keep)}
    ref = {f.file_name: f for f in load_coco_polygons(args.reference, keep=keep)}
    font_bar = load_font(14)

    rows = []
    for name, frame in mine.items():
        if name not in ref:
            continue
        w, h = frame.width, frame.height
        mm = [rasterize(x, w, h) for x in frame.polys]
        mr = [rasterize(x, w, h) for x in ref[name].polys]
        pairs, extra, missing = match_polys(frame.polys, ref[name].polys, mm, mr,
                                            args.iou_threshold)
        low = [(i, j, s) for i, j, s in pairs if s < args.low_iou]
        # Разорванная пара — один объект, а не пропуск плюс лишний. На картинке она
        # подписывается один раз своим цветом, иначе кадр говорит неправду.
        splits, extra, missing = split_pairs(frame.polys, ref[name].polys, mm, mr,
                                             extra, missing, args.split_floor)
        areas = {j: int(mr[j].sum()) for j in missing}
        rows.append((name, frame, ref[name], pairs, extra, missing, low, splits, areas))

    rows.sort(key=lambda r: -(len(r[4]) + len(r[5]) + len(r[6]) + len(r[7])))
    if args.keep_order and args.keep_order.exists():
        # Номера файлов заморожены по прежнему манифесту: так в git меняются только
        # те картинки, у которых поменялось содержимое, а не все 25 из-за сдвига.
        frozen = {e["frame"]: e["order"]
                  for e in json.loads(args.keep_order.read_text(encoding="utf-8"))}
        rows.sort(key=lambda r: frozen.get(r[0], 10_000))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for order, (name, frame, reference, pairs, extra, missing, low, splits,
                areas) in enumerate(rows, 1):
        src = args.images / name
        if not src.exists():
            print(f"пропущен {name}: нет файла в {args.images}")
            continue
        base = Image.open(src).convert("RGB")
        n_labels = max(len(missing), len(extra)) + len(low) + len(splits)
        font_small = load_font(10 if n_labels > 6 else 11)
        canvas, draw = build_pair(base, frame, reference, pairs, extra, missing,
                                  low, splits, font_small, font_bar)
        counts = f"matched {len(pairs)} · extra {len(extra)} · missed {len(missing)}"
        if low:
            counts += f" · weak {len(low)}"
        if splits:
            counts += f" · split {len(splits)}"
        if not extra and not missing and not low and not splits:
            counts += " · no defects"
        draw.text((6, canvas.height - FOOT_H + 6), f"{name} · {counts}",
                  fill=BAR_FG, font=font_bar)
        out_name = f"{order:02d}_{Path(name).stem}.jpg"
        canvas.save(args.out / out_name, quality=args.quality)
        manifest.append({
            "frame": name, "image": str(args.out / out_name), "order": order,
            "matched": len(pairs), "extra": len(extra), "missing": len(missing),
            "weak": len(low), "split": len(splits),
            "split_scores": [round(s, 2) for _, _, s in splits],
            "split_class_mismatch": sum(
                1 for i, j, _ in splits
                if frame.polys[i].cls != reference.polys[j].cls),
            "missing_classes": [reference.polys[j].cls for j in missing],
            "missing_areas": [areas[j] for j in missing],
            "extra_classes": [frame.polys[i].cls for i in extra],
            "weak_scores": [round(s, 2) for _, _, s in low],
        })

    if args.manifest:
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    clean = sum(1 for m in manifest
                if not (m["extra"] or m["missing"] or m["weak"] or m["split"]))
    print(f"{len(manifest)} pairs -> {args.out} (чистых {clean})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
