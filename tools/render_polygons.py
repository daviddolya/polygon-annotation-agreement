#!/usr/bin/env python3
"""Наложение своей полигональной разметки на эталонную (P4b, шаг 4).

Без картинок разбор расхождений не идёт: число «mask IoU 0.62» не говорит,
уехала граница по всему контуру или потерялась одна нога. Скрипт рисует
эталон одним цветом, свою разметку другим, и подписывает случай.

Режимы:
  --worst N   N кадров с худшим средним mask IoU — где разбирать в первую очередь
  --frames a.jpg b.jpg   конкретные кадры

Дополнительно к полному кадру сохраняется вырезка вокруг каждого спорного
объекта: контур в 30 px на кадре 640x480 глазом не разбирается.

    .venv/bin/python tools/render_polygons.py \
        --mine annotation/my_labels/instances_default.json \
        --reference ../detection-annotation-quality/data/coco/annotation/instances_val2017.json \
        --images data/subset/frames --out reports/review --worst 10
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from polygons import (CLASSES, load_coco_polygons, mask_iou,  # noqa: E402
                      match_polys, rasterize)

REF_COLOR = (31, 119, 180)
MINE_COLOR = (255, 127, 14)
BAR_H = 30
PAD = 40

FONT_CANDIDATES = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int = 15):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_poly(draw: ImageDraw.ImageDraw, poly, color, width=2):
    for part in poly.parts:
        points = [(part[i], part[i + 1]) for i in range(0, len(part) - 1, 2)]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=color, width=width)


def caption(image: Image.Image, text: str, font) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + BAR_H), (255, 255, 255))
    out.paste(image, (0, 0))
    ImageDraw.Draw(out).text((6, image.height + 6), text, fill=(0, 0, 0), font=font)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mine", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worst", type=int, default=10)
    p.add_argument("--frames", nargs="*", help="конкретные кадры вместо --worst")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--crop-below", type=float, default=0.8,
                   help="вырезку делать для пар с mask IoU ниже этого")
    args = p.parse_args()

    keep = set(CLASSES)
    mine = {f.file_name: f for f in load_coco_polygons(args.mine, keep=keep)}
    ref = {f.file_name: f for f in load_coco_polygons(args.reference, keep=keep)}
    font = load_font()

    scored = []
    for name, frame in mine.items():
        if name not in ref:
            continue
        w, h = frame.width, frame.height
        mm = [rasterize(x, w, h) for x in frame.polys]
        mr = [rasterize(x, w, h) for x in ref[name].polys]
        pairs, extra, missing = match_polys(frame.polys, ref[name].polys,
                                            mm, mr, args.iou_threshold)
        avg = sum(s for _, _, s in pairs) / len(pairs) if pairs else 0.0
        penalty = len(extra) + len(missing)
        scored.append((avg - 0.1 * penalty, name, pairs, extra, missing, mm, mr))

    scored.sort(key=lambda t: t[0])
    chosen = ([t for t in scored if t[1] in set(args.frames)] if args.frames
              else scored[:args.worst])
    if not chosen:
        raise SystemExit("нечего рисовать: кадры не найдены в своей разметке")

    args.out.mkdir(parents=True, exist_ok=True)
    made = 0
    for order, (_, name, pairs, extra, missing, mm, mr) in enumerate(chosen, 1):
        src = args.images / name
        if not src.exists():
            print(f"пропущен {name}: нет файла в {args.images}")
            continue
        base = Image.open(src).convert("RGB")
        frame, reference = mine[name], ref[name]

        full = base.copy()
        draw = ImageDraw.Draw(full)
        for poly in reference.polys:
            draw_poly(draw, poly, REF_COLOR)
        for poly in frame.polys:
            draw_poly(draw, poly, MINE_COLOR)
        text = (f"{name} | синий эталон, оранжевый мои | "
                f"пар {len(pairs)}, лишних {len(extra)}, пропущено {len(missing)}")
        caption(full, text, font).save(args.out / f"{order:02d}_{Path(name).stem}.jpg",
                                       quality=92)
        made += 1

        for i, j, score in pairs:
            if score >= args.crop_below:
                continue
            x, y, w, h = reference.polys[j].bbox()
            box = (max(0, int(x - PAD)), max(0, int(y - PAD)),
                   min(base.width, int(x + w + PAD)), min(base.height, int(y + h + PAD)))
            crop = full.crop(box)
            label = (f"{reference.polys[j].cls}: mask IoU {score:.2f}"
                     + ("" if frame.polys[i].cls == reference.polys[j].cls
                        else f" | у меня {frame.polys[i].cls}"))
            caption(crop, label, font).save(
                args.out / f"{order:02d}_{Path(name).stem}_obj{j}.jpg", quality=92)
            made += 1

    print(f"кадров разобрано {len(chosen)}, картинок {made} -> {args.out}")
    print("синий — эталон, оранжевый — своя разметка")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
