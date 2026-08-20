#!/usr/bin/env python3
"""Сборка README из метрик и парных картинок (P4b, шаг 5).

README на 25 покадровых разделов руками не поддерживается: числа в заголовках
обязаны сходиться с reports/polygon_metrics.json, а после переразметки —
пересобираться. Скрипт собирает файл заново из JSON, но комментарии владельца
не трогает: текст между маркерами <!-- note:кадр --> и <!-- /note --> читается
из существующего README и переносится в новый.

    .venv/bin/python tools/build_readme.py
"""

import argparse
import json
import re
from pathlib import Path

PLACEHOLDER = "> **Почему размечено так:** _заполнить_"
NOTE_RE = re.compile(r"<!-- note:(?P<frame>[^\s>]+) -->\n(?P<body>.*?)\n<!-- /note -->",
                     re.DOTALL)


def existing_notes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {m.group("frame"): m.group("body").strip()
            for m in NOTE_RE.finditer(path.read_text(encoding="utf-8"))}


def facts(entry: dict) -> str:
    """Строка фактов под заголовком кадра — генерируется, правке не подлежит."""
    bits = [f"сопоставлено {entry['matched']}"]
    if entry.get("split"):
        scores = entry["split_scores"]
        tail = (f", из них {entry['split_class_mismatch']} с другим классом"
                if entry.get("split_class_mismatch") else "")
        bits.append(f"разорванных пар {entry['split']} "
                    f"(IoU {min(scores):.2f}–{max(scores):.2f}{tail})")
    if entry["missing"]:
        classes = ", ".join(sorted(set(entry["missing_classes"])))
        areas = entry["missing_areas"]
        bits.append(f"пропущено {entry['missing']} ({classes}, "
                    f"{min(areas)}–{max(areas)} px²)")
    if entry["extra"]:
        bits.append(f"лишних {entry['extra']} ({', '.join(sorted(set(entry['extra_classes'])))})")
    if entry["weak"]:
        scores = entry["weak_scores"]
        bits.append(f"слабых пар {entry['weak']} (IoU {min(scores):.2f}–{max(scores):.2f})")
    if not (entry["missing"] or entry["extra"] or entry["weak"] or entry.get("split")):
        bits.append("расхождений нет")
    return " · ".join(bits)


def frame_section(entry: dict, notes: dict[str, str]) -> str:
    name = entry["frame"]
    body = notes.get(name, PLACEHOLDER)
    return (f"### {name}\n\n{facts(entry)}\n\n"
            f"![{name}]({entry['image']})\n\n"
            f"<!-- note:{name} -->\n{body}\n<!-- /note -->\n")


def head(m: dict) -> str:
    pc = m["per_class"]
    rows = "\n".join(
        f"| `{c}` | {d.get('matched', 0)} | {d.get('missing', 0)} | {d.get('extra', 0)} | "
        f"{m['per_class_mask_iou'].get(c, 0):.3f} | {m['per_class_boundary_iou'].get(c, 0):.3f} |"
        for c, d in pc.items())
    return f"""# polygon-annotation-agreement

Согласованность полигональной разметки: {m['frames']} кадров COCO val2017 размечены
вручную вслепую от эталона, свои контуры сравниваются с эталонными тремя метриками.
Этап A2 портфолио по контролю качества разметки.

Разметка боксами тех же кадров сделана раньше и живёт отдельно —
[detection-annotation-quality](https://github.com/daviddolya/detection-annotation-quality).
Это единственное, что даёт сравнить полигон с боксом на одних объектах.

## Результат

| | |
|---|---|
| кадров | {m['frames']} |
| полигонов своих / эталонных | {m['polygons_mine']} / {m['polygons_reference']} |
| сопоставлено | {m['matched']} |
| пропущено / лишних | {m['missing_annotation']} / {m['extra_annotation']} |
| **mask IoU** | **{m['mean_mask_iou']:.3f}** |
| Dice | {m['mean_dice']:.3f} |
| **Boundary IoU** | **{m['mean_boundary_iou']:.3f}** |
| Cohen's kappa по классам | {m['cohens_kappa']:.3f} |

Порог сопоставления — mask IoU {m['iou_threshold']}, сопоставление жадное и **без учёта
класса**: требовать совпадения метки нельзя, иначе ошибка класса сразу распадается
на пропуск и лишний объект, и согласие по классам считать будет не на чем.
Ширина полосы для Boundary IoU — {m['boundary_ratio']:.0%} диагонали кадра
([arXiv:2103.16562](https://arxiv.org/abs/2103.16562)).

**Разложение пропусков и лишних.** Из {m['missing_annotation']} пропущенных
и {m['extra_annotation']} лишних объектов {m['split_pairs']} случаев — это
**один объект, посчитанный дважды**: он размечен обеими сторонами, но контуры
разошлись сильнее порога, пара не сматчилась и распалась на пропуск плюс лишнее.
Средний mask IoU в таких парах {m['split_pairs_mean_iou']:.2f}, крайний случай — 0.48
при пороге 0.50. За их вычетом остаётся **пропущено {m['missing_after_split']},
лишних {m['extra_after_split']}**.

Порог при этом остаётся 0.5 и метрика выше не пересчитывается: подбирать порог под
результат нельзя, его называют. Разорванной парой считается перекрытие выше
mask IoU {m['split_floor']} — ниже начинаются случайные пересечения мелкого объекта
с чужим крупным контуром.

`kappa {m['cohens_kappa']:.3f}` — число вырожденное, а не достижение: среди
сопоставленных пар ни одной ошибки класса, считать согласие не на чем. При этом
ошибка класса в партии есть: {m['split_class_mismatch']} — `bicycle` в эталоне против
`motorcycle` у меня, mask IoU 0.37. В kappa она не попала именно потому, что пара
разорвана порогом, и это ограничение метрики, а не отсутствие ошибки.

| класс | сопоставлено | пропущено | лишних | mask IoU | Boundary IoU |
|---|---|---|---|---|---|
{rows}

Разрыв между mask IoU и Boundary IoU — {m['mean_mask_iou'] - m['mean_boundary_iou']:.2f}:
площадь совпадает заметно лучше границы. Так и должно быть, mask IoU почти не реагирует
на ошибку контура у крупных объектов.

## Как читать картинки

Каждый кадр показан дважды: слева эталон COCO, справа своя разметка.

| цвет | что значит |
|---|---|
| синий контур | эталон COCO |
| оранжевый контур | своя разметка |
| красная рамка, `missed` / `extra` | объект есть только у одной стороны |
| янтарная рамка, `IoU 0.NN` | пара нашлась, но контур разошёлся (mask IoU ниже 0.8) |
| фиолетовая рамка, `split` | один и тот же объект, но контуры разошлись сильнее порога 0.5 — пара разорвана, объект посчитан дважды |

Подписи на картинках английские: репозиторий публичный.

## Разбор кадров

Порядок — по убыванию числа расхождений. Комментарий под каждым кадром объясняет,
почему разметка сделана именно так; расхождение с эталоном не считается ошибкой
по умолчанию — на COCO пропуски мелких и перекрытых объектов регулярны.
"""


def tail(report: Path) -> str:
    # ссылка на отчёт ставится, только когда файл есть: битая ссылка
    # в публичном README хуже отсутствующей
    ref = (f"\n\nПодробный разбор — [`{report}`]({report})." if report.exists() else "")
    return """## Правила, которые из этого выросли

Разбор кадров дал два правила, дописанных в
[`annotation/GUIDELINES.md`](annotation/GUIDELINES.md), раздел 2:

- **2.1** мелкие объекты заднего плана размечаются, кадр проходится дважды — чинится
  переразметкой, а не смягчением порога;
- **2.2** детализация контура задаётся инструкцией, а не стремлением к точности —
  чинится инструкцией, сданная работа задним числом не бракуется.

Каждое правило записано с кадром-примером, числом и датой. История правил — такой же
артефакт, как метрики: по ней видно, откуда взялось решение.

## Полигон против бокса

Те же кадры размечены боксами в
[detection-annotation-quality](https://github.com/daviddolya/detection-annotation-quality)
(100 кадров, Cohen's kappa 0.914, средний IoU 0.867). На пересечении двух проектов
видно то, чего не видно ни в одном по отдельности: полигон занимает около 60% площади
своего бокса, то есть **40% боксовой разметки — фон**. По классам цена разная:
у `bus` 78% и 18 вершин, у `motorcycle` 53% и 37 вершин. Полигоны окупаются не везде,
и решение по классам принимается числом.

{report_link}

## Воспроизведение

Данные под git не хранятся, кроме отобранных 25 кадров. Полный набор COCO val2017 —
[изображения](http://images.cocodataset.org/zips/val2017.zip),
[аннотации](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

```bash
python3 -m venv .venv && .venv/bin/pip install -q pillow numpy
REF=../detection-annotation-quality/data/coco/annotation/instances_val2017.json

.venv/bin/python annotation/polygon_agreement.py \\
    --mine annotation/my_labels/instances_default.json \\
    --reference $REF --out reports/polygon_metrics.json

.venv/bin/python tools/render_pairs.py \\
    --mine annotation/my_labels/instances_default.json --reference $REF \\
    --images data/subset/frames --out reports/pairs \\
    --manifest reports/pairs_manifest.json

.venv/bin/python tools/build_readme.py
```

Последняя команда пересобирает этот файл. Комментарии под кадрами при пересборке
сохраняются — они лежат между маркерами `<!-- note:кадр -->` и `<!-- /note -->`.

## Состав

```
annotation/
├── GUIDELINES.md          инструкция: правила контура и решения по спорным случаям
├── my_labels/             своя разметка, экспорт CVAT в COCO 1.0
└── polygon_agreement.py   сопоставление объектов и метрики согласия
common/
├── polygons.py            растеризация, mask IoU, Dice, Boundary IoU
├── boxes.py, agreement.py перенесены из проекта по боксам
tools/
├── select_polygon_frames.py  отбор 25 кадров под полигоны
├── polygon_stats.py          полигон против бокса на эталоне
├── miss_breakdown.py         пропуск: не размечен или контур разошёлся
├── render_polygons.py        совмещённый кадр и вырезки для работы глазами
├── render_pairs.py           парные картинки для этого README
└── build_readme.py           сборка README
reports/
├── polygon_metrics.json   числа
├── pairs/                 парные картинки, 25 штук
└── review/                совмещённые кадры и вырезки
DEBT.md                    что написано ассистентом и что нужно уметь объяснить
```
""".replace("{report_link}", ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, default=Path("reports/polygon_metrics.json"))
    ap.add_argument("--manifest", type=Path, default=Path("reports/pairs_manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("README.md"))
    ap.add_argument("--report", type=Path, default=Path("reports/polygon_report.md"))
    args = ap.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    notes = existing_notes(args.out)

    defects = [e for e in manifest
               if e["missing"] or e["extra"] or e["weak"] or e.get("split")]
    clean = [e for e in manifest if e not in defects]

    parts = [head(metrics)]
    parts += [frame_section(e, notes) for e in defects]
    if clean:
        parts.append(f"### Без расхождений\n\nЕщё {len(clean)} кадров сошлись с эталоном "
                     f"полностью: ни пропусков, ни лишних, ни разорванных "
                     f"пар, все сопоставленные объекты выше mask IoU 0.8.\n")
        parts += [frame_section(e, notes) for e in clean]
    parts.append(tail(args.report))

    import re
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(parts))
    args.out.write_text(text, encoding="utf-8")
    kept = sum(1 for e in manifest
               if notes.get(e["frame"], PLACEHOLDER) != PLACEHOLDER)
    print(f"README собран: кадров {len(manifest)} "
          f"(с расхождениями {len(defects)}, чистых {len(clean)}), "
          f"комментариев владельца {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
