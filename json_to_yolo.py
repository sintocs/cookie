# json_to_yolo.py — конвертер разметки labelme (JSON) в YOLO (.txt)
#
# Зачем: в JSON-аннотациях класс "game" ЕСТЬ, а в labels/*.txt его НЕТ
# (потерялся при прошлой конвертации). Из-за этого best.pt не умеет game.
#
# Запуск:
#   python json_to_yolo.py                # конвертировать screenshots/*.json -> labels/
#   python json_to_yolo.py --report       # только показать, что есть в JSON
#
# Пишет <stem>.txt рядом: labels/<stem>.txt
from pathlib import Path
import argparse
import json
import collections

SRC_JSON = Path("screenshots")
DST_LABELS = Path("labels")

# Имена классов -> индекс (индексы как в data.yaml)
CLASS_MAP = {
    "game": 0,
    "uprade": 1,
    "upgrade": 1,
    "crest": 2,
}


def shapes_to_lines(data):
    """JSON labelme -> список строк YOLO + счётчик неизвестных меток."""
    w = data.get("imageWidth") or 0
    h = data.get("imageHeight") or 0
    if not w or not h:
        raise ValueError("нет imageWidth/imageHeight")

    lines = []
    unknown = collections.Counter()
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        cls = CLASS_MAP.get(label)
        if cls is None:
            unknown[label] += 1
            continue
        pts = shape.get("points") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        # отсечём мусор
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="только отчёт, без записи")
    ap.add_argument("--overwrite", action="store_true", help="перезаписывать существующие .txt")
    args = ap.parse_args()

    files = sorted(SRC_JSON.glob("*.json"))
    if not files:
        print(f"❌ Не найдено JSON в {SRC_JSON}/")
        return

    DST_LABELS.mkdir(exist_ok=True)
    total = collections.Counter()
    written = 0
    skipped = 0
    unknown_all = collections.Counter()

    for jp in files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            lines, unknown = shapes_to_lines(data)
        except Exception as e:
            print(f"⚠️ {jp.name}: {e}")
            continue

        unknown_all.update(unknown)
        for line in lines:
            total[int(line.split()[0])] += 1

        stem = jp.stem  # имя png без расширения
        out = DST_LABELS / f"{stem}.txt"
        print(f"{jp.name}: {len(lines)} объект(ов) -> {out.name}"
              + (" (уже есть)" if out.exists() else ""))
        if args.report:
            continue
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    print("\n=== ИТОГО ===")
    print("Объекты по классам:", dict(sorted(total.items())))
    if unknown_all:
        print("Неизвестные метки (пропущены):", dict(unknown_all))
    if not args.report:
        print(f"Записано файлов: {written}, пропущено (уже были): {skipped}")
        print("Теперь запустите split_dataset.py и train.py, чтобы вернуть game в best.pt.")


if __name__ == "__main__":
    main()
