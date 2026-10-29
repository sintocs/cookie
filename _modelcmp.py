# _modelcmp.py — сравнение моделей на одном наборе скриншотов + статистика датасета.
# Запуск: python _modelcmp.py
import glob
import collections
import os
from ultralytics import YOLO

WEIGHTS = [
    "best.pt",
    "bestV1.pt",
    "runs/detect/runs/train/pechenka/weights/best.pt",
    "runs/detect/runs/train/pechenka-2/weights/best.pt",
    "runs/detect/runs/train/pechenka-3/weights/best.pt",
    "runs/detect/runs/train/pechenka-4/weights/best.pt",
]

IMAGES = sorted(glob.glob("screenshots/*.png"))
CONF = 0.25


def infer_stats(model):
    cnt = collections.Counter()
    confs = collections.defaultdict(list)
    for f in IMAGES:
        r = model(f, conf=CONF, verbose=False, imgsz=640)
        for b in r[0].boxes:
            name = model.names[int(b.cls[0])]
            cnt[name] += 1
            confs[name].append(float(b.conf[0]))
    return cnt, confs


def dataset_stats():
    print("\n=== ДАТАСЕТ (itog/labels) ===")
    for split in ("train", "val"):
        files = glob.glob(f"itog/labels/{split}/*.txt")
        per_class = collections.Counter()
        empty = 0
        for p in files:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for l in lines:
                per_class[int(l.split()[0])] += 1
        print(f"{split}: файлов={len(files)}, пустых={empty}, объекты по классам={dict(sorted(per_class.items()))}")
    # класс из data.yaml / train.py
    for y in ("data.yaml", "itog/data.yaml", "dataset.yaml"):
        if os.path.exists(y):
            print(f"\n--- {y} ---")
            print(open(y, encoding="utf-8", errors="ignore").read())


def main():
    print(f"Изображений для теста: {len(IMAGES)}")
    for w in WEIGHTS:
        if not os.path.exists(w):
            print(f"\n--- {w} --- НЕТ ФАЙЛА")
            continue
        print(f"\n--- {w} ---")
        try:
            m = YOLO(w)
            print("names:", m.names)
            cnt, confs = infer_stats(m)
            print("детекций:", dict(cnt) or "нет")
            for k, v in confs.items():
                print(f"   {k}: max={max(v):.2f} avg={sum(v)/len(v):.2f} n={len(v)}")
        except Exception as e:
            print("ОШИБКА:", e)
    dataset_stats()


if __name__ == "__main__":
    main()
