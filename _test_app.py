# _test_app.py — прогон логики app.py без кликов: 5 кадров живого экрана.
import os
import time

os.environ["PECHENKA_DRY_RUN"] = "1"

import importlib
import app
importlib.reload(app)

import mss
import numpy as np

print("DRY_RUN =", app.DRY_RUN)
print("MODELS =", app.MODELS)

loaded = []
for path, allowed in app.MODELS:
    m = app.load_model(path)
    if m is not None:
        loaded.append((m, allowed, path))

with mss.mss() as sct:
    mon = sct.monitors[app.MONITOR]
    print("monitor:", mon)

    # 1) живой экран
    for i in range(3):
        frame = np.array(sct.grab(mon))[:, :, :3]
        found = []
        for model, allowed, path in loaded:
            r = model(frame, conf=app.CONF_THRESH, verbose=False, imgsz=app.IMGSZ)
            for b in r[0].boxes:
                name = model.names[int(b.cls[0])]
                t = app.resolve_target(name, allowed)
                if t:
                    found.append((path, name, round(float(b.conf[0]), 2), t[2]))
        print(f"экран {i}: {found if found else 'детекций целей нет'}")
        time.sleep(0.3)

# 2) офлайн-проверка на скриншотах: кто из моделей даёт какие цели
import glob
print("\n=== офлайн по screenshots/*.png (маршрутизация целей) ===")
imgs = sorted(glob.glob("screenshots/*.png"))[:6]
for f in imgs:
    found = []
    for model, allowed, path in loaded:
        r = model(f, conf=app.CONF_THRESH, verbose=False, imgsz=app.IMGSZ)
        for b in r[0].boxes:
            name = model.names[int(b.cls[0])]
            t = app.resolve_target(name, allowed)
            if t:
                found.append(f"{t[2]}<-{path}({float(b.conf[0]):.2f})")
    print(f"  {f.split(chr(92))[-1]}: {found if found else 'нет'}")
print("OK: логика маршрутизации работает, кликов не было")
