# _diag.py — БЕЗ кликов: что реально видит модель и какие центры она даёт.
# Запуск:  python _diag.py
# Выход:   q в окне предпросмотра
from ultralytics import YOLO
import numpy as np
import mss
import cv2
import time

MODEL_PATH = "best.pt"      # должно быть: 0=game, 1=uprade, 2=crest
CONF_THRESH = 0.25
MONITOR = 1

COLORS = {
    "game": (0, 255, 0),
    "uprade": (0, 165, 255),
    "upgrade": (0, 165, 255),
    "crest": (255, 0, 255),
}


def main():
    model = YOLO(MODEL_PATH)
    print(f"✅ Модель: {MODEL_PATH}")
    print(f"📋 Классы модели: {model.names}")
    names = [str(n).lower() for n in model.names.values()]
    for required in ("game", "crest"):
        print(f"   {'✅' if any(n.startswith(required) for n in names) else '❌'} класс '{required}'")

    with mss.mss() as sct:
        if MONITOR >= len(sct.monitors):
            print(f"❌ Монитор {MONITOR} недоступен, есть 1..{len(sct.monitors)-1}")
            return
        mon = sct.monitors[MONITOR]
        print(f"🖥️ Монитор {MONITOR}: ({mon['left']},{mon['top']}) {mon['width']}x{mon['height']}")

        frame_count = 0
        t0 = time.time()
        while True:
            frame = np.array(sct.grab(mon))[:, :, :3]
            results = model(frame, conf=CONF_THRESH, verbose=False)
            boxes = results[0].boxes
            frame_count += 1

            view = frame.copy()
            dets = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, f"cls_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = mon['left'] + (x1 + x2) // 2, mon['top'] + (y1 + y2) // 2
                dets.append((name, conf, cx, cy))
                color = COLORS.get(str(name).lower(), (200, 200, 200))
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                cv2.circle(view, ((x1 + x2) // 2, (y1 + y2) // 2), 4, (0, 0, 255), -1)
                cv2.putText(view, f"{name} {conf:.2f} ({cx},{cy})", (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            fps = frame_count / max(1e-6, time.time() - t0)
            cv2.putText(view, f"FPS {fps:.1f} | dets {len(dets)} | q=exit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if dets:
                for name, conf, cx, cy in dets:
                    print(f"  🎯 {name} {conf:.2f} -> screen({cx},{cy})")

            cv2.imshow("DIAG (no clicks) q=exit", view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("🛑 Диагностика завершена")


if __name__ == "__main__":
    main()
