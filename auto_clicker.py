# auto_clicker.py — YOLO автоклик: game → uprade, сверху вниз (ФОН)
from ultralytics import YOLO
import numpy as np
import mss
import pyautogui
import time
import threading

from input_watch import KeyToggle

# === НАСТРОЙКИ ===
MODEL_PATH = "best.pt"
CONF_THRESH = 0.25          # порог для печеньки/улучшений
MONITOR = 1                 # номер монитора (проверьте в debug_clicker.py)
CLICK_COOLDOWN = 0.15       # мин. интервал между кликами (сек)
# ПРИОРИТЕТ: сначала game (печенька), потом uprade (улучшения)
PRIORITY_CLASSES = ["game", "uprade"]
# ==================

PAUSE_KEY = KeyToggle()     # пробел = пауза/продолжить

pyautogui.FAILSAFE = True   # левый верхний угол = аварийный стоп
pyautogui.PAUSE = 0.0005

last_click_time = 0
click_lock = threading.Lock()

def click_at(x, y):
    global last_click_time
    with click_lock:
        now = time.time()
        if now - last_click_time >= CLICK_COOLDOWN:
            pyautogui.click(x, y)
            last_click_time = now
            return True
    return False

def main():
    # 1. Модель
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Модель: {MODEL_PATH}")
        print(f"📋 Классы модели: {model.names}")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # 2. Экран
    sct = mss.MSS()
    monitors = sct.monitors
    if MONITOR >= len(monitors):
        print(f"❌ Монитор {MONITOR} не найден")
        return
    mon = monitors[MONITOR]
    
    print(f"🖥️ Монитор {MONITOR}: offset=({mon['left']},{mon['top']}) {mon['width']}x{mon['height']}")
    print(f"🎯 ПРИОРИТЕТ: {' → '.join(PRIORITY_CLASSES)} (сверху вниз)")
    print(f"   conf>={CONF_THRESH}, cooldown={CLICK_COOLDOWN}s")
    print("   ПРОБЕЛ = пауза/продолжить | Ctrl+C = стоп | Левый верхний угол = аварийный стоп")

    paused = False
    try:
        while True:
            # Пробел: пауза/продолжить (работает независимо от фокуса окна)
            if PAUSE_KEY.pressed():
                paused = not paused
                print("\n⏸️  ПАУЗА (пробел — продолжить)" if paused else "▶️  ПРОДОЛЖАЮ")
            if paused:
                time.sleep(0.02)
                continue

            # Скриншот
            img = np.array(sct.grab(mon))
            frame = img[:, :, :3]
            
            # Детекция
            results = model(frame, conf=CONF_THRESH, verbose=False)
            boxes = results[0].boxes
            
            if len(boxes) == 0:
                time.sleep(0.005)
                continue
            
            # Собираем клики: (priority, y_screen, x_screen, class_name, conf)
            clicks = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names.get(cls_id, f"cls_{cls_id}")
                
                if class_name not in PRIORITY_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_frame = (x1 + x2) // 2
                cy_frame = (y1 + y2) // 2
                cx_screen = mon['left'] + cx_frame
                cy_screen = mon['top'] + cy_frame
                
                # Приоритет: game=0, uprade=1
                priority = PRIORITY_CLASSES.index(class_name)
                clicks.append((priority, cy_screen, cx_screen, class_name, conf))
            
            if not clicks:
                time.sleep(0.005)
                continue
            
            # Сортируем: сначала по приоритету (game → uprade), потом по Y (сверху вниз)
            clicks.sort(key=lambda c: (c[0], c[1]))
            
            # Кликаем по очереди
            for _, y_scr, cx, class_name, conf in clicks:
                if PAUSE_KEY.pressed():
                    paused = not paused
                    print("\n⏸️  ПАУЗА (пробел — продолжить)" if paused else "▶️  ПРОДОЛЖАЮ")
                if paused:
                    break

                if click_at(cx, y_scr):
                    print(f"  🖱️ [{time.strftime('%H:%M:%S')}] {class_name} ({conf:.2f}) @ ({cx}, {y_scr})")
            
            time.sleep(0.005)
    
    except KeyboardInterrupt:
        print("\n🛑 Остановлено")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()