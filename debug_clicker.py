# debug_clicker.py — ДИАГНОСТИКА: что видит модель и куда кликает
from ultralytics import YOLO
import numpy as np
import mss
import pyautogui
import cv2
import time

# === НАСТРОЙКИ ===
MODEL_PATH = "best.pt"
CONF_THRESH = 0.25          # НИЗКИЙ порог — чтобы увидеть ВСЕ детекции
MONITOR = 1
# ==================

CLASS_NAMES = ["game", "uprade", "crest"]  # ПРОВЕРЬТЕ ПОРЯДОК!
CLASS_COLORS = {
    "game": (0, 255, 0),
    "uprade": (0, 165, 255),
    "crest": (255, 0, 255)
}

def main():
    print("🔍 ДИАГНОСТИКА: запуск с окном для проверки детекций")
    print("   Нажмите 'q' в окне для выхода")
    print("   Наведите мышку на печеньку — посмотрите координаты в консоли")
    
    # Модель
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Модель: {MODEL_PATH}")
        print(f"📋 Классы модели: {model.names}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    # Экран
    sct = mss.MSS()
    monitors = sct.monitors
    if MONITOR >= len(monitors):
        print(f"❌ Монитор {MONITOR} не найден. Доступно: {len(monitors)-1}")
        for i, m in enumerate(monitors[1:], 1):
            print(f"     {i}: left={m['left']}, top={m['top']}, {m['width']}x{m['height']}")
        return
    mon = monitors[MONITOR]
    print(f"🖥️ Монитор {MONITOR}: left={mon['left']}, top={mon['top']}, {mon['width']}x{mon['height']}")

    # Текущая позиция мышки (для сверки)
    def print_mouse_pos():
        x, y = pyautogui.position()
        print(f"   🖱️ Мышь: ({x}, {y})")

    try:
        while True:
            # Скриншот
            img = np.array(sct.grab(mon))
            frame = img[:, :, :3]
            
            # Детекция
            results = model(frame, conf=CONF_THRESH, verbose=False)
            boxes = results[0].boxes
            
            debug_frame = frame.copy()
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Реальное имя класса из модели
                real_class_name = model.names.get(cls_id, f"cls_{cls_id}")
                display_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else real_class_name
                
                # Координаты
                cx_frame = (x1 + x2) // 2
                cy_frame = (y1 + y2) // 2
                cx_screen = mon['left'] + cx_frame
                cy_screen = mon['top'] + cy_frame
                
                color = CLASS_COLORS.get(display_name, (255, 255, 255))
                
                # Рисуем
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{real_class_name} ({cls_id}) {conf:.2f}"
                cv2.putText(debug_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(debug_frame, (cx_frame, cy_frame), 5, (0, 0, 255), -1)
                cv2.putText(debug_frame, f"SCR: ({cx_screen},{cy_screen})", (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Лог в консоль
                print(f"  🎯 {real_class_name} (id={cls_id}) conf={conf:.2f} box=({x1},{y1},{x2},{y2}) center_frame=({cx_frame},{cy_frame}) screen=({cx_screen},{cy_screen})")
            
            # Показ мышки
            mx, my = pyautogui.position()
            cv2.putText(debug_frame, f"MOUSE: ({mx},{my})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"MONITOR OFFSET: ({mon['left']},{mon['top']})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("DEBUG - что видит модель (q=выход)", debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Пробел = показать позицию мышки
                print_mouse_pos()
    
    finally:
        cv2.destroyAllWindows()
        print("🛑 Диагностика завершена")

if __name__ == "__main__":
    main()