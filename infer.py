# infer.py — быстрый инференс YOLO (веб-камера / фото / видео)
from ultralytics import YOLO
import cv2

# === НАСТРОЙКИ ===
MODEL_PATH = "best.pt"          # модель в корне проекта
CONF_THRESH = 0.25              # порог уверенности
SOURCE = "webcam"               # "webcam" | "photo.jpg" | "video.mp4"
SAVE_RESULT = False             # True — сохранить результат
# ==================

CLASS_NAMES = ["game", "uprade", "crest"]

def main():
    # 1. Загружаем модель
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Модель загружена: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        print("   Убедитесь, что best.pt лежит рядом с infer.py")
        return

    # 2. Источник
    if SOURCE == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Камера не найдена")
            return
        print("🎥 Веб-камера запущена. Нажмите 'q' для выхода.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, conf=CONF_THRESH, verbose=False)
            annotated = results[0].plot()
            
            # FPS
            fps = 1 / (results[0].speed['inference'] / 1000)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif SOURCE.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        results = model(SOURCE, conf=CONF_THRESH, save=SAVE_RESULT)
        print(f"✅ Готово. Найдено объектов: {len(results[0].boxes)}")
        results[0].show()
    
    elif SOURCE.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        results = model(SOURCE, stream=True, conf=CONF_THRESH, save=SAVE_RESULT)
        for r in results:
            cv2.imshow("YOLO Video", r.plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    else:
        print(f"❌ Неизвестный источник: {SOURCE}")

if __name__ == "__main__":
    main()