# train.py — Обучение YOLOv8 на вашем датасете
# Запуск: python train.py

from ultralytics import YOLO

if __name__ == "__main__":
    # 1. Загружаем предобученную модель (nano — быстро, small — точнее)
    model = YOLO("yolov8n.pt")  # или "yolov8s.pt" для лучшего качества

    # 2. Запускаем обучение
    results = model.train(
        data="data.yaml",        # конфиг датасета
        epochs=20,               # эпох (быстро для дообучения)
        imgsz=640,               # размер изображения
        batch=16,                # батч
        device="cpu",            # процессор
        project="runs/train",    # папка для сохранения
        name="pechenka",         # имя эксперимента
        save=True,               # сохранять чекпоинты
        val=True,                # валидация каждую эпоху
        plots=True,              # графики loss/mAP
        patience=10,             # early stopping
        verbose=True
    )

    # 3. Путь к лучшей модели
    best_pt = results.save_dir / "weights" / "best.pt"
    print(f"\n✅ Обучение завершено!")
    print(f"🏆 Лучшая модель: {best_pt}")
    print(f"📊 Результаты: {results.save_dir}")
    print(f"\n👉 Для инференса используйте: YOLO('{best_pt}')")