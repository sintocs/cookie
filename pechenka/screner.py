import os
import sys
import shutil
import random
import argparse
from pathlib import Path

# Расширения, которые считаем фотографиями
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.heic', '.heif', '.tiff', '.tif',
    '.bmp', '.gif', '.webp', '.raw', '.cr2', '.nef', '.arw', '.dng'
}


def split_dataset(source_dir: str, dest_dir: str, train_ratio: float, mode: str):
    """
    Разделяет фотографии на train и val в заданном соотношении.
    
    Args:
        source_dir: Путь к папке с исходными фото
        dest_dir: Путь к папке itog (где будут train и val)
        train_ratio: Доля фото для train (например, 0.8 = 80%)
        mode: 'copy' или 'move'
    """
    if not os.path.isdir(source_dir):
        print(f"❌ Исходная папка не найдена: {source_dir}")
        sys.exit(1)

    # Собираем все фото
    files = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    total = len(files)
    if total == 0:
        print("📭 В папке нет подходящих фотографий.")
        return

    # Перемешиваем случайным образом
    random.shuffle(files)

    # Разделяем на train и val
    train_count = int(total * train_ratio)
    train_files = files[:train_count]
    val_files = files[train_count:]

    print(f"🔍 Найдено фото: {total}")
    print(f"📊 Разделение: train={len(train_files)} ({train_ratio*100:.0f}%), val={len(val_files)} ({(1-train_ratio)*100:.0f}%)")
    print(f"📂 Режим: {'копирование' if mode == 'copy' else 'перемещение'}\n")

    # Создаём папки
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Функция для копирования/перемещения
    def process_file(filename, target_dir):
        src_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # Если файл с таким именем уже есть, добавляем _1, _2...
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
            counter += 1

        if mode == "copy":
            shutil.copy2(src_path, target_path)
        else:
            shutil.move(src_path, target_path)

    # Обрабатываем train
    print(f"📁 Копируем в train...")
    for i, filename in enumerate(train_files, 1):
        process_file(filename, train_dir)
        if i % 50 == 0 or i == len(train_files):
            print(f"   Train: {i}/{len(train_files)}")

    # Обрабатываем val
    print(f"📁 Копируем в val...")
    for i, filename in enumerate(val_files, 1):
        process_file(filename, val_dir)
        if i % 50 == 0 or i == len(val_files):
            print(f"   Val: {i}/{len(val_files)}")

    print(f"\n✅ Готово!")
    print(f"   📂 train: {len(train_files)} фото")
    print(f"   📂 val: {len(val_files)} фото")


def main():
    parser = argparse.ArgumentParser(
        description="Разделение фотографий на train и val для YOLO."
    )
    parser.add_argument("source", help="Путь к папке с исходными фото")
    parser.add_argument("dest", help="Путь к папке itog (где будут train и val)")
    parser.add_argument(
        "--ratio", type=float, default=0.8,
        help="Доля фото для train (по умолчанию 0.8 = 80%%)"
    )
    parser.add_argument(
        "--mode", choices=["copy", "move"], default="copy",
        help="copy — копировать (по умолчанию), move — перемещать"
    )
    args = parser.parse_args()

    if not 0 < args.ratio < 1:
        print("❌ Ratio должен быть между 0 и 1")
        sys.exit(1)

    split_dataset(args.source, args.dest, args.ratio, args.mode)


if __name__ == "__main__":
    main()