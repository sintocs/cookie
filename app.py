# app.py — YOLO-автокликер для печеньки (ДВУХМОДЕЛЬНЫЙ)
#
# Почему две модели:
#   best.pt    — 0=game, 1=uprade, 2=crest, НО в его датасете НЕТ ни одного
#                объекта класса "game" (разметка потерялась при JSON->YOLO).
#                Поэтому он физически не детектит печеньку.
#   bestV1.pt  — 0=game, 1=upgrade, 2=advertising. Печеньку видит отлично
#                (conf ~0.94), поэтому game берём из него.
#
# Итог: crest + uprade кликаем по best.pt, game — по bestV1.pt.
# Когда переобучите best.pt с классом game (см. json_to_yolo.py), можно
# оставить в MODELS только best.pt.
from ultralytics import YOLO
import numpy as np
import mss
import pyautogui
import time
import os

# === НАСТРОЙКИ ===
CONF_THRESH = 0.25          # порог уверенности
MONITOR = 1                 # номер монитора
MIN_CLICK_GAP = 0.03        # мин. пауза между ЛЮБЫМИ кликами (сек)
IMGSZ = 640                 # размер инференса

# Целевые классы: префикс имени -> (приоритет, кулдаун по классу в сек)
# Префикс — чтобы работали и "uprade", и "upgrade".
TARGETS = {
    "crest":  (0, 1.5),      # сначала редкие бонусы — чтобы не голодали
    "uprad":  (1, 0.8),      # upgrade / uprade
    "game":   (2, 0.01),     # печеньку спамим постоянно
}

# Какая модель за какие классы отвечает.
# Порядок важен только для логов.
MODELS = [
    # (путь, {префикс: (приоритет, кулдаун)})
    ("best.pt",   {"crest": (0, 1.5), "uprad": (1, 0.8)}),
    ("bestV1.pt", {"game":  (2, 0.01)}),
]

# Дедуп: два бокса одного класса ближе этого расстояния (px) — один клик
DEDUP_DIST = 30

# Размер ячейки (px) для кулдауна по конкретной точке экрана.
# Кулдаун привязан к месту, а не к классу целиком — иначе после клика
# по верхней кнопке upgrade скрипт «залипает» на ней и не идёт вниз.
POINT_CELL = 40

DEBUG_ALL_DETECTIONS = False   # печатать ВСЕ детекции, включая игнорируемые
DRY_RUN = os.environ.get("PECHENKA_DRY_RUN") == "1"   # без реальных кликов
# ==================

pyautogui.FAILSAFE = True   # левый верхний угол = аварийный стоп
pyautogui.PAUSE = 0.0005

last_click_any = 0.0
last_click_by_point = {}   # (класс, ячейка_x, ячейка_y) -> время последнего клика


def point_key(key, x, y):
    """Ключ точки клика для кулдауна: конкретное место на экране, не весь класс."""
    return (key, x // POINT_CELL, y // POINT_CELL)


def resolve_target(class_name, allowed):
    """Возвращает (приоритет, кулдаун, ключ) или None, если класс не нужен модели."""
    name = str(class_name).strip().lower()
    for prefix, (prio, cooldown) in allowed.items():
        if name.startswith(prefix):
            return prio, cooldown, prefix
    return None


def click_at(x, y):
    """Кликает, соблюдая глобальную минимальную паузу. True если кликнули."""
    global last_click_any
    now = time.time()
    if now - last_click_any < MIN_CLICK_GAP:
        return False
    if not DRY_RUN:
        try:
            pyautogui.click(x, y, duration=0)
        except pyautogui.FailSafeException:
            raise KeyboardInterrupt("FAILSAFE")
    last_click_any = time.time()
    return True


def load_model(path):
    if not os.path.exists(path):
        print(f"❌ Нет файла модели: {path}")
        return None
    try:
        m = YOLO(path)
        print(f"✅ Модель {path}: {m.names}")
        return m
    except Exception as e:
        print(f"❌ Ошибка загрузки {path}: {e}")
        return None


def main():
    # 1. Модели
    loaded = []
    for path, allowed in MODELS:
        m = load_model(path)
        if m is None:
            continue
        names = m.names
        names_list = list(names.values()) if isinstance(names, dict) else list(names)
        available = [str(n).strip().lower() for n in names_list]
        missing = [p for p in allowed if not any(a.startswith(p) for a in available)]
        if missing:
            print(f"⚠️  В {path} НЕТ классов {missing} — они не будут кликаться этой моделью.")
        loaded.append((m, allowed, path))

    if not loaded:
        print("❌ Ни одна модель не загружена — выход.")
        return

    covered = {p for _, allowed, _ in loaded for p in allowed}
    uncovered = [p for p in TARGETS if p not in covered]
    if uncovered:
        print(f"❗ Классы без модели: {uncovered} — кликаться не будут.")
    print(f"🎯 Покрытые цели: {sorted(covered)}")

    # 2. Экран
    sct = mss.mss()
    monitors = sct.monitors
    if MONITOR >= len(monitors) or MONITOR < 1:
        print(f"❌ Монитор {MONITOR} не найден. Доступно: {len(monitors) - 1}")
        for i, m in enumerate(monitors[1:], 1):
            print(f"     {i}: left={m['left']}, top={m['top']}, {m['width']}x{m['height']}")
        print("   Измените MONITOR в начале файла.")
        return
    mon = monitors[MONITOR]

    print(f"🖥️ Монитор {MONITOR}: offset=({mon['left']},{mon['top']}) {mon['width']}x{mon['height']}")
    print("🎯 ЦЕЛИ: " + " → ".join(f"{p}(cd={cd}s)" for p, (_, cd) in
                                  sorted(TARGETS.items(), key=lambda kv: kv[1][0])))
    print(f"   conf>={CONF_THRESH}, min_gap={MIN_CLICK_GAP}s"
          + (" | DRY-RUN (клики отключены)" if DRY_RUN else ""))
    print("   ВАЖНО: если игра запущена от администратора — этот скрипт тоже")
    print("          должен быть запущен от администратора, иначе клики игнорируются.")
    print("   Ctrl+C = стоп | Левый верхний угол = аварийный стоп")

    stats = {}
    try:
        while True:
            img = np.array(sct.grab(mon))
            frame = img[:, :, :3]

            now = time.time()
            pending = []     # (priority, y, x, class_name, conf, key)
            seen = []        # (key, cx, cy) — для дедупа между моделями

            for model, allowed, path in loaded:
                results = model(frame, conf=CONF_THRESH, verbose=False, imgsz=IMGSZ)
                boxes = results[0].boxes

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names.get(cls_id, f"cls_{cls_id}")

                    if DEBUG_ALL_DETECTIONS:
                        print(f"  🔎 [{path}] {class_name} ({conf:.2f})")

                    target = resolve_target(class_name, allowed)
                    if target is None:
                        continue
                    priority, cooldown, key = target

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = mon['left'] + (x1 + x2) // 2
                    cy = mon['top'] + (y1 + y2) // 2

                    # кулдаун по конкретной точке, а не по всему классу
                    pk = point_key(key, cx, cy)
                    if now - last_click_by_point.get(pk, 0.0) < cooldown:
                        continue

                    # дедуп: тот же класс, почти та же точка
                    dup = any(k == key and abs(cx - sx) <= DEDUP_DIST and abs(cy - sy) <= DEDUP_DIST
                              for k, sx, sy in seen)
                    if dup:
                        continue
                    seen.append((key, cx, cy))
                    pending.append((priority, cy, cx, class_name, conf, key, cooldown))

            if not pending:
                time.sleep(0.002)
                continue

            # Сначала бонусы (crest, uprade), затем печенька;
            # внутри класса — строго сверху вниз (по cy), затем слева направо (cx).
            pending.sort(key=lambda p: (p[0], p[1], p[2]))

            for priority, cy, cx, class_name, conf, key, cooldown in pending:
                pk = point_key(key, cx, cy)
                if time.time() - last_click_by_point.get(pk, 0.0) < cooldown:
                    continue
                if click_at(cx, cy):
                    last_click_by_point[pk] = time.time()
                    stats[key] = stats.get(key, 0) + 1
                    print(f"  🖱️ [{time.strftime('%H:%M:%S')}] {class_name} "
                          f"({conf:.2f}) @ ({cx}, {cy}) | всего: {stats[key]}")
                else:
                    # глобальный лимит MIN_CLICK_GAP не дал кликнуть —
                    # прерываем список, чтобы следующий проход снова начал сверху
                    break

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\n🛑 Остановлено")
        print(f"📊 Кликов по классам: {stats}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
