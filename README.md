# Pechenka — YOLOv8 автокликер по экрану

Проект для обучения и применения YOLOv8-модели, которая в реальном времени
распознаёт элементы игры на экране (скриншоты через [`mss`](https://pypi.org/project/mss/))
и автоматически кликает по нужным объектам через [`pyautogui`](https://pypi.org/project/PyAutoGUI/).

## Классы модели

| id | имя      | описание          |
|----|----------|-------------------|
| 0  | `game`   | игровое поле      |
| 1  | `uprade` | кнопка улучшения  |
| 2  | `crest`  | герб / элемент UI |

## Структура

```
app.py             основной автокликер (crest/uprade -> best.pt, game -> bestV1.pt)
auto_clicker.py    упрощённый автокликер (game -> uprade)
debug_clicker.py   визуальная отладка: рисует найденные боксы на экране
infer.py           инференс модели на изображениях/папке
screner.py         сбор скриншотов с экрана
augment.py         аугментация датасета (albumentations)
json_to_yolo.py    конвертация разметки JSON -> формат YOLO
split_dataset.py   разделение датасета на train/val
train.py           обучение YOLOv8
ptToOnxx.py        экспорт .pt -> .onnx
data.yaml          описание датасета для обучения
config.yaml        конфиг для ONNX-рантайма
```

## Установка

```bash
python -m venv yolo_env
# Windows
yolo_env\Scripts\activate
pip install -r requirements.txt
```

## Обучение

Разметка должна лежать в `itog/images` и `itog/labels` (см. [`data.yaml`](data.yaml:1)).

```bash
python train.py
```

## Запуск автокликера

```bash
python app.py
```

Настройки (порог уверенности, монитор, задержки, размер входа) задаются
константами в начале [`app.py`](app.py:14).

Для безопасной проверки без кликов:

```bash
set PECHENKA_DRY_RUN=1 && python app.py
```

## Модели

Веса (`*.pt`, `*.onnx`), датасеты (`itog/`, `labels/`, `screenshots/`),
результаты обучения (`runs/`) и виртуальные окружения в репозиторий
**не включены** — они исключены через [`.gitignore`](.gitignore:1).
Положите `best.pt` / `bestV1.pt` (или `.onnx`) в корень проекта перед запуском.

## Примечание

`pyautogui.FAILSAFE = True`: аварийная остановка — резко увести курсор
в левый верхний угол экрана.
