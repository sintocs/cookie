# input_watch.py — глобальные горячие клавиши без сторонних библиотек.
#
# На Windows используется GetAsyncKeyState из user32: состояние клавиши
# читается в любой момент, независимо от того, какое окно в фокусе
# (а фокус, как правило, у игры, куда кликает автокликер).
#
# Использование:
#     watcher = KeyToggle()          # пробел по умолчанию
#     ...
#     if watcher.pressed():          # True ровно один раз на нажатие
#         ...                        # (edge-triggered, без автоповтора)
from __future__ import annotations

import ctypes
import sys

VK_SPACE = 0x20


class KeyToggle:
    """Отслеживает нажатие клавиши «по фронту» (нажали -> True, удержание -> False)."""

    def __init__(self, vk: int = VK_SPACE):
        self.vk = vk
        self._was_down = False
        self._get_async = None
        if sys.platform == "win32":
            try:
                self._get_async = ctypes.windll.user32.GetAsyncKeyState
                self._get_async.argtypes = [ctypes.c_int]
                self._get_async.restype = ctypes.c_short
            except Exception:  # pragma: no cover - экзотическое окружение
                self._get_async = None
        else:
            print("⚠️  Глобальная горячая клавиша поддерживается только на Windows.")

    @property
    def available(self) -> bool:
        return self._get_async is not None

    def pressed(self) -> bool:
        """True, если клавишу нажали с момента прошлого вызова."""
        if self._get_async is None:
            return False
        down = bool(self._get_async(self.vk) & 0x8000)
        pressed = down and not self._was_down
        self._was_down = down
        return pressed
