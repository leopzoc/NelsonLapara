"""
GPIO Button Controller — 3 physical buttons for mode selection.

Uses RPi.GPIO with interrupt-driven callbacks and software debounce.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

import config as cfg

log = logging.getLogger(__name__)


class Mode(Enum):
    THERAPEUTIC = auto()    # SER + hill-climbing intervention
    AVION = auto()          # Boeing 737 cabin lighting
    CIRCADIAN = auto()      # Daylight rhythm cycle


# Pin-to-mode mapping
_PIN_MODE_MAP = {
    cfg.BTN_THERAPEUTIC: Mode.THERAPEUTIC,
    cfg.BTN_AVION:       Mode.AVION,
    cfg.BTN_CIRCADIAN:   Mode.CIRCADIAN,
}


class ButtonController:
    """
    Interrupt-driven GPIO button handler.

    When a button is pressed, the `on_mode_change` callback is called
    with the new `Mode`.  Thread-safe; debounced.
    """

    def __init__(self, on_mode_change: Optional[Callable[[Mode], None]] = None):
        import RPi.GPIO as GPIO

        self._GPIO = GPIO
        self._callback = on_mode_change
        self._current_mode: Mode = Mode.THERAPEUTIC
        self._lock = threading.Lock()
        self._last_press_time: float = 0.0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in _PIN_MODE_MAP:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                pin,
                GPIO.FALLING,
                callback=self._on_button_press,
                bouncetime=cfg.BTN_DEBOUNCE_MS,
            )
            log.info("Button registered: GPIO %d → %s", pin, _PIN_MODE_MAP[pin].name)

        log.info("Button controller ready — default mode: %s", self._current_mode.name)

    @property
    def current_mode(self) -> Mode:
        with self._lock:
            return self._current_mode

    def _on_button_press(self, channel: int):
        now = time.monotonic()
        with self._lock:
            # Extra debounce guard
            if now - self._last_press_time < cfg.BTN_DEBOUNCE_MS / 1000.0:
                return
            self._last_press_time = now

            new_mode = _PIN_MODE_MAP.get(channel)
            if new_mode is None or new_mode == self._current_mode:
                return

            old_mode = self._current_mode
            self._current_mode = new_mode

        log.info("Mode switch: %s → %s (GPIO %d)", old_mode.name, new_mode.name, channel)

        if self._callback:
            # Fire callback outside the lock
            self._callback(new_mode)

    def close(self):
        self._GPIO.cleanup()
        log.info("GPIO cleaned up")
