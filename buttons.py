"""
GPIO Button Controller — single button cycles through modes.

Uses RPi.GPIO with interrupt-driven callback and software debounce.
A single press on the button cycles:  THERAPEUTIC → AVION → CIRCADIAN → ...
After each press the LED strip briefly flashes the mode's colour
(1 second) to give visual feedback, then the new mode starts.
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


# Ordered cycle list — the button rotates through these
_MODE_CYCLE = [Mode.THERAPEUTIC, Mode.AVION, Mode.CIRCADIAN]

# Feedback colour shown for 1 second when switching to each mode
MODE_FEEDBACK_COLORS = {
    Mode.THERAPEUTIC: cfg.MODE_COLOR_THERAPEUTIC,   # green
    Mode.AVION:       cfg.MODE_COLOR_AVION,         # violet
    Mode.CIRCADIAN:   cfg.MODE_COLOR_CIRCADIAN,     # orange
}

FEEDBACK_DURATION_SEC = 1.0


class ButtonController:
    """
    Interrupt-driven GPIO button handler (single button).

    When the button is pressed the mode advances to the next one in the
    cycle and the ``on_mode_change`` callback is called with the new
    `Mode`.  An optional ``led_strip`` reference is used to flash the
    mode colour for visual feedback.

    Thread-safe; debounced.
    """

    def __init__(
        self,
        on_mode_change: Optional[Callable[[Mode], None]] = None,
        led_strip=None,
    ):
        import RPi.GPIO as GPIO

        self._GPIO = GPIO
        self._callback = on_mode_change
        self._led = led_strip
        self._current_mode: Mode = Mode.THERAPEUTIC
        self._lock = threading.Lock()
        self._last_press_time: float = 0.0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Only one button — uses the THERAPEUTIC pin (GPIO 17)
        pin = cfg.BTN_MODE_CYCLE
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(
            pin,
            GPIO.FALLING,
            callback=self._on_button_press,
            bouncetime=cfg.BTN_DEBOUNCE_MS,
        )
        log.info(
            "Single-button controller ready on GPIO %d — default mode: %s",
            pin, self._current_mode.name,
        )

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

            # Advance to the next mode in the cycle
            idx = _MODE_CYCLE.index(self._current_mode)
            new_mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
            old_mode = self._current_mode
            self._current_mode = new_mode

        log.info(
            "Mode switch: %s → %s (GPIO %d)",
            old_mode.name, new_mode.name, channel,
        )

        # Visual feedback: flash the mode colour for 1 second
        self._flash_feedback(new_mode)

        if self._callback:
            # Fire callback outside the lock
            self._callback(new_mode)

    def _flash_feedback(self, mode: Mode):
        """Flash the LED strip with the mode's colour for 1 second."""
        if self._led is None:
            return
        color = MODE_FEEDBACK_COLORS.get(mode)
        if color:
            log.info("LED feedback: %s → %s", mode.name, color)
            self._led.set_color(color)
            time.sleep(FEEDBACK_DURATION_SEC)

    def close(self):
        self._GPIO.cleanup()
        log.info("GPIO cleaned up")
