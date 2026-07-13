"""
GPIO Button Controller — multi-button system using lgpio directly.

Uses lgpio (native RPi 5 GPIO library) with edge-detection callbacks.
This avoids conflicts with adafruit-blinka which also uses lgpio
for the NeoPixel SPI driver.

Buttons:
  • GPIO 17 — Mode cycle: THERAPEUTIC → AVION → CIRCADIAN → AUTISM → ...
  • GPIO  5 — Lamp toggle: turn LED strip on/off
  • GPIO  6 — Play/Stop:  toggle audio playback
  • GPIO 13 — Volume Up:  increase audio volume
  • GPIO 19 — Volume Down: decrease audio volume

After each mode press the LED strip briefly flashes the mode's colour
(1 second) to give visual feedback, then the new mode starts.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

import lgpio

import config as cfg

log = logging.getLogger(__name__)


class Mode(Enum):
    THERAPEUTIC = auto()    # SER + hill-climbing intervention
    AVION = auto()          # Boeing 737 cabin lighting
    CIRCADIAN = auto()      # Daylight rhythm cycle
    AUTISM = auto()         # Neuro-relaxation sensory lighting


# Ordered cycle list — the button rotates through these
_MODE_CYCLE = [Mode.THERAPEUTIC, Mode.AVION, Mode.CIRCADIAN, Mode.AUTISM]

# Feedback colour shown for 1 second when switching to each mode
MODE_FEEDBACK_COLORS = {
    Mode.THERAPEUTIC: cfg.MODE_COLOR_THERAPEUTIC,   # green
    Mode.AVION:       cfg.MODE_COLOR_AVION,         # violet
    Mode.CIRCADIAN:   cfg.MODE_COLOR_CIRCADIAN,     # orange
    Mode.AUTISM:      cfg.MODE_COLOR_AUTISM,         # soft blue
}

# Map modes to their audio tracks (None = no audio for that mode)
MODE_AUDIO_TRACKS = {
    Mode.THERAPEUTIC: None,
    Mode.AVION:       cfg.AUDIO_TRACK_AVION,
    Mode.CIRCADIAN:   cfg.AUDIO_TRACK_CIRCADIAN,
    Mode.AUTISM:      cfg.AUDIO_TRACK_AUTISM,
}

FEEDBACK_DURATION_SEC = 1.0


class ButtonController:
    """
    GPIO button handler using lgpio directly (RPi 5 compatible).

    Uses lgpio.gpio_claim_alert + lgpio.callback for interrupt-driven
    edge detection.  Opens its own chip handle to avoid conflicts with
    adafruit-blinka's SPI GPIO usage.

    Buttons:
      • Mode cycle (GPIO 17): advances to the next mode.
      • Lamp toggle (GPIO 5): fires ``on_lamp_toggle`` callback.
      • Play/Stop (GPIO 6): fires ``on_play_stop`` callback.
      • Volume Up (GPIO 13): fires ``on_volume_up`` callback.
      • Volume Down (GPIO 19): fires ``on_volume_down`` callback.

    Thread-safe; debounced.
    """

    def __init__(
        self,
        on_mode_change: Optional[Callable[[Mode], None]] = None,
        on_lamp_toggle: Optional[Callable[[], None]] = None,
        on_play_stop: Optional[Callable[[], None]] = None,
        on_volume_up: Optional[Callable[[], None]] = None,
        on_volume_down: Optional[Callable[[], None]] = None,
        led_strip=None,
    ):
        self._cb_mode_change = on_mode_change
        self._cb_lamp_toggle = on_lamp_toggle
        self._cb_play_stop = on_play_stop
        self._cb_volume_up = on_volume_up
        self._cb_volume_down = on_volume_down
        self._led = led_strip
        self._current_mode: Mode = Mode.THERAPEUTIC
        self._lock = threading.Lock()
        self._last_press_times: dict[int, float] = {}
        self._callbacks: list = []  # lgpio callback objects

        # ── Open GPIO chip ─────────────────────────────────────────
        # RPi 5 uses gpiochip4 (RP1), RPi 4 uses gpiochip0
        self._chip_handle = None
        for chip_num in (4, 0):
            try:
                self._chip_handle = lgpio.gpiochip_open(chip_num)
                log.info("Opened /dev/gpiochip%d", chip_num)
                break
            except lgpio.error:
                continue

        if self._chip_handle is None:
            log.error("Could not open any GPIO chip! Buttons will not work.")
            return

        # ── Setup all buttons ──────────────────────────────────────
        buttons = {
            cfg.BTN_MODE_CYCLE: ("Mode Cycle", self._on_mode_press),
            cfg.BTN_LAMP_TOGGLE: ("Lamp Toggle", self._on_lamp_press),
            cfg.BTN_PLAY_STOP: ("Play/Stop", self._on_play_stop_press),
            cfg.BTN_VOL_UP: ("Volume Up", self._on_vol_up_press),
            cfg.BTN_VOL_DOWN: ("Volume Down", self._on_vol_down_press),
        }

        registered = 0
        for pin, (name, callback) in buttons.items():
            try:
                # Claim the pin for alert (edge detection) with pull-up
                lgpio.gpio_claim_alert(
                    self._chip_handle, pin,
                    lgpio.FALLING_EDGE,
                    lgpio.SET_PULL_UP,
                )
                # Register the callback
                cb = lgpio.callback(
                    self._chip_handle, pin,
                    lgpio.FALLING_EDGE,
                    callback,
                )
                self._callbacks.append(cb)
                registered += 1
                log.info("✓ Button '%s' registered on GPIO %d", name, pin)
            except lgpio.error as e:
                log.error(
                    "✗ Failed to setup '%s' on GPIO %d: %s — "
                    "pin may be busy or reserved. "
                    "Change the pin in config.py if needed.",
                    name, pin, e,
                )

        log.info(
            "ButtonController ready — %d/%d buttons active, default mode: %s",
            registered, len(buttons), self._current_mode.name,
        )

    @property
    def current_mode(self) -> Mode:
        with self._lock:
            return self._current_mode

    # ── Debounce helper ────────────────────────────────────────────

    def _debounced(self, gpio: int) -> bool:
        """Returns True if the press should be ignored (too recent)."""
        now = time.monotonic()
        with self._lock:
            last = self._last_press_times.get(gpio, 0.0)
            if now - last < cfg.BTN_DEBOUNCE_MS / 1000.0:
                return True
            self._last_press_times[gpio] = now
        return False

    # ── lgpio callback signature: (chip, gpio, level, timestamp) ──

    def _on_mode_press(self, chip: int, gpio: int, level: int, timestamp: int):
        if self._debounced(gpio):
            return

        with self._lock:
            idx = _MODE_CYCLE.index(self._current_mode)
            new_mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
            old_mode = self._current_mode
            self._current_mode = new_mode

        log.info(
            "Mode switch: %s → %s (GPIO %d)",
            old_mode.name, new_mode.name, gpio,
        )

        self._flash_feedback(new_mode)

        if self._cb_mode_change:
            self._cb_mode_change(new_mode)

    def _on_lamp_press(self, chip: int, gpio: int, level: int, timestamp: int):
        if self._debounced(gpio):
            return
        log.info("Lamp toggle pressed (GPIO %d)", gpio)
        if self._cb_lamp_toggle:
            self._cb_lamp_toggle()

    def _on_play_stop_press(self, chip: int, gpio: int, level: int, timestamp: int):
        if self._debounced(gpio):
            return
        log.info("Play/Stop pressed (GPIO %d)", gpio)
        if self._cb_play_stop:
            self._cb_play_stop()

    def _on_vol_up_press(self, chip: int, gpio: int, level: int, timestamp: int):
        if self._debounced(gpio):
            return
        log.info("Volume UP pressed (GPIO %d)", gpio)
        if self._cb_volume_up:
            self._cb_volume_up()

    def _on_vol_down_press(self, chip: int, gpio: int, level: int, timestamp: int):
        if self._debounced(gpio):
            return
        log.info("Volume DOWN pressed (GPIO %d)", gpio)
        if self._cb_volume_down:
            self._cb_volume_down()

    # ── Visual feedback ────────────────────────────────────────────

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
        # Cancel all lgpio callbacks
        for cb in self._callbacks:
            cb.cancel()
        self._callbacks.clear()
        # Close the chip handle
        if self._chip_handle is not None:
            try:
                lgpio.gpiochip_close(self._chip_handle)
            except Exception:
                pass
            self._chip_handle = None
        log.info("GPIO cleaned up")
