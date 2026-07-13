"""
GPIO Button Controller — multi-button system using gpiozero.

Uses gpiozero, which automatically selects the best available GPIO backend
(e.g., lgpio) and handles debouncing and pull-ups transparently.
This is the most robust way to handle buttons on Raspberry Pi 5.

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

from gpiozero import Button

import config as cfg

log = logging.getLogger(__name__)


class Mode(Enum):
    THERAPEUTIC = auto()    # SER + hill-climbing intervention (Autism)
    AVION = auto()          # Boeing 737 cabin lighting
    CIRCADIAN = auto()      # Daylight rhythm cycle


# Ordered cycle list — the button rotates through these
_MODE_CYCLE = [Mode.THERAPEUTIC, Mode.AVION, Mode.CIRCADIAN]

# Map modes to their audio tracks
MODE_AUDIO_TRACKS = {
    Mode.THERAPEUTIC: cfg.AUDIO_TRACK_AUTISM,
    Mode.AVION:       cfg.AUDIO_TRACK_AVION,
    Mode.CIRCADIAN:   cfg.AUDIO_TRACK_CIRCADIAN,
}

class ButtonController:
    """
    GPIO button handler using gpiozero.Button.

    Buttons:
      • Mode cycle (GPIO 17): advances to the next mode.
      • Lamp toggle (GPIO 5): fires ``on_lamp_toggle`` callback.
      • Play/Stop (GPIO 6): fires ``on_play_stop`` callback.
      • Volume Up (GPIO 13): fires ``on_volume_up`` callback.
      • Volume Down (GPIO 19): fires ``on_volume_down`` callback.

    Thread-safe; automatically debounced via gpiozero.
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
        
        self._buttons: list[Button] = []

        # ── Setup all buttons ──────────────────────────────────────
        buttons_config = {
            cfg.BTN_MODE_CYCLE: ("Mode Cycle", self._handle_mode),
            cfg.BTN_LAMP_TOGGLE: ("Lamp Toggle", self._handle_lamp),
            cfg.BTN_PLAY_STOP: ("Play/Stop", self._handle_play_stop),
            cfg.BTN_VOL_UP: ("Volume Up", self._handle_vol_up),
            cfg.BTN_VOL_DOWN: ("Volume Down", self._handle_vol_down),
        }

        for pin, (name, handler) in buttons_config.items():
            try:
                # gpiozero automatically handles pull up/down and debouncing
                btn = Button(
                    pin, 
                    pull_up=True, 
                    bounce_time=cfg.BTN_DEBOUNCE_MS / 1000.0
                )
                btn.when_pressed = handler
                self._buttons.append(btn)
                log.info("✓ Button '%s' registered on GPIO %d", name, pin)
            except Exception as e:
                log.error(
                    "✗ Failed to setup '%s' on GPIO %d: %s — "
                    "pin may be busy or reserved. Change in config.py.",
                    name, pin, e,
                )

        if not self._buttons:
            log.error("No buttons could be registered.")

        log.info(
            "ButtonController ready — %d/%d buttons active, default mode: %s",
            len(self._buttons), len(buttons_config), self._current_mode.name,
        )

    @property
    def current_mode(self) -> Mode:
        with self._lock:
            return self._current_mode

    # ── Button handlers ────────────────────────────────────────────

    def _handle_mode(self):
        with self._lock:
            idx = _MODE_CYCLE.index(self._current_mode)
            new_mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
            old_mode = self._current_mode
            self._current_mode = new_mode

        log.info("Mode switch: %s → %s", old_mode.name, new_mode.name)
        
        if self._cb_mode_change:
            self._cb_mode_change(new_mode)

    def _handle_lamp(self):
        log.info("Lamp toggle pressed")
        if self._cb_lamp_toggle:
            self._cb_lamp_toggle()

    def _handle_play_stop(self):
        log.info("Play/Stop pressed")
        if self._cb_play_stop:
            self._cb_play_stop()

    def _handle_vol_up(self):
        log.info("Volume UP pressed")
        if self._cb_volume_up:
            self._cb_volume_up()

    def _handle_vol_down(self):
        log.info("Volume DOWN pressed")
        if self._cb_volume_down:
            self._cb_volume_down()

    def close(self):
        for btn in self._buttons:
            try:
                btn.close()
            except Exception:
                pass
        self._buttons.clear()
        log.info("GPIO cleaned up")
