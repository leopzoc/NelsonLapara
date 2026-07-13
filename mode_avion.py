"""
Avion Mode — Boeing 737 cabin lighting simulation.

Cycles through the Boeing Sky Interior color phases with smooth
cross-fade transitions on the NeoPixel strip.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import config as cfg

if TYPE_CHECKING:
    from led_strip import LedStrip

log = logging.getLogger(__name__)


class AvionMode:
    """
    Boeing 737 cabin lighting cycle.

    Transitions:  boarding → takeoff → cruise_day → cruise_night → meal → landing
    Each phase holds for AVION_HOLD_SEC, then cross-fades to the next.
    The cycle repeats indefinitely until stopped.
    """

    def __init__(self, led: LedStrip):
        self.led = led
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Avion mode started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        log.info("Avion mode stopped")

    def _loop(self):
        colors = cfg.AVION_COLORS
        idx = 0

        # Set initial color immediately
        self.led.set_color(colors[0]["hex"])

        while not self._stop_event.is_set():
            current = colors[idx]
            next_idx = (idx + 1) % len(colors)
            next_color = colors[next_idx]

            log.info("Avion phase: %s (%ds) → %s", current["name"], current["duration"], next_color["name"])

            # Hold current phase
            if self._wait(current["duration"]):
                return

            # Cross-fade to next
            self.led.fade_to(
                next_color["hex"],
                duration_sec=cfg.AVION_TRANSITION_SEC,
            )
            if self._wait(cfg.AVION_TRANSITION_SEC):
                return

            idx = next_idx

    def _wait(self, seconds: float) -> bool:
        """Wait for *seconds*. Returns True if stop was requested."""
        return self._stop_event.wait(timeout=seconds)
