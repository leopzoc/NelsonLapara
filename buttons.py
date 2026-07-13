"""
GPIO Button Controller — multi-button system using gpiod (libgpiod2).

Uses the Linux kernel GPIO character device interface (/dev/gpiochip4)
which is completely independent from lgpio and does NOT conflict with
adafruit-blinka's SPI driver.

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
import select
import threading
import time
from datetime import timedelta
from enum import Enum, auto
from typing import Callable, Optional

import gpiod
from gpiod.line_settings import LineSettings

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
    GPIO button handler using gpiod (libgpiod2, kernel char device).

    This uses /dev/gpiochip4 (RPi 5) via the Linux kernel interface,
    which is completely separate from lgpio and cannot conflict with
    adafruit-blinka's NeoPixel SPI driver.

    A background polling thread waits for edge events and dispatches
    to the appropriate callback.

    Buttons:
      • Mode cycle (GPIO 17): advances to the next mode.
      • Lamp toggle (GPIO 5): fires ``on_lamp_toggle`` callback.
      • Play/Stop (GPIO 6): fires ``on_play_stop`` callback.
      • Volume Up (GPIO 13): fires ``on_volume_up`` callback.
      • Volume Down (GPIO 19): fires ``on_volume_down`` callback.

    Thread-safe; debounced via hardware (gpiod debounce_period).
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
        self._request = None
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

        # Map GPIO pin → (name, handler)
        self._pin_handlers = {
            cfg.BTN_MODE_CYCLE: ("Mode Cycle", self._handle_mode),
            cfg.BTN_LAMP_TOGGLE: ("Lamp Toggle", self._handle_lamp),
            cfg.BTN_PLAY_STOP: ("Play/Stop", self._handle_play_stop),
            cfg.BTN_VOL_UP: ("Volume Up", self._handle_vol_up),
            cfg.BTN_VOL_DOWN: ("Volume Down", self._handle_vol_down),
        }

        self._requests = []
        self._active_pin_handlers = {}

        # ── Open GPIO chip ─────────────────────────────────────────
        # RPi 5 = /dev/gpiochip4, RPi 4 = /dev/gpiochip0
        chip_path = None
        for path in ("/dev/gpiochip4", "/dev/gpiochip0"):
            try:
                with open(path):
                    chip_path = path
                    break
            except FileNotFoundError:
                continue

        if chip_path is None:
            log.error("No GPIO chip found! Buttons will not work.")
            return

        # ── Request lines individually ─────────────────────────────
        # We request pins one by one so that if one pin is busy,
        # it doesn't fail the whole request.
        for pin, (name, handler) in self._pin_handlers.items():
            try:
                line_config = {
                    pin: LineSettings(
                        direction=gpiod.line.Direction.INPUT,
                        bias=gpiod.line.Bias.PULL_UP,
                        edge_detection=gpiod.line.Edge.FALLING,
                        debounce_period=timedelta(milliseconds=cfg.BTN_DEBOUNCE_MS),
                    )
                }
                req = gpiod.request_lines(
                    chip_path,
                    consumer=f"onix-btn-{pin}",
                    config=line_config,
                )
                self._requests.append(req)
                self._active_pin_handlers[pin] = (name, handler)
                log.info("✓ Button '%s' registered on GPIO %d", name, pin)
            except Exception as e:
                log.error(
                    "✗ Failed to setup '%s' on GPIO %d: %s — "
                    "pin may be busy or reserved. Change in config.py.",
                    name, pin, e,
                )

        if not self._requests:
            log.error("No buttons could be registered. Button polling disabled.")
            return

        # ── Start polling thread ───────────────────────────────────
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poll_thread.start()

        log.info(
            "ButtonController ready — %d buttons, default mode: %s",
            len(self._pin_handlers), self._current_mode.name,
        )

    @property
    def current_mode(self) -> Mode:
        with self._lock:
            return self._current_mode

    # ── Polling loop (background thread) ───────────────────────────

    def _poll_loop(self):
        """Poll for GPIO edge events using select() on individual line descriptors."""
        while not self._stop_event.is_set():
            try:
                fd_to_req = {req.fd: req for req in self._requests}
                # Wait up to 1 second for any edge event
                ready, _, _ = select.select(fd_to_req.keys(), [], [], 1.0)
                
                for fd in ready:
                    req = fd_to_req[fd]
                    events = req.read_edge_events()
                    for event in events:
                        pin = event.line_offset
                        if pin in self._active_pin_handlers:
                            name, handler = self._active_pin_handlers[pin]
                            log.info("Button '%s' pressed (GPIO %d)", name, pin)
                            threading.Thread(target=handler, daemon=True).start()
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error("Poll error: %s", e)
                    time.sleep(0.5)

    # ── Button handlers ────────────────────────────────────────────

    def _handle_mode(self):
        with self._lock:
            idx = _MODE_CYCLE.index(self._current_mode)
            new_mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
            old_mode = self._current_mode
            self._current_mode = new_mode

        log.info("Mode switch: %s → %s", old_mode.name, new_mode.name)
        self._flash_feedback(new_mode)

        if self._cb_mode_change:
            self._cb_mode_change(new_mode)

    def _handle_lamp(self):
        if self._cb_lamp_toggle:
            self._cb_lamp_toggle()

    def _handle_play_stop(self):
        if self._cb_play_stop:
            self._cb_play_stop()

    def _handle_vol_up(self):
        if self._cb_volume_up:
            self._cb_volume_up()

    def _handle_vol_down(self):
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
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=3.0)
        
        for req in self._requests:
            try:
                req.release()
            except Exception:
                pass
        self._requests.clear()
        log.info("GPIO cleaned up")
