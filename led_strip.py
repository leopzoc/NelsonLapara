"""
LED Strip Driver — WS2812B NeoPixel control via Adafruit SPI.

Provides:
  • Solid color fill
  • Smooth cross-fade transitions
  • Hex-to-RGB conversion
  • Brightness control

Requires: 
  • SPI enabled on RPi (sudo raspi-config)
  • GPIO 10 (SPI0 MOSI) connected to NeoPixel Data IN
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Optional, Tuple

import config as cfg

log = logging.getLogger(__name__)

# ── Hex / RGB helpers ───────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colors, t ∈ [0, 1]."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# ── NeoPixel Driver ────────────────────────────────────────────────

class LedStrip:
    """
    WS2812B NeoPixel strip controller.

    Uses adafruit-circuitpython-neopixel-spi for RPi 5 compatibility.
    All methods are thread-safe; cross-fades run in a background thread.
    """

    def __init__(
        self,
        num_leds: int = cfg.LED_COUNT,
        pin: int = cfg.LED_PIN,
        brightness: int = cfg.LED_BRIGHTNESS,
    ):
        import board
        import neopixel_spi

        self.num_leds = num_leds
        
        # Adafruit SPI neopixel uses a float 0.0 - 1.0 for brightness
        initial_brightness = min(255, max(0, brightness)) / 255.0

        # board.SPI() maps to /dev/spidev0.0 (GPIO 10)
        self.strip = neopixel_spi.NeoPixel_SPI(
            board.SPI(),
            num_leds,
            brightness=initial_brightness,
            auto_write=False,
            pixel_order=neopixel_spi.GRB
        )

        self._current_rgb: Tuple[int, int, int] = (0, 0, 0)
        self._lock = threading.Lock()
        self._fade_thread: Optional[threading.Thread] = None
        self._fade_cancel = threading.Event()

        log.info(
            "NeoPixel SPI strip initialised: %d LEDs on SPI0 (GPIO 10)",
            num_leds,
        )

    # ── immediate set ──────────────────────────────────────────────

    def set_color(self, hex_color: str) -> None:
        """Instantly fill the entire strip with a solid color."""
        rgb = hex_to_rgb(hex_color)
        self._cancel_fade()
        self._fill_rgb(rgb)

    def set_color_rgb(self, r: int, g: int, b: int) -> None:
        self._cancel_fade()
        self._fill_rgb((r, g, b))

    def off(self) -> None:
        """Turn off all LEDs."""
        self.set_color("#000000")

    # ── cross-fade ─────────────────────────────────────────────────

    def fade_to(
        self,
        hex_color: str,
        duration_sec: float = 3.0,
        steps: int = 60,
    ) -> None:
        """
        Smoothly cross-fade from the current color to *hex_color*.
        Runs in a background thread; a new call cancels any ongoing fade.
        """
        target_rgb = hex_to_rgb(hex_color)
        self._cancel_fade()
        self._fade_cancel.clear()

        self._fade_thread = threading.Thread(
            target=self._fade_worker,
            args=(self._current_rgb, target_rgb, duration_sec, steps),
            daemon=True,
        )
        self._fade_thread.start()

    def _fade_worker(
        self,
        start: Tuple[int, int, int],
        end: Tuple[int, int, int],
        duration: float,
        steps: int,
    ):
        step_delay = duration / steps
        for i in range(1, steps + 1):
            if self._fade_cancel.is_set():
                return
            t = i / steps
            rgb = lerp_color(start, end, t)
            self._fill_rgb(rgb)
            time.sleep(step_delay)

    def _cancel_fade(self):
        self._fade_cancel.set()
        if self._fade_thread and self._fade_thread.is_alive():
            self._fade_thread.join(timeout=1.0)

    # ── low-level ──────────────────────────────────────────────────

    def _fill_rgb(self, rgb: Tuple[int, int, int]) -> None:
        with self._lock:
            for i in range(self.num_leds):
                self.strip[i] = rgb
            self.strip.show()
            self._current_rgb = rgb

    def set_brightness(self, brightness: int) -> None:
        """Set global brightness (0–255)."""
        with self._lock:
            self.strip.brightness = min(255, max(0, brightness)) / 255.0
            self.strip.show()

    def close(self) -> None:
        self._cancel_fade()
        self.off()
