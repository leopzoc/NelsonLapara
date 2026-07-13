"""
LED Strip Driver — WS2812B NeoPixel control via Adafruit SPI.

Provides:
  • Solid color fill (active zones only: first N + last N LEDs)
  • Smooth cross-fade transitions
  • Hex-to-RGB conversion
  • Brightness control

Only the first LED_ACTIVE_HEAD and last LED_ACTIVE_TAIL LEDs are driven.
Middle LEDs are kept off to avoid hardware issues.

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

        # ── Skip zone ──────────────────────────────────────────────
        # LEDs in the skip range are kept off; all others are active.
        skip_start = cfg.LED_SKIP_START
        skip_end = cfg.LED_SKIP_END
        self._part1_indices = list(range(0, skip_start))
        self._part2_indices = list(range(skip_end + 1, num_leds))
        self._inactive_indices = list(range(skip_start, skip_end + 1))

        self._current_rgb: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((0, 0, 0), (0, 0, 0))
        self._lock = threading.Lock()
        self._fade_thread: Optional[threading.Thread] = None
        self._fade_cancel = threading.Event()

        log.info(
            "NeoPixel SPI strip initialised: %d LEDs on SPI0 (GPIO 10), "
            "skip zone: %d–%d (%d off), %d active",
            num_leds, skip_start, skip_end,
            len(self._inactive_indices), len(self._part1_indices) + len(self._part2_indices),
        )

    # ── immediate set ──────────────────────────────────────────────

    def set_color(self, hex_color: str | list[str]) -> None:
        """Instantly fill the entire strip with a solid color, or a pair of colors."""
        if isinstance(hex_color, list):
            rgb1 = hex_to_rgb(hex_color[0])
            rgb2 = hex_to_rgb(hex_color[1])
        else:
            rgb1 = rgb2 = hex_to_rgb(hex_color)
            
        self._cancel_fade()
        self._fill_dual_rgb(rgb1, rgb2)

    def set_color_rgb(self, r: int, g: int, b: int) -> None:
        self._cancel_fade()
        self._fill_dual_rgb((r, g, b), (r, g, b))

    def off(self) -> None:
        """Turn off all LEDs."""
        self.set_color("#000000")

    # ── cross-fade ─────────────────────────────────────────────────

    def fade_to(
        self,
        hex_color: str | list[str],
        duration_sec: float = 3.0,
        steps: int = 60,
    ) -> None:
        """
        Smoothly cross-fade from the current color to *hex_color*.
        Runs in a background thread; a new call cancels any ongoing fade.
        """
        if isinstance(hex_color, list):
            target_rgb1 = hex_to_rgb(hex_color[0])
            target_rgb2 = hex_to_rgb(hex_color[1])
        else:
            target_rgb1 = target_rgb2 = hex_to_rgb(hex_color)

        self._cancel_fade()
        self._fade_cancel.clear()

        self._fade_thread = threading.Thread(
            target=self._fade_worker,
            args=(self._current_rgb[0], self._current_rgb[1], target_rgb1, target_rgb2, duration_sec, steps),
            daemon=True,
        )
        self._fade_thread.start()

    def _fade_worker(
        self,
        start1: Tuple[int, int, int],
        start2: Tuple[int, int, int],
        end1: Tuple[int, int, int],
        end2: Tuple[int, int, int],
        duration: float,
        steps: int,
    ):
        step_delay = duration / steps
        for i in range(1, steps + 1):
            if self._fade_cancel.is_set():
                return
            t = i / steps
            rgb1 = lerp_color(start1, end1, t)
            rgb2 = lerp_color(start2, end2, t)
            self._fill_dual_rgb(rgb1, rgb2)
            time.sleep(step_delay)

    def _cancel_fade(self):
        self._fade_cancel.set()
        if self._fade_thread and self._fade_thread.is_alive():
            self._fade_thread.join(timeout=1.0)

    # ── low-level ──────────────────────────────────────────────────

    def _fill_dual_rgb(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> None:
        """Fill part1 with rgb1 and part2 with rgb2; middle stays off."""
        with self._lock:
            for i in self._part1_indices:
                self.strip[i] = rgb1
            for i in self._part2_indices:
                self.strip[i] = rgb2
            for i in self._inactive_indices:
                self.strip[i] = (0, 0, 0)
            self.strip.show()
            self._current_rgb = (rgb1, rgb2)

    def set_brightness(self, brightness: int) -> None:
        """Set global brightness (0–255)."""
        with self._lock:
            self.strip.brightness = min(255, max(0, brightness)) / 255.0
            self.strip.show()

    def close(self) -> None:
        self._cancel_fade()
        self.off()
