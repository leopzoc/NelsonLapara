"""
Circadian Mode — Daylight rhythm cycle on the LED strip.

Reads the current local time and continuously adjusts the strip color
to match the natural light temperature defined in config.CIRCADIAN_SCHEDULE.
Cross-fades smoothly between time blocks.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple

import config as cfg
from led_strip import hex_to_rgb, lerp_color, rgb_to_hex

if TYPE_CHECKING:
    from led_strip import LedStrip

log = logging.getLogger(__name__)


def _get_circadian_color(now: Optional[datetime] = None) -> Tuple[str, str]:
    """
    Determine the current circadian color based on time of day.

    Returns (hex_color, description).
    Handles wrap-around for overnight slots (e.g. 23:00 → 06:00).
    """
    if now is None:
        now = datetime.now()
    hour = now.hour + now.minute / 60.0

    schedule = cfg.CIRCADIAN_SCHEDULE

    for start_h, end_h, hex_color, desc in schedule:
        if start_h < end_h:
            # Normal range (e.g. 8–10)
            if start_h <= hour < end_h:
                return hex_color, desc
        else:
            # Wrap-around (e.g. 23–6)
            if hour >= start_h or hour < end_h:
                return hex_color, desc

    # Fallback
    return schedule[0][2], schedule[0][3]


def _get_blended_circadian_color(now: Optional[datetime] = None) -> str:
    """
    Compute a smoothly interpolated color between the current and next
    circadian blocks based on how far through the current block we are.
    """
    if now is None:
        now = datetime.now()
    hour = now.hour + now.minute / 60.0

    schedule = cfg.CIRCADIAN_SCHEDULE

    # Find current block index
    current_idx = 0
    for i, (start_h, end_h, _, _) in enumerate(schedule):
        if start_h < end_h:
            if start_h <= hour < end_h:
                current_idx = i
                break
        else:
            if hour >= start_h or hour < end_h:
                current_idx = i
                break

    cur_start, cur_end, cur_hex, _ = schedule[current_idx]
    next_idx = (current_idx + 1) % len(schedule)
    _, _, next_hex, _ = schedule[next_idx]

    # Calculate progress through current block
    if cur_start < cur_end:
        block_duration = cur_end - cur_start
        progress = (hour - cur_start) / block_duration
    else:
        # Wrap-around block
        block_duration = (24 - cur_start) + cur_end
        if hour >= cur_start:
            progress = (hour - cur_start) / block_duration
        else:
            progress = (24 - cur_start + hour) / block_duration

    progress = max(0.0, min(1.0, progress))

    # Only blend in the last 30% of the block (transition zone)
    if progress < 0.7:
        return cur_hex
    else:
        blend_t = (progress - 0.7) / 0.3  # 0→1 over last 30%
        rgb = lerp_color(hex_to_rgb(cur_hex), hex_to_rgb(next_hex), blend_t)
        return rgb_to_hex(*rgb)


class CircadianMode:
    """
    Continuous circadian lighting.

    Updates the LED strip every 30 seconds based on the current time,
    with smooth interpolation between time blocks.
    """

    UPDATE_INTERVAL_SEC = 30.0

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
        log.info("Circadian mode started")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        log.info("Circadian mode stopped")

    def _loop(self):
        last_color = ""

        while not self._stop_event.is_set():
            color = _get_blended_circadian_color()

            if color != last_color:
                _, desc = _get_circadian_color()
                log.info("Circadian → %s (%s)", color, desc)
                self.led.fade_to(
                    color,
                    duration_sec=min(cfg.CIRCADIAN_TRANSITION_SEC, self.UPDATE_INTERVAL_SEC),
                )
                last_color = color

            self._stop_event.wait(timeout=self.UPDATE_INTERVAL_SEC)
