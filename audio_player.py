"""
Audio Player — MP3 playback controller using pygame.mixer.

Provides:
  • Play / Pause / Stop
  • Volume up / down (step-based)
  • Track switching (per-mode)
  • Plays independently of mute state (always reproduces even at 0 volume)

Each operating mode has its own dedicated audio track:
  • Avion     → Cabin ambience (ASMR white noise)
  • Circadian → Pure white noise
  • Autism    → Neuro-relaxation therapy music

The player uses pygame.mixer for cross-platform MP3 support.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import config as cfg

log = logging.getLogger(__name__)


class AudioPlayer:
    """
    Thread-safe MP3 audio player.

    Wraps pygame.mixer.music for play/pause/stop and volume control.
    Tracks always play (even at volume 0) — volume only controls output level.
    """

    def __init__(self):
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self._pygame = pygame

        self._volume: float = cfg.AUDIO_DEFAULT_VOLUME  # 0.0 – 1.0
        self._pygame.mixer.music.set_volume(self._volume)

        self._current_track: Optional[str] = None
        self._is_paused: bool = False
        self._lock = threading.Lock()

        log.info(
            "AudioPlayer initialised — default volume: %.0f%%",
            self._volume * 100,
        )

    # ── Track management ───────────────────────────────────────────

    def play_track(self, filepath: str, loops: int = -1) -> None:
        """
        Load and play an MP3 file.

        Args:
            filepath: Absolute path to the MP3 file.
            loops: Number of loops (-1 = infinite loop).
        """
        with self._lock:
            if not os.path.isfile(filepath):
                log.error("Audio file not found: %s", filepath)
                return

            try:
                self._pygame.mixer.music.load(filepath)
                self._pygame.mixer.music.set_volume(self._volume)
                self._pygame.mixer.music.play(loops=loops)
                self._current_track = filepath
                self._is_paused = False
                log.info(
                    "▶ Playing: %s (vol=%.0f%%)",
                    os.path.basename(filepath),
                    self._volume * 100,
                )
            except Exception as e:
                log.error("Failed to play %s: %s", filepath, e)

    def stop(self) -> None:
        """Stop the current track entirely."""
        with self._lock:
            self._pygame.mixer.music.stop()
            self._pygame.mixer.music.unload()
            self._current_track = None
            self._is_paused = False
            log.info("⏹ Audio stopped")

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        with self._lock:
            if self._is_paused:
                self._pygame.mixer.music.unpause()
                self._is_paused = False
                log.info("▶ Audio resumed")
            elif self._pygame.mixer.music.get_busy():
                self._pygame.mixer.music.pause()
                self._is_paused = True
                log.info("⏸ Audio paused")
            else:
                # Nothing is playing — if we have a track loaded, replay it
                if self._current_track:
                    self._pygame.mixer.music.play(loops=-1)
                    self._is_paused = False
                    log.info("▶ Audio restarted")

    # ── Volume control ─────────────────────────────────────────────

    @property
    def volume(self) -> float:
        return self._volume

    def volume_up(self) -> float:
        """Increase volume by one step. Returns the new volume."""
        with self._lock:
            self._volume = min(1.0, self._volume + cfg.AUDIO_VOLUME_STEP)
            self._pygame.mixer.music.set_volume(self._volume)
            log.info("🔊 Volume UP → %.0f%%", self._volume * 100)
            return self._volume

    def volume_down(self) -> float:
        """Decrease volume by one step. Returns the new volume."""
        with self._lock:
            self._volume = max(0.0, self._volume - cfg.AUDIO_VOLUME_STEP)
            self._pygame.mixer.music.set_volume(self._volume)
            log.info("🔉 Volume DOWN → %.0f%%", self._volume * 100)
            return self._volume

    def set_volume(self, level: float) -> None:
        """Set volume to a specific level (0.0 – 1.0)."""
        with self._lock:
            self._volume = max(0.0, min(1.0, level))
            self._pygame.mixer.music.set_volume(self._volume)

    # ── State queries ──────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._pygame.mixer.music.get_busy() and not self._is_paused

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def current_track(self) -> Optional[str]:
        return self._current_track

    # ── Cleanup ────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop playback and quit the mixer."""
        try:
            self._pygame.mixer.music.stop()
            self._pygame.mixer.quit()
            log.info("AudioPlayer closed")
        except Exception:
            pass
