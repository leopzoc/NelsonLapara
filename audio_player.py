"""
Audio Player — MP3 playback controller using pygame.mixer.

Provides:
  • Play / Pause / Stop
  • Volume up / down (step-based)
  • Track switching (per-mode)
  • Plays independently of mute state (always reproduces even at 0 volume)
  • 15-minute session timer: auto-pauses after AUDIO_SESSION_SEC,
    press Play to resume for another session.  Tracks loop infinitely
    (restart from the beginning when they reach the end).

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
    Thread-safe MP3 audio player with session-based auto-pause.

    Wraps pygame.mixer.music for play/pause/stop and volume control.
    Tracks loop infinitely but auto-pause after AUDIO_SESSION_SEC (15 min).
    Pressing Play resumes for another 15-minute session.
    Volume only controls output level — tracks always reproduce even at 0.
    """

    def __init__(self):
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self._pygame = pygame

        self._volume_db: float = getattr(cfg, "AUDIO_DEFAULT_VOLUME_DB", -12.0)
        self._pygame.mixer.music.set_volume(self._db_to_linear(self._volume_db))

        self._current_track: Optional[str] = None
        self._is_paused: bool = False
        self._lock = threading.Lock()

        # ── Session timer ──────────────────────────────────────────
        self._session_timer: Optional[threading.Timer] = None

        log.info(
            "AudioPlayer initialised — default volume: %.1f dB, "
            "session duration: %ds (%.0f min)",
            self._volume_db, cfg.AUDIO_SESSION_SEC, cfg.AUDIO_SESSION_SEC / 60,
        )

    def _db_to_linear(self, db: float) -> float:
        if db <= -60.0:
            return 0.0
        return 10.0 ** (db / 20.0)

    # ── Session timer management ───────────────────────────────────

    def _start_session_timer(self):
        """Start (or restart) the auto-pause timer."""
        self._cancel_session_timer()
        self._session_timer = threading.Timer(
            cfg.AUDIO_SESSION_SEC, self._on_session_timeout
        )
        self._session_timer.daemon = True
        self._session_timer.start()
        log.info(
            "⏱ Session timer started: auto-pause in %d min",
            cfg.AUDIO_SESSION_SEC // 60,
        )

    def _cancel_session_timer(self):
        """Cancel any running session timer."""
        if self._session_timer is not None:
            self._session_timer.cancel()
            self._session_timer = None

    def _on_session_timeout(self):
        """Called when the 15-minute session expires — auto-pause."""
        with self._lock:
            if self._pygame.mixer.music.get_busy() and not self._is_paused:
                self._pygame.mixer.music.pause()
                self._is_paused = True
                log.info(
                    "⏸ Auto-paused after %d min — press Play to continue",
                    cfg.AUDIO_SESSION_SEC // 60,
                )

    # ── Track management ───────────────────────────────────────────

    def play_track(self, filepath: str) -> None:
        """
        Load and play an MP3 file in infinite loop.

        The track loops continuously (restarts when it ends).
        After AUDIO_SESSION_SEC (15 min) it auto-pauses.
        Press Play to resume for another session.

        Args:
            filepath: Absolute path to the MP3 file.
        """
        with self._lock:
            if not os.path.isfile(filepath):
                log.error("Audio file not found: %s", filepath)
                return

            try:
                self._cancel_session_timer()
                self._pygame.mixer.music.load(filepath)
                self._pygame.mixer.music.set_volume(self._db_to_linear(self._volume_db))
                self._pygame.mixer.music.play(loops=-1)  # infinite loop
                self._current_track = filepath
                self._is_paused = False
                log.info(
                    "▶ Playing: %s (vol=%.1f dB, loop=∞)",
                    os.path.basename(filepath),
                    self._volume_db,
                )
            except Exception as e:
                log.error("Failed to play %s: %s", filepath, e)
                return

        # Start the session timer (outside the lock to avoid deadlock)
        self._start_session_timer()

    def stop(self) -> None:
        """Stop the current track entirely and cancel the session timer."""
        self._cancel_session_timer()
        with self._lock:
            self._pygame.mixer.music.stop()
            self._pygame.mixer.music.unload()
            self._current_track = None
            self._is_paused = False
            log.info("⏹ Audio stopped")

    def toggle_play_pause(self) -> None:
        """
        Toggle between play and pause.

        When resuming (from pause or auto-pause), a new 15-minute
        session timer starts.
        """
        resume = False
        with self._lock:
            if self._is_paused:
                self._pygame.mixer.music.unpause()
                self._is_paused = False
                resume = True
                log.info("▶ Audio resumed — new 15-min session")
            elif self._pygame.mixer.music.get_busy():
                self._cancel_session_timer()
                self._pygame.mixer.music.pause()
                self._is_paused = True
                log.info("⏸ Audio paused")
            else:
                # Nothing is playing — if we have a track loaded, replay it
                if self._current_track:
                    self._pygame.mixer.music.play(loops=-1)
                    self._is_paused = False
                    resume = True
                    log.info("▶ Audio restarted — new 15-min session")

        # Restart session timer on resume (outside lock)
        if resume:
            self._start_session_timer()

    # ── Volume control ─────────────────────────────────────────────

    @property
    def volume_db(self) -> float:
        return self._volume_db

    def volume_up(self) -> float:
        """Increase volume by one step. Returns the new volume in dB."""
        with self._lock:
            self._volume_db = min(0.0, self._volume_db + getattr(cfg, "AUDIO_VOLUME_STEP_DB", 3.0))
            self._pygame.mixer.music.set_volume(self._db_to_linear(self._volume_db))
            log.info("🔊 Volume UP → %.1f dB", self._volume_db)
            return self._volume_db

    def volume_down(self) -> float:
        """Decrease volume by one step. Returns the new volume in dB."""
        with self._lock:
            self._volume_db = max(-60.0, self._volume_db - getattr(cfg, "AUDIO_VOLUME_STEP_DB", 3.0))
            self._pygame.mixer.music.set_volume(self._db_to_linear(self._volume_db))
            log.info("🔉 Volume DOWN → %.1f dB", self._volume_db)
            return self._volume_db

    def set_volume_db(self, level_db: float) -> None:
        """Set volume to a specific dB level."""
        with self._lock:
            self._volume_db = max(-60.0, min(0.0, level_db))
            self._pygame.mixer.music.set_volume(self._db_to_linear(self._volume_db))
            log.info("🔊 Volume set to %.1f dB", self._volume_db)

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
        """Stop playback, cancel timers, and quit the mixer."""
        self._cancel_session_timer()
        try:
            self._pygame.mixer.music.stop()
            self._pygame.mixer.quit()
            log.info("AudioPlayer closed")
        except Exception:
            pass
