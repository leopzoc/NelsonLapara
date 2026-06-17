"""
Audio Pipeline — Real-time chunked capture with VAD (RPi 5 optimised).

Uses sounddevice for low-latency ALSA/PulseAudio access.  Provides:
  • Non-blocking ring-buffer capture
  • RMS / dB calculation
  • Simple energy-based VAD
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

import config as cfg


# ── Utilities ───────────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    """Root-mean-square of a 1-D float32 array."""
    return float(np.sqrt(np.mean(audio ** 2)))


def rms_to_db(rms_val: float, ref: float = 1.0) -> float:
    """Convert RMS amplitude to decibels (dBFS)."""
    if rms_val < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms_val / ref)


def has_speech(audio: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> bool:
    """Energy-based VAD: checks if enough frames exceed the RMS threshold."""
    frame_len = int(0.025 * sr)   # 25 ms frames
    hop = int(0.010 * sr)         # 10 ms hop
    voiced_frames = 0
    total_frames = 0

    for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start : start + frame_len]
        total_frames += 1
        if rms(frame) > cfg.VAD_RMS_THRESHOLD:
            voiced_frames += 1

    if total_frames == 0:
        return False
    voiced_ratio = voiced_frames / total_frames
    min_ratio = cfg.VAD_MIN_SPEECH_SEC / (len(audio) / sr)
    return voiced_ratio >= min_ratio


# ── Streamer ────────────────────────────────────────────────────────

class AudioStreamer:
    """
    Non-blocking audio capture that fills fixed-length windows.

    Usage:
        streamer = AudioStreamer()
        streamer.start()
        chunk = streamer.get_chunk()   # blocks until a window is ready
        streamer.stop()
    """

    def __init__(
        self,
        window_sec: float = cfg.LISTEN_WINDOW_SEC,
        sr: int = cfg.SAMPLE_RATE,
        channels: int = cfg.CHANNELS,
    ):
        self.sr = sr
        self.channels = channels
        self.window_samples = int(window_sec * sr)
        self._buffer: np.ndarray = np.zeros(self.window_samples, dtype=np.float32)
        self._write_pos = 0
        self._ready = threading.Event()
        self._chunks: deque[np.ndarray] = deque(maxlen=4)
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    # ── callbacks ───────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            pass   # drop xruns silently on RPi
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        remaining = self.window_samples - self._write_pos
        take = min(len(mono), remaining)
        self._buffer[self._write_pos : self._write_pos + take] = mono[:take]
        self._write_pos += take

        if self._write_pos >= self.window_samples:
            self._chunks.append(self._buffer.copy())
            self._buffer[:] = 0.0
            self._write_pos = 0
            self._ready.set()

    # ── public API ──────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype=cfg.DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Block until a full window is captured. Returns None on timeout."""
        if self._ready.wait(timeout=timeout):
            self._ready.clear()
            return self._chunks.pop() if self._chunks else None
        return None

    def record_fixed(self, duration_sec: float) -> np.ndarray:
        """Synchronous single-shot recording of *duration_sec* seconds."""
        n_samples = int(duration_sec * self.sr)
        audio = sd.rec(
            n_samples,
            samplerate=self.sr,
            channels=self.channels,
            dtype=cfg.DTYPE,
        )
        sd.wait()
        return audio.flatten()
