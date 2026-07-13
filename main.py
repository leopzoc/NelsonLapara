"""
Main Orchestrator — Closed-Loop Adaptive Emotional Intervention.

Runs on RPi 5.  Hardware:
  • WS2812B NeoPixel strip (144 LED) on SPI0 (GPIO 10)
  • 5 physical buttons:
      - GPIO 17: cycle modes (Therapeutic → Avion → Circadian → Autism)
      - GPIO  5: lamp on/off (LED strip toggle)
      - GPIO  6: play/pause audio track
      - GPIO 13: volume up
      - GPIO 19: volume down
  • USB microphone
  • Audio output (speaker / 3.5mm jack)

Architecture:

    ┌───────────┐     ┌───────────┐     ┌──────────────────┐
    │  Buttons  │────▶│   Main    │────▶│  Active Mode     │
    │  GPIO     │     │  Orchest. │     │  ┌─ Therapeutic   │
    │ 17,5,6,   │     │           │     │  ├─ Avion         │
    │ 13,19     │     │           │     │  ├─ Circadian     │
    └───────────┘     └─────┬─────┘     │  └─ Autism (SER)  │
                            │           └────────┬─────────┘
    ┌───────────┐     ┌─────▼─────┐              │
    │    Mic    │────▶│ SER Engine│     ┌────────▼─────────┐
    │ (stream)  │     │ (Wav2Vec2)│     │  NeoPixel Strip  │
    └───────────┘     └───────────┘     │  (144 WS2812B)   │
                                        └──────────────────┘
    ┌──────────────────────────────┐
    │  AudioPlayer (pygame.mixer)  │
    │  Per-mode MP3 tracks         │
    └──────────────────────────────┘

Modes:
  • Therapeutic: SER + hill-climbing color mitigation (no audio)
  • Avion:       Boeing 737 cabin lighting cycle + cabin ambience audio
  • Circadian:   Daylight rhythm LED cycle + white noise audio
  • Autism:      SER + hill-climbing color mitigation + neuro-relaxation audio
"""

from __future__ import annotations

import logging
import signal
import time
from enum import Enum, auto

import config as cfg
from audio_pipeline import AudioStreamer, has_speech, rms, rms_to_db
from ser_engine import SEREngine
from color_mitigation import ColorMitigation
from led_strip import LedStrip
from buttons import ButtonController, Mode, MODE_AUDIO_TRACKS
from mode_avion import AvionMode
from mode_circadian import CircadianMode
from audio_player import AudioPlayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Therapeutic sub-states ─────────────────────────────────────────

class TherapeuticState(Enum):
    LISTENING = auto()
    MITIGATING = auto()
    COOLDOWN = auto()


COOLDOWN_SEC = 15.0
MODE_FADE_SEC = 1.5   # Fade-out / fade-in duration on mode switch

# Modes that use the SER + ColorMitigation pipeline
_SER_MODES = {Mode.THERAPEUTIC}


# ── Main System ────────────────────────────────────────────────────

class InterventionSystem:
    """
    Central orchestrator.

    Manages mode switching (button-driven) and delegates to:
      • Therapeutic mode — SER + hill-climbing on the LED strip (no audio)
      • Avion mode       — Boeing 737 cabin lighting cycle + audio
      • Circadian mode   — Daylight rhythm LED cycle + audio
      • Autism mode      — SER + hill-climbing on the LED strip + audio
    """

    def __init__(self):
        log.info("Initialising subsystems …")

        # ── Hardware ────────────────────────────────────────────────
        self.led = LedStrip()
        self.streamer = AudioStreamer()
        self.audio = AudioPlayer()

        # ── Modes ───────────────────────────────────────────────────
        self.ser = SEREngine()
        self.mitigation = ColorMitigation()
        self.avion = AvionMode(self.led)
        self.circadian = CircadianMode(self.led)

        # ── Buttons (5 buttons) ────────────────────────────────────
        self.buttons = ButtonController(
            on_mode_change=self._on_mode_change,
            on_lamp_toggle=self._on_lamp_toggle,
            on_play_stop=self._on_play_stop,
            on_volume_up=self._on_volume_up,
            on_volume_down=self._on_volume_down,
            led_strip=self.led,
        )

        # ── State ───────────────────────────────────────────────────
        self._active_mode: Mode = Mode.THERAPEUTIC
        self._therapeutic_state = TherapeuticState.LISTENING
        self._cooldown_until = 0.0
        self._running = False
        self._lamp_on = True   # LED strip starts ON

    # ── Mode switching ─────────────────────────────────────────────

    def _on_mode_change(self, new_mode: Mode):
        """Callback fired by the ButtonController on GPIO interrupt."""
        log.info("⚙ Switching to %s", new_mode.name)
        self._stop_current_mode()
        self._active_mode = new_mode
        self._start_mode(new_mode)

    def _stop_current_mode(self):
        """Cleanly stop whichever mode is currently running."""
        # Fade out LED strip smoothly before stopping
        if self._lamp_on:
            self.led.fade_to("#000000", duration_sec=MODE_FADE_SEC)
            time.sleep(MODE_FADE_SEC)
        self.avion.stop()
        self.circadian.stop()
        self.mitigation.reset()
        self._therapeutic_state = TherapeuticState.LISTENING
        # Audio is stopped and restarted per-mode in _start_mode

    def _start_mode(self, mode: Mode):
        # ── Start the audio track for this mode ────────────────────
        # Audio always plays (even at volume 0) — it reproduces
        # independently of the mute state.
        track = MODE_AUDIO_TRACKS.get(mode)
        if track:
            self.audio.play_track(track)
            log.info("♪ Audio track started for %s", mode.name)
        else:
            self.audio.stop()
            log.info("♪ No audio for %s — stopped", mode.name)

        # ── Start the LED mode (with smooth fade-in) ───────────────
        match mode:
            case Mode.THERAPEUTIC:
                log.info("▶ Therapeutic mode — SER + color mitigation (Autism)")
                # LEDs stay off until SER triggers them
            case Mode.AVION:
                log.info("▶ Avion mode — Boeing 737 cabin lighting")
                self.avion.start()
            case Mode.CIRCADIAN:
                log.info("▶ Circadian mode — daylight rhythm cycle")
                self.circadian.start()

    # ── Lamp toggle ────────────────────────────────────────────────

    def _on_lamp_toggle(self):
        """Toggle the LED strip on/off."""
        self._lamp_on = not self._lamp_on
        if self._lamp_on:
            log.info("💡 Lamp ON")
            # Re-enter the current mode to restore LED behavior
            self._start_mode_leds_only(self._active_mode)
        else:
            log.info("💡 Lamp OFF")
            self.led.off()

    def _start_mode_leds_only(self, mode: Mode):
        """Restart LED behavior for the current mode (without touching audio)."""
        match mode:
            case Mode.THERAPEUTIC | Mode.AUTISM:
                self.led.set_color("#000000")  # Will light up on SER trigger
            case Mode.AVION:
                # Avion thread is already running, just needs to continue
                if not self.avion.running:
                    self.avion.start()
            case Mode.CIRCADIAN:
                if not self.circadian.running:
                    self.circadian.start()

    # ── Audio controls ─────────────────────────────────────────────

    def _on_play_stop(self):
        """Toggle play/pause on the current audio track."""
        self.audio.toggle_play_pause()

    def _on_volume_up(self):
        """Increase audio volume."""
        self.audio.volume_up()

    def _on_volume_down(self):
        """Decrease audio volume."""
        self.audio.volume_down()

    # ── Main loop ──────────────────────────────────────────────────

    def run(self):
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.streamer.start()
        self._start_mode(self._active_mode)
        log.info("System online — mode: %s", self._active_mode.name)

        try:
            while self._running:
                if self._active_mode in _SER_MODES:
                    self._therapeutic_tick()
                else:
                    # Avion and Circadian run in their own threads;
                    # main loop just idles, waiting for button press.
                    time.sleep(0.5)
        finally:
            self.shutdown()

    def shutdown(self):
        log.info("Shutting down …")
        self._running = False
        self._stop_current_mode()
        self.audio.close()
        self.streamer.stop()
        self.led.close()
        self.buttons.close()

    def _handle_signal(self, sig, frame):
        log.info("Signal %s received", sig)
        self._running = False

    # ── THERAPEUTIC / AUTISM mode ticks (SER + ColorMitigation) ────

    def _therapeutic_tick(self):
        match self._therapeutic_state:
            case TherapeuticState.LISTENING:
                self._listen_tick()
            case TherapeuticState.MITIGATING:
                self._mitigate_tick()
            case TherapeuticState.COOLDOWN:
                self._cooldown_tick()

    def _listen_tick(self):
        chunk = self.streamer.get_chunk(timeout=6.0)
        if chunk is None:
            return

        # Check if mode changed during wait
        if self._active_mode not in _SER_MODES:
            return

        if not has_speech(chunk):
            return

        emotion, arousal, confidence = self.ser.predict(chunk)

        if arousal in cfg.AROUSAL_TRIGGER_STATES and confidence > 0.45:
            log.info(
                "⚡ Trigger: %s (arousal=%s, conf=%.2f)",
                emotion, arousal, confidence,
            )
            initial_db = rms_to_db(rms(chunk))
            self.mitigation.trigger(initial_db)
            self._therapeutic_state = TherapeuticState.MITIGATING

    def _mitigate_tick(self):
        if self._active_mode not in _SER_MODES:
            return

        action = self.mitigation.tick()

        match action["action"]:
            case "show_color":
                if self._lamp_on:
                    self.led.fade_to(action["color"], duration_sec=1.5)
                log.info("LED strip → %s", action["color"])

            case "record":
                audio = self.streamer.record_fixed(action["duration"])
                db = rms_to_db(rms(audio))
                self.mitigation.feed_baseline_db(db)

            case "wait_then_record":
                time.sleep(action["wait"])
                if self._active_mode not in _SER_MODES:
                    return
                audio = self.streamer.record_fixed(action["duration"])
                db = rms_to_db(rms(audio))
                result = self.mitigation.evaluate(db)
                self._handle_eval_result(result)

            case "evaluate":
                pass

            case "noop" | "done":
                self._enter_cooldown()

    def _handle_eval_result(self, result: dict):
        match result["action"]:
            case "continue":
                log.info("✓ dB improved — continuing desaturation")
            case "rollback":
                if self._lamp_on:
                    self.led.fade_to(result["color"], duration_sec=1.5)
                log.info("↩ Rolled back → %s", result["color"])
            case "converged":
                if self._lamp_on:
                    self.led.fade_to(result["color"], duration_sec=1.5)
                log.info("● Converged at %s", result["color"])
                self._enter_cooldown()

    # ── COOLDOWN ───────────────────────────────────────────────────

    def _enter_cooldown(self):
        self._therapeutic_state = TherapeuticState.COOLDOWN
        self._cooldown_until = time.monotonic() + COOLDOWN_SEC
        self.mitigation.reset()
        log.info("Cooldown for %.0f s", COOLDOWN_SEC)

    def _cooldown_tick(self):
        if time.monotonic() >= self._cooldown_until:
            log.info("Cooldown complete — resuming listening")
            if self._lamp_on:
                self.led.fade_to("#000000", duration_sec=1.5)
            self._therapeutic_state = TherapeuticState.LISTENING
        else:
            time.sleep(0.5)


# ── Entry point ─────────────────────────────────────────────────────

def main():
    system = InterventionSystem()
    system.run()


if __name__ == "__main__":
    main()
