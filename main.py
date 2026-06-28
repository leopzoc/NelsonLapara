"""
Main Orchestrator — Closed-Loop Adaptive Emotional Intervention.

Runs on RPi 5.  Hardware:
  • WS2812B NeoPixel strip (144 LED) on GPIO 18
  • 1 physical button: GPIO 17 (cycles modes: Therapeutic → Avion → Circadian)
  • USB microphone

Architecture:

    ┌───────────┐     ┌───────────┐     ┌──────────────────┐
    │  Button   │────▶│   Main    │────▶│  Active Mode     │
    │  GPIO 17  │     │  Orchest. │     │  ┌─ Therapeutic   │
    │ (cycle)   │     │           │     │  ├─ Avion         │
    └───────────┘     └─────┬─────┘     │  └─ Circadian    │
                            │           └────────┬─────────┘
    ┌───────────┐     ┌─────▼─────┐              │
    │    Mic    │────▶│ SER Engine│     ┌────────▼─────────┐
    │ (stream)  │     │ (Wav2Vec2)│     │  NeoPixel Strip  │
    └───────────┘     └───────────┘     │  (144 WS2812B)   │
                                        └──────────────────┘
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
from buttons import ButtonController, Mode
from mode_avion import AvionMode
from mode_circadian import CircadianMode

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


# ── Main System ────────────────────────────────────────────────────

class InterventionSystem:
    """
    Central orchestrator.

    Manages mode switching (button-driven) and delegates to:
      • Therapeutic mode — SER + hill-climbing on the LED strip
      • Avion mode       — Boeing 737 cabin lighting cycle
      • Circadian mode   — Daylight rhythm LED cycle
    """

    def __init__(self):
        log.info("Initialising subsystems …")

        # ── Hardware ────────────────────────────────────────────────
        self.led = LedStrip()
        self.streamer = AudioStreamer()

        # ── Modes ───────────────────────────────────────────────────
        self.ser = SEREngine()
        self.mitigation = ColorMitigation()
        self.avion = AvionMode(self.led)
        self.circadian = CircadianMode(self.led)

        # ── Button (single, cycles modes) ────────────────────────
        self.buttons = ButtonController(
            on_mode_change=self._on_mode_change,
            led_strip=self.led,
        )

        # ── State ───────────────────────────────────────────────────
        self._active_mode: Mode = Mode.THERAPEUTIC
        self._therapeutic_state = TherapeuticState.LISTENING
        self._cooldown_until = 0.0
        self._running = False

    # ── Mode switching ─────────────────────────────────────────────

    def _on_mode_change(self, new_mode: Mode):
        """Callback fired by the ButtonController on GPIO interrupt."""
        log.info("⚙ Switching to %s", new_mode.name)
        self._stop_current_mode()
        self._active_mode = new_mode
        self._start_mode(new_mode)

    def _stop_current_mode(self):
        """Cleanly stop whichever mode is currently running."""
        self.avion.stop()
        self.circadian.stop()
        self.mitigation.reset()
        self._therapeutic_state = TherapeuticState.LISTENING

    def _start_mode(self, mode: Mode):
        match mode:
            case Mode.THERAPEUTIC:
                log.info("▶ Therapeutic mode — listening for emotional arousal")
                self.led.set_color("#000000")   # LEDs off until triggered
            case Mode.AVION:
                log.info("▶ Avion mode — Boeing 737 cabin lighting")
                self.avion.start()
            case Mode.CIRCADIAN:
                log.info("▶ Circadian mode — daylight rhythm cycle")
                self.circadian.start()

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
                if self._active_mode == Mode.THERAPEUTIC:
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
        self.streamer.stop()
        self.led.close()
        self.buttons.close()

    def _handle_signal(self, sig, frame):
        log.info("Signal %s received", sig)
        self._running = False

    # ── THERAPEUTIC mode ticks ─────────────────────────────────────

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
        if self._active_mode != Mode.THERAPEUTIC:
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
        if self._active_mode != Mode.THERAPEUTIC:
            return

        action = self.mitigation.tick()

        match action["action"]:
            case "show_color":
                self.led.set_color(action["color"])
                log.info("LED strip → %s", action["color"])

            case "record":
                audio = self.streamer.record_fixed(action["duration"])
                db = rms_to_db(rms(audio))
                self.mitigation.feed_baseline_db(db)

            case "wait_then_record":
                time.sleep(action["wait"])
                if self._active_mode != Mode.THERAPEUTIC:
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
                self.led.set_color(result["color"])
                log.info("↩ Rolled back → %s", result["color"])
            case "converged":
                self.led.set_color(result["color"])
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
            self.led.off()
            self._therapeutic_state = TherapeuticState.LISTENING
        else:
            time.sleep(0.5)


# ── Entry point ─────────────────────────────────────────────────────

def main():
    system = InterventionSystem()
    system.run()


if __name__ == "__main__":
    main()
