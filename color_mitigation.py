"""
Color Mitigation Loop — Hill-Climbing on saturation with dB reward.

Manages the closed-loop: show color → measure dB → adjust saturation →
repeat until convergence or rollback limit.
"""

from __future__ import annotations

import colorsys
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import config as cfg

log = logging.getLogger(__name__)


# ── Color helpers ───────────────────────────────────────────────────

def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
    """#RRGGBB → (H, S, L) each in [0, 1]."""
    h_str = hex_color.lstrip("#")
    r, g, b = (int(h_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    # colorsys gives HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """(H, S, L) each in [0, 1] → #RRGGBB."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def desaturate(hex_color: str, step: float = cfg.DESAT_STEP) -> str:
    """Reduce saturation by *step*, clamped to MIN_SATURATION."""
    h, s, l = hex_to_hsl(hex_color)
    s_new = max(s - step, cfg.MIN_SATURATION)
    return hsl_to_hex(h, s_new, l)


# ── State Machine ──────────────────────────────────────────────────

class Phase(Enum):
    IDLE = auto()
    SHOW_BASE = auto()
    MEASURE_BASELINE = auto()
    APPLY_DESAT = auto()
    MEASURE_POST = auto()
    EVALUATE = auto()
    CONVERGED = auto()


@dataclass
class MitigationState:
    """Mutable state for one mitigation episode."""
    phase: Phase = Phase.IDLE
    base_color: str = cfg.BASE_COLORS_HEX[0]
    current_color: str = cfg.BASE_COLORS_HEX[0]
    previous_color: str = cfg.BASE_COLORS_HEX[0]
    baseline_db: float = 0.0
    current_db: float = 0.0
    best_db: float = 0.0
    rollback_count: int = 0
    step_count: int = 0
    history: List[dict] = field(default_factory=list)


class ColorMitigation:
    """
    Non-blocking state-machine driving the hill-climbing loop.

    Call `tick()` repeatedly from the main loop; it returns an *action*
    dict telling the caller what to do next (show color, record audio,
    wait, or nothing).
    """

    def __init__(self, palette: Optional[List[str]] = None):
        self.palette = palette or list(cfg.BASE_COLORS_HEX)
        self._state = MitigationState()

    # ── public API ──────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._state.phase not in (Phase.IDLE, Phase.CONVERGED)

    @property
    def current_color(self) -> str:
        return self._state.current_color

    def trigger(self, initial_db: float, base_color: Optional[str] = None):
        """Start a new mitigation episode."""
        base = base_color or self.palette[0]
        self._state = MitigationState(
            phase=Phase.SHOW_BASE,
            base_color=base,
            current_color=base,
            previous_color=base,
            baseline_db=initial_db,
            best_db=initial_db,
        )
        log.info(
            "Mitigation triggered — base=%s  baseline_db=%.1f",
            base, initial_db,
        )

    def tick(self) -> dict:
        """
        Advance the state machine by one step.

        Returns
        -------
        dict with keys:
            action : str — "show_color" | "record" | "wait" | "done" | "noop"
            color  : str | None
            duration : float | None   (seconds to record / wait)
        """
        s = self._state
        match s.phase:

            case Phase.IDLE | Phase.CONVERGED:
                return {"action": "noop"}

            case Phase.SHOW_BASE:
                s.phase = Phase.MEASURE_BASELINE
                return {"action": "show_color", "color": s.current_color}

            case Phase.MEASURE_BASELINE:
                s.phase = Phase.APPLY_DESAT
                return {
                    "action": "record",
                    "duration": cfg.POST_INTERVENTION_LISTEN_SEC,
                }

            case Phase.APPLY_DESAT:
                s.previous_color = s.current_color
                s.current_color = desaturate(s.current_color)
                s.step_count += 1
                s.phase = Phase.MEASURE_POST
                return {"action": "show_color", "color": s.current_color}

            case Phase.MEASURE_POST:
                s.phase = Phase.EVALUATE
                return {
                    "action": "wait_then_record",
                    "wait": cfg.INTERVENTION_SETTLE_SEC,
                    "duration": cfg.POST_INTERVENTION_LISTEN_SEC,
                }

            case Phase.EVALUATE:
                return {"action": "evaluate"}

        return {"action": "noop"}

    def feed_baseline_db(self, db: float):
        """Feed the baseline dB measurement after initial color shown."""
        self._state.baseline_db = db
        self._state.best_db = db
        log.info("Baseline dB recorded: %.1f", db)

    def evaluate(self, new_db: float) -> dict:
        """
        Evaluate the dB change and decide: continue desaturating or rollback.

        Returns
        -------
        dict:  {"action": "continue"} or {"action": "rollback", "color": ...}
               or {"action": "converged"}
        """
        s = self._state
        s.current_db = new_db
        delta = new_db - s.best_db

        entry = {
            "step": s.step_count,
            "color": s.current_color,
            "db": new_db,
            "delta": delta,
        }
        s.history.append(entry)
        log.info(
            "Step %d — color=%s  dB=%.1f  ΔdB=%.1f",
            s.step_count, s.current_color, new_db, delta,
        )

        # Check near-minimum saturation → converged
        _, sat, _ = hex_to_hsl(s.current_color)
        if sat <= cfg.MIN_SATURATION + 0.01:
            s.phase = Phase.CONVERGED
            log.info("Converged (minimum saturation reached)")
            return {"action": "converged", "color": s.current_color}

        if delta < 0:
            # Improved — keep going
            s.best_db = new_db
            s.rollback_count = 0
            s.phase = Phase.APPLY_DESAT
            return {"action": "continue", "color": s.current_color}
        else:
            # Worsened — rollback
            s.rollback_count += 1
            rolled_back_to = s.previous_color
            s.current_color = rolled_back_to
            log.info(
                "Rollback #%d → %s", s.rollback_count, rolled_back_to,
            )
            if s.rollback_count >= cfg.MAX_ROLLBACKS:
                s.phase = Phase.CONVERGED
                log.info("Converged (max rollbacks reached)")
                return {"action": "converged", "color": rolled_back_to}

            s.phase = Phase.APPLY_DESAT
            return {"action": "rollback", "color": rolled_back_to}

    def reset(self):
        self._state = MitigationState()
