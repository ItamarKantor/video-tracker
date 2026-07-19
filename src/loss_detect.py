"""Loss detection.

Decides when the tracker's confidence (PSR) means the target is gone.

A FIXED PSR threshold fails twice over: healthy PSR levels vary per target
and scene (our sample clip idles near ~49; a weak-texture target lives far
lower), and on the measured loss event a fixed "PSR < 8" fired ~34 frames
late. So the detector is RELATIVE: it keeps a running median of recent
healthy PSR values as the reference level, and a frame is LOW when

    psr < max(rel_drop * reference, abs_floor)

(abs_floor only catches the degenerate case of a reference that is itself
tiny), with one absolute override: a frame whose PSR is inside Bolme's
strong-lock band (>= abs_healthy, ~20) is NEVER low, whatever the relative
drop says. This guards against an inflated reference: right after (re-)init
the template correlates with nearly its own source patch, PSR idles at
100+, and when real motion starts the settle-down to a perfectly healthy
~30-40 would otherwise read as a collapse (measured on the sample clip:
false LOST at f49 with reference 117, PSR 25-37). Hysteresis works in both
directions:

  * n_enter consecutive LOW frames confirm LOST - a 1-2 frame dip (motion
    blur, HUD flicker) must not flip the state;
  * a single LOW frame already makes the verdict SUSPECT, and the caller
    freezes template learning on SUSPECT: freezing is cheap and reversible,
    template poisoning is neither (measured: PSR 2900 on a dead patch);
  * after LOW frames, n_exit consecutive healthy frames are needed before
    the LOW streak is forgotten (one lucky frame mid-collapse must not
    reset the evidence).

Only frames observed while fully healthy feed the reference median, so the
reference does not chase the collapse downward.
"""

from __future__ import annotations

import statistics
from collections import deque
from enum import Enum


class Verdict(Enum):
    HEALTHY = "healthy"    # confidence at its normal level
    SUSPECT = "suspect"    # in the low zone, not yet confirmed -> freeze
    LOST = "lost"          # confirmed loss -> switch to re-acquisition


class LossDetector:
    """Relative-PSR loss detector with two-sided hysteresis."""

    def __init__(self, window: int = 45, rel_drop: float = 0.35,
                 n_enter: int = 3, n_exit: int = 5, abs_floor: float = 4.0,
                 abs_healthy: float = 20.0):
        self._history: deque[float] = deque(maxlen=window)
        self._rel_drop = rel_drop
        self._n_enter = n_enter
        self._n_exit = n_exit
        self._abs_floor = abs_floor
        self._abs_healthy = abs_healthy
        self._low_run = 0
        self._recover_run = 0

    @property
    def reference(self) -> float:
        """Current healthy-PSR reference (running median)."""
        return statistics.median(self._history) if self._history else 0.0

    def update(self, psr: float) -> Verdict:
        """Feed one frame's PSR, get the current verdict."""
        ref = self.reference if self._history else psr
        low = (psr < self._abs_healthy and
               psr < max(self._rel_drop * ref, self._abs_floor))

        if low:
            self._low_run += 1
            self._recover_run = 0
        elif self._low_run > 0:
            self._recover_run += 1
            if self._recover_run >= self._n_exit:
                self._low_run = 0
                self._recover_run = 0

        if self._low_run == 0:
            self._history.append(psr)
            return Verdict.HEALTHY
        if self._low_run >= self._n_enter:
            return Verdict.LOST
        return Verdict.SUSPECT

    def reset(self) -> None:
        """Forget everything (call after re-acquisition re-seeds the tracker:
        a fresh template starts a new PSR regime)."""
        self._history.clear()
        self._low_run = 0
        self._recover_run = 0
