"""Pins deliberately-chosen defaults against accidental drift.

These values were set by measurement (see experiments/bench_results.md and
experiments/remediation/results.md); changing them should be a conscious,
re-measured decision, so a silent edit fails loudly here.
"""

from src.main import REACQ_EVERY_N, STATIC_GUARD_FRAMES


def test_reacq_cadence_default():
    # N=8: amortized LOST-state cost 10.7 ms single-threaded (measured;
    # >=30 fps on a 3x-slower CPU) vs ~0.27 s worst-case recovery latency.
    assert REACQ_EVERY_N == 8


def test_static_guard_confirmation_default():
    # 8 consecutive not-following frames: brief hesitations survive,
    # sustained screen-glue is caught within ~0.13-0.27 s.
    assert STATIC_GUARD_FRAMES == 8
