"""FpsMeter: the displayed fps must stay physically plausible.

Regression test for a measured bug: LOST frames that skip the
re-acquisition attempt returned from process() in microseconds, and the
naive 1/process-time EMA ballooned to six figures (467,000 fps observed on
the live overlay).
"""

from src.main import DISPLAY_FPS_CAP, FpsMeter


def test_synthetic_lost_stretch_stays_bounded():
    meter = FpsMeter()
    # a LOST stretch where per-frame work collapses to ~1 microsecond
    for _ in range(500):
        meter.update(0.001)
    assert 0.0 < meter.value <= DISPLAY_FPS_CAP


def test_normal_work_reads_normally():
    meter = FpsMeter()
    for _ in range(100):
        meter.update(5.0)  # 5 ms/frame -> 200 fps
    assert 190.0 <= meter.value <= 210.0


def test_ema_smooths_single_outlier():
    meter = FpsMeter()
    for _ in range(100):
        meter.update(10.0)  # steady 100 fps
    meter.update(0.001)     # one skipped-work frame
    assert meter.value < 200.0  # cannot leap toward the cap in one frame


def test_alternating_cheap_and_expensive_reads_true_throughput():
    """The LOST-state pattern: 7 cheap skip-frames per 1 expensive attempt.
    The display must read ~1000/mean-time, not the inflated mean-of-rates."""
    meter = FpsMeter()
    for _ in range(100):
        for _ in range(7):
            meter.update(1.7)   # skip frames
        meter.update(40.0)      # attempt frame
    true_fps = 1000.0 / ((7 * 1.7 + 40.0) / 8)  # ~155 fps
    assert abs(meter.value - true_fps) / true_fps < 0.35
    assert meter.value < 300.0  # far from the inflated ~500+ figure
