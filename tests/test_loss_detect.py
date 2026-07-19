"""LossDetector state transitions and hysteresis (pure logic, no cv2)."""

from src.loss_detect import LossDetector, Verdict

HEALTHY_PSR = 50.0
LOW_PSR = 5.0


def _warmed_detector(**kwargs) -> LossDetector:
    det = LossDetector(**kwargs)
    for _ in range(60):
        assert det.update(HEALTHY_PSR) is Verdict.HEALTHY
    return det


def test_healthy_to_suspect_to_lost():
    det = _warmed_detector()  # defaults: n_enter=3
    assert det.update(LOW_PSR) is Verdict.SUSPECT
    assert det.update(LOW_PSR) is Verdict.SUSPECT
    assert det.update(LOW_PSR) is Verdict.LOST


def test_short_dip_recovers_with_hysteresis():
    det = _warmed_detector()  # defaults: n_exit=5
    det.update(LOW_PSR)
    det.update(LOW_PSR)  # 2 low frames: SUSPECT, not yet LOST
    # one lucky frame must NOT reset the low streak...
    assert det.update(HEALTHY_PSR) is Verdict.SUSPECT
    # ...but n_exit consecutive healthy frames must fully recover
    for _ in range(3):
        det.update(HEALTHY_PSR)
    assert det.update(HEALTHY_PSR) is Verdict.HEALTHY
    # and the streak was forgotten: one new low frame is SUSPECT, not LOST
    assert det.update(LOW_PSR) is Verdict.SUSPECT


def test_strong_lock_band_never_low():
    det = LossDetector()
    for _ in range(60):
        det.update(100.0)  # reference median = 100
    # 25 < rel_drop * 100 = 35, but PSR >= abs_healthy (20) is never low
    assert det.update(25.0) is Verdict.HEALTHY


def test_reset_forgets_history():
    det = _warmed_detector()
    for _ in range(3):
        det.update(LOW_PSR)
    det.reset()
    assert det.reference == 0.0
    assert det.update(HEALTHY_PSR) is Verdict.HEALTHY
