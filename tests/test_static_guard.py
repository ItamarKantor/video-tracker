"""StaticGuard: affine-referenced velocity semantics (synthetic frames)."""

import cv2
import numpy as np

from src.main import STATIC_GUARD_FRAMES, StaticGuard

W, H = 800, 600
BOX_SIZE = 120


def _base(seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (H, W), dtype=np.uint8)


def _bgr(gray):
    return np.dstack([gray, gray, gray])


def _translating_frames(n, shift=8, seed=3):
    base = _base(seed)
    for k in range(n):
        yield k, _bgr(np.roll(base, shift * k, axis=1))


def _box_at(x, y):
    return (int(x), int(y), BOX_SIZE, BOX_SIZE)


def test_fires_when_box_ignores_scene_motion():
    guard = StaticGuard()
    fired = []
    for k, frame in _translating_frames(STATIC_GUARD_FRAMES + 4):
        fired.append(guard.update(frame, _box_at(300, 200)))  # box frozen
    assert fired[-1] is True
    assert not any(fired[:STATIC_GUARD_FRAMES - 1])


def test_quiet_when_box_follows_scene_motion():
    guard = StaticGuard()
    shift = 8
    fired = [guard.update(frame, _box_at(100 + shift * k, 200))
             for k, frame in _translating_frames(STATIC_GUARD_FRAMES + 6,
                                                 shift=shift)]
    assert not any(fired)


def test_abstains_when_scene_is_still():
    guard = StaticGuard()
    frame = _bgr(_base())
    fired = [guard.update(frame.copy(), _box_at(300, 200))
             for _ in range(STATIC_GUARD_FRAMES + 5)]
    assert not any(fired)


def test_abstains_for_target_at_rotation_center():
    """A target at the rotation center genuinely moves little while the
    scene spins around it - the guard must NOT call that screen-glue."""
    guard = StaticGuard()
    base = _base()
    cx, cy = 400, 300  # rotation center = box center
    fired = []
    for k in range(STATIC_GUARD_FRAMES + 6):
        m = cv2.getRotationMatrix2D((cx, cy), 3.0 * k, 1.0)
        g = cv2.warpAffine(base, m, (W, H), borderMode=cv2.BORDER_REFLECT)
        fired.append(guard.update(
            _bgr(g), _box_at(cx - BOX_SIZE // 2, cy - BOX_SIZE // 2)))
    assert not any(fired)


def test_reset_clears_streak():
    guard = StaticGuard()
    frames = list(_translating_frames(STATIC_GUARD_FRAMES + 4))
    for k, frame in frames[:STATIC_GUARD_FRAMES - 1]:
        guard.update(frame, _box_at(300, 200))
    guard.reset()
    assert guard.streak == 0
    assert guard.update(frames[-1][1], _box_at(300, 200)) is False
