"""Reacquirer: featureless rejection, planted-target recovery, no-match."""

import cv2
import numpy as np

from src.reacquire import Reacquirer

PATCH_BOX = (240, 200, 128, 128)  # where the patch is cut from / planted


def _noise_scene(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return np.dstack([gray, gray, gray])


def test_featureless_patch_deactivates(capsys):
    re = Reacquirer()
    re.set_target(np.full((128, 128, 3), 128, dtype=np.uint8))
    assert not re.active
    assert "re-acquisition INACTIVE" in capsys.readouterr().out
    assert re.attempt(_noise_scene(seed=1)) is None


def test_finds_planted_target():
    cv2.setRNGSeed(0)  # findHomography RANSAC determinism
    scene = _noise_scene(seed=42)
    x, y, w, h = PATCH_BOX
    re = Reacquirer()
    re.set_target(scene[y:y + h, x:x + w].copy())
    assert re.active

    box = re.attempt(scene)
    assert box is not None
    bx, by, bw, bh = box
    assert abs((bx + bw / 2) - (x + w / 2)) <= 10
    assert abs((by + bh / 2) - (y + h / 2)) <= 10
    assert 0.6 * w <= bw <= 1.5 * w


def test_no_match_on_unrelated_scene():
    cv2.setRNGSeed(0)
    scene = _noise_scene(seed=42)
    x, y, w, h = PATCH_BOX
    re = Reacquirer()
    re.set_target(scene[y:y + h, x:x + w].copy())
    assert re.attempt(_noise_scene(seed=7)) is None


def test_sticky_target_ignores_later_set_target_calls():
    """Only the first (pristine) appearance is matched; later re-seeds must
    not contaminate the memory."""
    cv2.setRNGSeed(0)
    scene_a = _noise_scene(seed=42)
    scene_b = _noise_scene(seed=9)
    x, y, w, h = PATCH_BOX
    re = Reacquirer()
    re.set_target(scene_a[y:y + h, x:x + w].copy())
    re.set_target(scene_b[y:y + h, x:x + w].copy())  # must be ignored
    assert re.attempt(scene_a) is not None   # still matches the first target
    assert re.attempt(scene_b) is None       # never learned the second


def test_context_entry_reseeds_inner_target_box():
    cv2.setRNGSeed(0)
    scene = _noise_scene(seed=42)
    # target box inside the scene, context = 2x crop around it
    tx, ty, tw, th = 254, 214, 128, 128
    cx0, cy0 = tx - 64, ty - 64
    context = scene[cy0:cy0 + 256, cx0:cx0 + 256].copy()
    inner = (tx - cx0, ty - cy0, tw, th)

    re = Reacquirer()
    re.set_target([(context, inner)])
    assert re.active
    box = re.attempt(scene)
    assert box is not None
    bx, by, bw, bh = box
    # the INNER target box comes back, not the outer context bounds
    assert abs((bx + bw / 2) - (tx + tw / 2)) <= 10
    assert abs((by + bh / 2) - (ty + th / 2)) <= 10
    assert 0.6 * tw <= bw <= 1.5 * tw
