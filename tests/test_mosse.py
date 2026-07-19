"""Our MOSSE tracker on synthetic frames (fixed seeds, no video I/O)."""

import numpy as np

from src.mosse import MosseTracker

BOX = (288, 208, 64, 64)


def _noise_frame(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    """Deterministic textured BGR frame (noise = plenty of structure)."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return np.dstack([gray, gray, gray])


def test_tracks_known_translation():
    frame1 = _noise_frame(seed=42)
    dx, dy = 5, 3
    frame2 = np.roll(frame1, (dy, dx), axis=(0, 1))  # scene shifts right+down

    tracker = MosseTracker(seed=0)
    tracker.init(frame1, BOX)
    box, psr = tracker.update(frame2)

    assert abs(box[0] - (BOX[0] + dx)) <= 1
    assert abs(box[1] - (BOX[1] + dy)) <= 1
    assert psr > 20  # clean match: strong-lock band


def test_psr_collapses_on_unrelated_content():
    tracker = MosseTracker(seed=0)
    tracker.init(_noise_frame(seed=42), BOX)
    _, psr = tracker.update(_noise_frame(seed=7))  # statistically unrelated
    assert psr < 10


def test_frozen_stops_template_and_appearance_updates():
    frame1 = _noise_frame(seed=42)
    tracker = MosseTracker(seed=0)
    tracker.init(frame1, BOX)
    patch_before = tracker.last_good_patch.copy()

    tracker.frozen = True
    tracker.update(_noise_frame(seed=7))
    assert np.array_equal(tracker.last_good_patch, patch_before)


def test_quality_gate_prefers_high_psr_patch():
    frame1 = _noise_frame(seed=42)
    tracker = MosseTracker(seed=0)
    tracker.init(frame1, BOX)
    # a perfect repeat scores far higher PSR than an unrelated frame
    tracker.update(frame1)
    good_patch = tracker.last_good_patch.copy()
    tracker.update(_noise_frame(seed=7))  # low-PSR garbage gets buffered too
    assert np.array_equal(tracker.last_good_patch, good_patch)


def test_context_patch_geometry():
    frame = _noise_frame(seed=42)
    tracker = MosseTracker(seed=0)
    tracker.init(frame, BOX)
    contexts = tracker.reacq_contexts
    assert contexts, "context entries must exist right after init"
    ctx, (ix, iy, iw, ih) = contexts[0]
    x, y, w, h = BOX
    assert ctx.shape[0] == int(h * MosseTracker.CONTEXT_FACTOR)
    assert ctx.shape[1] == int(w * MosseTracker.CONTEXT_FACTOR)
    assert (iw, ih) == (w, h)
    # the inner region of the context crop is exactly the frame's box content
    assert np.array_equal(ctx[iy:iy + ih, ix:ix + iw],
                          frame[y:y + h, x:x + w])


def test_init_context_survives_buffer_eviction():
    frame = _noise_frame(seed=42)
    tracker = MosseTracker(seed=0)
    tracker.init(frame, BOX)
    init_ctx = tracker.reacq_contexts[-1][0].copy()
    for k in range(MosseTracker.APPEARANCE_DELAY + 5):  # evict the window
        tracker.update(_noise_frame(seed=100 + k))
    contexts = tracker.reacq_contexts
    assert len(contexts) == 2  # best recent + permanent init view
    assert np.array_equal(contexts[-1][0], init_ctx)
