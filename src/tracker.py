"""Tracking core.

BaseTracker is the common interface every tracking backend implements, so
the concrete algorithm stays hot-swappable via the --tracker CLI flag.
Benchmarking showed the winner is content-dependent (MOSSE-128 won on the
aerial sample; CSRT is the documented fallback), so swappability is a core
requirement rather than a nicety.

update() returns (box, confidence). Confidence semantics are backend-
specific (higher = more confident): our MOSSE returns the PSR of its
response map (healthy ~20+, loss ~<7); the OpenCV built-ins can only map
their boolean success flag to 1.0 / 0.0 - and our measurements showed that
flag to be unreliable (CSRT reported success while ~600 px off-target; KCF
and MOSSE falsely "re-locked" after drifting). The loss detector
thresholds the PSR instead of trusting any backend flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2

Box = tuple[int, int, int, int]  # (x, y, w, h) in pixels


def pixel_to_box(x: int, y: int, size: int, frame_w: int, frame_h: int) -> Box:
    """Square box of side `size` centered on pixel (x, y), clamped into the frame.

    The task specifies a single target pixel; trackers need a region. `size`
    trades context/texture (robustness) against locality - 128 px beat 64 px
    for every tracker in our benchmark.
    """
    size = min(size, frame_w, frame_h)
    bx = min(max(x - size // 2, 0), frame_w - size)
    by = min(max(y - size // 2, 0), frame_h - size)
    return (bx, by, size, size)


def box_center(box: Box) -> tuple[int, int]:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


class BaseTracker(ABC):
    """Common interface for all tracking backends.

    provides_confidence: True only for backends whose update() confidence is
    a real graded signal (our MOSSE's PSR). The loss detector and
    re-acquisition require it; the Pipeline falls back to plain tracking for
    backends that only echo a boolean flag.
    """

    name: str = "base"
    provides_confidence: bool = False

    @property
    def reacq_contexts(self) -> list:
        """(context_patch, inner_box) entries for re-acquisition; only
        confidence-providing backends need to supply them."""
        return []

    @abstractmethod
    def init(self, frame, box: Box) -> None:
        """Start tracking the target enclosed by `box` on `frame`."""

    @abstractmethod
    def update(self, frame) -> tuple[Box | None, float]:
        """Track into `frame` -> (box, confidence).

        Confidence is backend-specific, higher = more confident (see module
        docstring). box is None when the backend reports no target at all.
        """


class OpenCVTracker(BaseTracker):
    """Adapter for the OpenCV tracker implementations."""

    def __init__(self, name: str, factory):
        self.name = name
        self._factory = factory
        self._tracker = None

    def init(self, frame, box: Box) -> None:
        self._tracker = self._factory()
        self._tracker.init(frame, box)

    def update(self, frame) -> tuple[Box | None, float]:
        ok, bb = self._tracker.update(frame)
        if not ok:
            return None, 0.0
        return tuple(int(round(v)) for v in bb), 1.0


_CV_FACTORIES = {
    # cv2's MOSSE, kept for A/B comparison against our own implementation.
    "mosse-cv2": lambda: cv2.legacy.TrackerMOSSE_create(),
    "kcf": cv2.TrackerKCF_create,
    # Documented fallback: robust in the literature, ~72 fps end-to-end in
    # our benchmark, but silently drifted on the aerial sample clip.
    "csrt": cv2.TrackerCSRT_create,
}

# "mosse" (the default) is OUR implementation (src/mosse.py): same algorithm
# family as mosse-cv2, but exposes the response map (PSR confidence) and a
# controllable template-update policy, which loss detection requires.
TRACKER_CHOICES = sorted(_CV_FACTORIES) + ["mosse"]


def make_tracker(name: str) -> BaseTracker:
    """Build a tracking backend by CLI name."""
    if name == "mosse":
        from .mosse import MosseTracker  # local import avoids a cycle
        return MosseTracker()
    if name not in _CV_FACTORIES:
        raise ValueError(f"unknown tracker '{name}' (choices: {TRACKER_CHOICES})")
    return OpenCVTracker(name, _CV_FACTORIES[name])
