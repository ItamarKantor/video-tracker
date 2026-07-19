"""Candidate ④-evolved: CONTEXT-patch re-acquisition.

Root-cause finding (measured): the v1 re-acquisition stored the 128 px
tracker window, whose content is locally repetitive on this footage - every
ORB descriptor has near-twins, so Lowe's ratio test annihilates ALL matches
(median best/second distance ratio 0.97, 0/116 survivors). Storing a patch
of the SAME target with 2x the spatial context captures the compound's
unique larger structure: 27 inliers (vs 7) at f300, and the f456-f540 LOST
window re-verifies at 9-13 inliers with zero false hits after the true
exit.

This prototype stores context patches (2x the box side, target box centered
inside) - one at init plus quality-gated refreshes while tracking - and
re-seeds the INNER target box mapped through the verified homography.

Usage: python experiments/remediation/proto_09_context_reacq.py \
           --video <path> --pixel <i>,<j> [--refresh 45]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.io_video import VideoSource            # noqa: E402
from src.loss_detect import LossDetector        # noqa: E402
from src.main import Pipeline                   # noqa: E402
from src.mosse import MosseTracker              # noqa: E402
from src.reacquire import Reacquirer            # noqa: E402
from src.tracker import box_center, pixel_to_box  # noqa: E402

CONTEXT_FACTOR = 2.0   # context side = factor x target box side
MAX_ENTRIES = 3        # newest context snapshots kept


def crop_context(frame, box):
    """Context crop around the box; returns (bgr, inner_box_in_crop)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    cw, ch = int(round(w * CONTEXT_FACTOR)), int(round(h * CONTEXT_FACTOR))
    fh, fw = frame.shape[:2]
    cw, ch = min(cw, fw), min(ch, fh)
    x0 = int(round(min(max(cx - cw / 2, 0), fw - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), fh - ch)))
    return frame[y0:y0 + ch, x0:x0 + cw], (x - x0, y - y0, w, h)


class ContextReacquirer(Reacquirer):
    """Match stored CONTEXT patches; re-seed the inner target box."""

    def __init__(self, veto_static: bool = True, **kw):
        super().__init__(**kw)
        self._orb_patch = cv2.ORB_create(nfeatures=800, fastThreshold=10)
        self._entries = []  # (kp, des, inner_box, ctx_wh)
        self._veto_static = veto_static
        self._prev_small = None  # downscaled gray of the previous attempt
        self.attempts = 0
        self.hits = 0
        self.vetoed = 0
        self.first_hit_frame = None
        self.frame_idx = 0

    def add_context(self, frame, box) -> None:
        patch, inner = crop_context(frame, box)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb_patch.detectAndCompute(gray, None)
        if des is not None and len(kp) >= self._min_inliers:
            self._entries.append((kp, des, inner,
                                  (gray.shape[1], gray.shape[0])))
            if len(self._entries) > MAX_ENTRIES:
                self._entries.pop(0)

    def set_target(self, patch_bgr) -> None:  # keep v1 patch as extra entry
        super().set_target(patch_bgr)

    @property
    def active(self) -> bool:
        return bool(self._entries) or self._des is not None

    def attempt(self, frame):
        self.attempts += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        kp_f, des_f = self._orb_frame.detectAndCompute(gray, None)  # ONCE
        if des_f is None or len(kp_f) < self._min_inliers:
            self._prev_small = small
            return None
        best = None
        for kp_p, des_p, inner, wh in reversed(self._entries):
            res = self._match(kp_p, des_p, inner, kp_f, des_f, frame.shape)
            if res and (best is None or res[0] > best[0]):
                best = res
        accepted = None
        if best is not None:
            if self._veto_static and self._is_static(best[1], small):
                self.vetoed += 1
            else:
                accepted = best[1]
        self._prev_small = small
        if accepted is None:
            return None
        self.hits += 1
        if self.first_hit_frame is None:
            self.first_hit_frame = self.frame_idx
        return accepted

    def _is_static(self, box, small) -> bool:
        """Reject boxes glued to non-moving screen content (burned-in HUD):
        when the frame as a whole moves, a genuine ground target moves with
        it; a region with far less temporal change than the global scene is
        screen furniture, not scenery."""
        if self._prev_small is None or self._prev_small.shape != small.shape:
            return False
        diff = cv2.absdiff(small, self._prev_small)
        global_motion = float(diff.mean())
        if global_motion < 1.0:      # everything static: nothing to judge
            return False
        x, y, w, h = (int(v * 0.25) for v in box)
        region = diff[max(y, 0):y + max(h, 1), max(x, 0):x + max(w, 1)]
        if region.size == 0:
            return True
        return float(region.mean()) < 0.25 * global_motion

    def _match(self, kp_p, des_p, inner, kp_f, des_f, frame_shape):
        pairs = self._matcher.knnMatch(des_p, des_f, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self._ratio * n.distance]
        if len(good) < self._min_inliers:
            return None
        src = np.float32([kp_p[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        hom, mask = cv2.findHomography(src, dst, cv2.RANSAC, self._ransac_px)
        if hom is None or mask is None or int(mask.sum()) < self._min_inliers:
            return None
        ix, iy, iw, ih = inner
        corners = np.float32([[ix, iy], [ix + iw, iy], [ix + iw, iy + ih],
                              [ix, iy + ih]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, hom).reshape(-1, 2)
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        w, h = x1 - x0, y1 - y0
        fh, fw = frame_shape[:2]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (0 <= cx < fw and 0 <= cy < fh):
            return None
        if not (0.3 * iw <= w <= 3.0 * iw and 0.3 * ih <= h <= 3.0 * ih):
            return None
        x0 = int(max(0, min(x0, fw - w)))
        y0 = int(max(0, min(y0, fh - h)))
        return (int(mask.sum()), (x0, y0, int(min(w, fw)), int(min(h, fh))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--refresh", type=int, default=45)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))

    update_ms = []
    with VideoSource(args.video) as src:
        first = src.read()
        tracker = MosseTracker()
        box0 = pixel_to_box(j, i, 128, src.width, src.height)
        tracker.init(first, box0)
        reacq = ContextReacquirer()
        reacq.add_context(first, box0)
        pipe = Pipeline(tracker, LossDetector(), reacq)
        n = 0
        for idx, frame in enumerate(src.frames(), start=1):
            reacq.frame_idx = idx
            t0 = time.perf_counter()
            box, conf = pipe.process(idx, frame)
            update_ms.append((time.perf_counter() - t0) * 1000)
            n = idx
            if (pipe.state == "TRACKING" and box is not None
                    and idx % args.refresh == 0):
                reacq.add_context(frame, box)

    mean = statistics.fmean(update_ms)
    lost_ms = [m for m in update_ms if m > 10]
    print(f"[proto_09 context-reacq] {n} frames | transitions: "
          f"{' -> '.join(f'{s}@f{k}' for k, s in pipe.transitions)}")
    print(f"reacq attempts={reacq.attempts} hits={reacq.hits} "
          f"vetoed(static)={reacq.vetoed} first hit f{reacq.first_hit_frame}")
    print(f"process {mean:.2f} ms mean ({1000 / mean:.0f} fps); "
          f"heaviest frames mean {statistics.fmean(lost_ms):.1f} ms"
          if lost_ms else f"process {mean:.2f} ms mean")


if __name__ == "__main__":
    main()
