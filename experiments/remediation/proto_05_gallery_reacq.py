"""Candidate ④ appearance gallery for re-acquisition.

Keeps up to `capacity` templates snapshotted during tracking (via the
harness observe hook, every N tracked frames), so the stored appearances
span the scales/angles the target actually went through. While LOST, the
frame is ORB-detected ONCE and matched against every gallery entry;
the best RANSAC-verified hypothesis wins.

Usage: python experiments/remediation/proto_05_gallery_reacq.py \
           --video <path> --pixel <i>,<j> [--observe-every 45]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mosse import MosseTracker              # noqa: E402
import harness                                  # noqa: E402


class GalleryReacquirer(harness.CountingReacquirer):
    """v1 matching per entry, over a gallery of appearances."""

    def __init__(self, capacity: int = 8, **kw):
        super().__init__(**kw)
        self._capacity = capacity
        self._gallery = []  # (kp, des, (w, h)) per stored appearance

    # -- gallery maintenance (called by the harness while TRACKING) --------
    def observe(self, patch) -> None:
        if patch is None:
            return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self._add_entry(gray)

    def _add_entry(self, gray) -> None:
        kp, des = self._orb_patch.detectAndCompute(gray, None)
        if des is not None and len(kp) >= self._min_inliers:
            self._gallery.append((kp, des, (gray.shape[1], gray.shape[0])))
            if len(self._gallery) > self._capacity:
                self._gallery.pop(0)

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    # -- matching -----------------------------------------------------------
    def attempt(self, frame):
        self.attempts += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = self._orb_frame.detectAndCompute(gray, None)  # ONCE
        if des_f is None or len(kp_f) < self._min_inliers:
            return None
        entries = list(self._gallery)
        if self._des is not None:  # v1's primary last-good patch, if any
            entries.append((self._kp, self._des, self._patch_wh))
        best = None
        for kp_p, des_p, wh in entries:
            res = self._match_entry(kp_p, des_p, wh, kp_f, des_f, frame.shape)
            if res and (best is None or res[0] > best[0]):
                best = res
        if best is None:
            return None
        self.hits += 1
        if self.first_hit_frame is None:
            self.first_hit_frame = self.frame_idx
        return best[1]

    def _match_entry(self, kp_p, des_p, wh, kp_f, des_f, frame_shape):
        """v1's ratio + RANSAC + sanity pipeline for one gallery entry."""
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
        saved = self._patch_wh
        self._patch_wh = wh
        try:
            box = self._verified_box(hom, frame_shape)
        finally:
            self._patch_wh = saved
        return (int(mask.sum()), box) if box is not None else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--observe-every", type=int, default=45)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    m = harness.run("proto_05 gallery-reacq", MosseTracker, args.video, (i, j),
                    reacquirer_factory=GalleryReacquirer,
                    observe_every=args.observe_every, csv_path=args.csv)
