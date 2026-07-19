"""Re-acquisition.

While LOST, search the whole frame for the target's last known appearance:

  * ORB keypoints + descriptors of stored CONTEXT patches (the target plus
    2x surrounding context, inner target box recorded), matched against
    the current frame. ORB is rotation-invariant and its image pyramid
    tolerates moderate scale change - chosen precisely because an early
    feasibility probe showed plain NCC template matching dies under aerial
    footage's in-plane rotation. Context patches, not window-sized ones:
    on locally repetitive texture (near-identical pen walls) every
    window-scale descriptor has near-twins and the ratio test annihilates
    all matches (measured: median best/second ratio 0.97, 0/116
    survivors); the 2x context captures unique larger structure
    (measured: 27 vs 7 inliers).
  * Lowe ratio test prunes ambiguous matches, then a RANSAC homography must
    agree on >= min_inliers geometrically consistent matches - a single
    repeated-texture coincidence cannot fake that.
  * The INNER target-box corners are mapped through the homography back to
    full-frame coordinates; the resulting box is sanity-checked (size
    within loose bounds of the stored target box, center inside the frame)
    to reject degenerate homographies. The mapped size also restores the
    target's CURRENT scale, so re-seeding after a zoom re-fits the box for
    free.

Design points measured on the aerial sample (low-contrast desert):
  * Full-resolution search, NOT downscaled: at 0.5x the small weak-contrast
    features simply vanish (0-2 ratio matches vs. 10 RANSAC inliers at
    1.0x). Cost is controlled by attempting only every N-th frame while
    LOST instead (the tracker is idle then, so the budget is available).
  * min_inliers=8: the 128px patch yields only ~60 ORB keypoints on this
    texture, so 8-10 geometrically consistent matches is a strong signal.
    IMPORTANT limitation of what this gate proves: a re-seed that lands on
    BACKGROUND self-corrects (PSR collapses within a few frames and the
    machine returns to LOST) - but a re-seed that lands on a similar REAL
    structure does not. RANSAC proves the match is geometrically
    self-consistent and PSR proves the lock is sharp; a wrong-yet-real
    instance satisfies both perfectly, so on scenes with repeated
    near-identical structures re-acquisition can lock onto the wrong
    instance and then track it stably (measured on the sample clip: 4 of 8
    tested init pixels, offsets up to ~925 px, one accept at exactly
    min_inliers). Neither signal can express "wrong instance"; a
    motion-plausibility gate on re-seeds is the proposed future fix.
  * The patch detector runs with a lowered FAST threshold to harvest more
    corners from weak wall/road edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import Box

# accept a re-seeded box only within these factors of the original patch
SIZE_RANGE = (0.3, 3.0)     # linear scale vs. the stored patch
ASPECT_RANGE = (0.4, 2.5)   # (w/h) ratio vs. the stored patch's ratio


class Reacquirer:
    """ORB + RANSAC search for the last-good appearance of the target."""

    def __init__(self, scale: float = 1.0, nfeatures: int = 6000,
                 ratio: float = 0.75, min_inliers: int = 8,
                 ransac_px: float = 4.0):
        self._scale = scale
        self._ratio = ratio
        self._min_inliers = min_inliers
        self._ransac_px = ransac_px
        self._orb_patch = cv2.ORB_create(nfeatures=800, fastThreshold=10)
        self._orb_frame = cv2.ORB_create(nfeatures=nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._entries = []  # (keypoints, descriptors, inner_box) per context

    def set_target(self, contexts) -> None:
        """Store the appearances to search for - STICKY: only the first
        successful call takes effect.

        The first LOST happens while the appearance memory still holds the
        verified true target; anything captured after a re-seed may be
        contaminated by an imperfect re-acquisition (measured: a re-seed
        that drifted onto a burned-in HUD overlay poisoned the stored
        context, and every later "re-acquisition" matched the overlay).
        Matching only the pristine first-loss appearance trades a little
        adaptivity for correctness.

        `contexts` is a list of (context_patch_bgr, inner_box) pairs (see
        BaseTracker.reacq_contexts), or a single BGR patch, in which case
        the whole patch is the target (inner box = full patch).
        """
        if self._entries:
            return
        self._entries = []
        if contexts is None:
            contexts = []
        if isinstance(contexts, np.ndarray):
            h, w = contexts.shape[:2]
            contexts = [(contexts, (0, 0, w, h))]
        for patch, inner in contexts:
            if patch is None or patch.size == 0:
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            kp, des = self._orb_patch.detectAndCompute(gray, None)
            # fewer keypoints than the inlier bar can never verify a match
            if des is not None and len(kp) >= self._min_inliers:
                self._entries.append((kp, des, inner))
        if not self._entries:
            print("warning: re-acquisition INACTIVE - the stored target "
                  "appearance has too few ORB keypoints (need >= "
                  f"{self._min_inliers} to ever verify a match). A "
                  "featureless target cannot be re-found; re-select a "
                  "textured point to enable re-acquisition.")

    @property
    def active(self) -> bool:
        """False when every stored patch is too featureless to ever match."""
        return bool(self._entries)

    def attempt(self, frame_bgr: np.ndarray) -> Box | None:
        """Search one frame. Returns a verified full-resolution box, or None."""
        if not self.active:
            return None
        full_shape = frame_bgr.shape
        if self._scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, None, fx=self._scale,
                                   fy=self._scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = self._orb_frame.detectAndCompute(gray, None)  # ONCE
        if des_f is None or len(kp_f) < self._min_inliers:
            return None
        best = None
        for kp_p, des_p, inner in self._entries:
            res = self._match_entry(kp_p, des_p, inner, kp_f, des_f,
                                    full_shape)
            if res is not None and (best is None or res[0] > best[0]):
                best = res
        return best[1] if best else None

    def _match_entry(self, kp_p, des_p, inner, kp_f, des_f, frame_shape):
        """Ratio test + RANSAC + sanity for one stored context entry.

        Returns (inlier_count, box) or None.
        """
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
        box = self._verified_box(hom, inner, frame_shape)
        return (int(mask.sum()), box) if box is not None else None

    def _verified_box(self, hom: np.ndarray, inner: Box,
                      frame_shape) -> Box | None:
        """Map the INNER target-box corners through the homography; reject
        degenerate results."""
        ix, iy, iw, ih = inner
        corners = np.float32([[ix, iy], [ix + iw, iy], [ix + iw, iy + ih],
                              [ix, iy + ih]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, hom).reshape(-1, 2) / self._scale
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        w, h = x1 - x0, y1 - y0
        fh, fw = frame_shape[:2]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (0 <= cx < fw and 0 <= cy < fh):
            return None
        if not (SIZE_RANGE[0] * iw <= w <= SIZE_RANGE[1] * iw and
                SIZE_RANGE[0] * ih <= h <= SIZE_RANGE[1] * ih):
            return None
        aspect = (w / h) / (iw / ih)
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            return None
        x0 = int(max(0, min(x0, fw - w)))
        y0 = int(max(0, min(y0, fh - h)))
        return (x0, y0, int(min(w, fw)), int(min(h, fh)))
