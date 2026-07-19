"""Candidate ③ scale adaptation: per-frame scale search {0.95, 1.0, 1.05}.

The template stays a fixed 128 px filter; each frame we crop the search
window at three candidate object scales, resize each crop to the template
size, correlate all three and keep the best-PSR one. The winning factor
multiplies a persistent object-scale estimate, so the reported box (and the
stored appearance) grows/shrinks with the target — attacking the measured
+0.31 %/frame zoom that the fixed-scale v1 cannot absorb.

Usage: python experiments/remediation/proto_01_scale_search.py \
           --video <path> --pixel <i>,<j> [--csv out.csv]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mosse import EPS, MosseTracker, _preprocess   # noqa: E402
import harness                                         # noqa: E402

SCALES = (0.95, 1.0, 1.05)


class ScaleSearchMosse(MosseTracker):
    """v1 MOSSE + best-PSR scale search each frame."""

    name = "mosse-scale"

    def init(self, frame, box):
        self._obj_scale = 1.0
        super().init(frame, box)

    # window size in image px is template size x current object scale
    def _win(self, extra=1.0):
        w0, h0 = self._size
        s = getattr(self, "_obj_scale", 1.0) * extra
        return max(16, int(round(w0 * s))), max(16, int(round(h0 * s)))

    def _clamp_center(self, frame_shape):
        fh, fw = frame_shape[:2]
        cw, ch = self._win()
        self._cx = float(np.clip(self._cx, cw / 2, max(cw / 2, fw - cw / 2)))
        self._cy = float(np.clip(self._cy, ch / 2, max(ch / 2, fh - ch / 2)))

    def _crop_gray(self, frame, extra=1.0):
        """Crop the scaled window and resize to template size."""
        w0, h0 = self._size
        cw, ch = self._win(extra)
        fh, fw = frame.shape[:2]
        x0 = int(round(min(max(self._cx - cw / 2, 0), fw - cw)))
        y0 = int(round(min(max(self._cy - ch / 2, 0), fh - ch)))
        patch = cv2.cvtColor(frame[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2GRAY)
        return cv2.resize(patch, (w0, h0), interpolation=cv2.INTER_AREA)

    def update(self, frame):
        w0, h0 = self._size
        best = None
        for s in SCALES:
            f = np.fft.rfft2(_preprocess(self._crop_gray(frame, s), self._window))
            r = np.fft.irfft2(f * self._A / (self._B + EPS), s=(h0, w0))
            py, px = np.unravel_index(np.argmax(r), r.shape)
            psr = self._psr(r, px, py)
            if best is None or psr > best[0]:
                best = (psr, s, px, py)
        psr, s, px, py = best

        self._obj_scale = float(np.clip(self._obj_scale * s, 0.25, 6.0))
        eff = self._obj_scale  # image px per template px at the winning scale
        self._cx += (px - w0 // 2) * eff
        self._cy += (py - h0 // 2) * eff
        self._clamp_center(frame.shape)

        if not self.frozen:
            f2 = np.fft.rfft2(_preprocess(self._crop_gray(frame), self._window))
            self._A = self._eta * self._G * np.conj(f2) + (1 - self._eta) * self._A
            self._B = self._eta * (f2 * np.conj(f2)).real + (1 - self._eta) * self._B
            self._remember_appearance(frame, float(psr))
        return self._box(), float(psr)

    def _box(self):
        cw, ch = self._win()
        return (int(round(self._cx - cw / 2)), int(round(self._cy - ch / 2)),
                cw, ch)

    def _remember_appearance(self, frame, quality: float) -> None:
        """Store the scaled window resized to template size (consistent
        appearance for re-acquisition regardless of current scale)."""
        w0, h0 = self._size
        cw, ch = self._win()
        fh, fw = frame.shape[:2]
        x0 = int(round(min(max(self._cx - cw / 2, 0), fw - cw)))
        y0 = int(round(min(max(self._cy - ch / 2, 0), fh - ch)))
        patch = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w0, h0),
                           interpolation=cv2.INTER_AREA)
        self._patches.append((quality, patch))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    harness.run("proto_01 scale-search", ScaleSearchMosse, args.video,
                (i, j), csv_path=args.csv)
