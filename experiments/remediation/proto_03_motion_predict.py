"""Candidate ① motion prediction: constant-velocity search centering.

Before correlating, shift the search center by an EMA estimate of the
recent per-frame displacement, so the residual displacement the filter has
to find stays small even during the measured 15-21 px/frame bursts. The
velocity is re-estimated from the TOTAL movement each frame (prediction +
correlation correction).

Usage: python experiments/remediation/proto_03_motion_predict.py \
           --video <path> --pixel <i>,<j>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mosse import MosseTracker              # noqa: E402
import harness                                  # noqa: E402


class MotionPredictMosse(MosseTracker):
    """v1 MOSSE + constant-velocity pre-shift of the search center."""

    name = "mosse-mp"

    VEL_EMA = 0.5  # weight of the newest displacement in the velocity

    def init(self, frame, box):
        self._vx = 0.0
        self._vy = 0.0
        super().init(frame, box)

    def update(self, frame):
        cx0, cy0 = self._cx, self._cy
        self._cx += self._vx                     # predict...
        self._cy += self._vy
        self._clamp_center(frame.shape)
        box, psr = super().update(frame)         # ...correlate corrects
        a = self.VEL_EMA
        self._vx = a * (self._cx - cx0) + (1 - a) * self._vx
        self._vy = a * (self._cy - cy0) + (1 - a) * self._vy
        return box, psr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    harness.run("proto_03 motion-predict", MotionPredictMosse, args.video,
                (i, j), csv_path=args.csv)
