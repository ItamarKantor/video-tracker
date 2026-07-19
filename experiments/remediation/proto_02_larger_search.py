"""Candidate ② larger search window: correlate over a padded window.

Classic CF-tracker trick: train/track with a window pad x the target box
(the Hann window already de-emphasizes the border), so the correlation
peak can be found up to pad x further away per frame - attacking the
measured 15-21 px/frame translation bursts that push v1's +-64 px limit.

Usage: python experiments/remediation/proto_02_larger_search.py \
           --video <path> --pixel <i>,<j> [--pad 1.5]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mosse import MosseTracker              # noqa: E402
import harness                                  # noqa: E402


class PaddedSearchMosse(MosseTracker):
    """v1 MOSSE with the correlation window padded around the target box."""

    name = "mosse-pad"

    def __init__(self, pad: float = 1.5, **kw):
        super().__init__(**kw)
        self._pad = pad

    def init(self, frame, box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        fh, fw = frame.shape[:2]
        pw = min(int(round(w * self._pad)), fw)
        ph = min(int(round(h * self._pad)), fh)
        px = int(round(min(max(cx - pw / 2, 0), fw - pw)))
        py = int(round(min(max(cy - ph / 2, 0), fh - ph)))
        super().init(frame, (px, py, pw, ph))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    harness.run(f"proto_02 padded x{args.pad}",
                lambda: PaddedSearchMosse(pad=args.pad), args.video, (i, j),
                csv_path=args.csv)
