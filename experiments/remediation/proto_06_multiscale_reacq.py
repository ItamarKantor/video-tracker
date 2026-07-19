"""Candidate ⑤ gallery x multi-scale: gallery entries stored at 3 scales.

Same as proto_05, but every snapshot also enters the gallery resized to
x0.67 and x1.5, so scale robustness comes from the STORED side while the
frame is still detected only once per attempt.

Usage: python experiments/remediation/proto_06_multiscale_reacq.py \
           --video <path> --pixel <i>,<j> [--observe-every 45]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mosse import MosseTracker                      # noqa: E402
import harness                                          # noqa: E402
from proto_05_gallery_reacq import GalleryReacquirer    # noqa: E402

VARIANT_SCALES = (0.67, 1.0, 1.5)


class MultiScaleGalleryReacquirer(GalleryReacquirer):
    """Gallery entries additionally stored at rescaled variants."""

    def observe(self, patch) -> None:
        if patch is None:
            return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        for s in VARIANT_SCALES:
            g = gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s,
                                                 interpolation=cv2.INTER_AREA)
            self._add_entry(g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--observe-every", type=int, default=45)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    harness.run("proto_06 multiscale-gallery", MosseTracker, args.video,
                (i, j), reacquirer_factory=MultiScaleGalleryReacquirer,
                observe_every=args.observe_every, csv_path=args.csv)
