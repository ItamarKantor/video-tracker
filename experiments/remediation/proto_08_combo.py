"""Combination of the tracker levers: ① motion prediction + ③ scale search
(+ optional window padding ② and a configurable learning rate).

Each lever attacks a different part of the measured motion (translation
bursts / zoom / adaptation rate), so the combination is measured even
though the single levers did not move the needle individually.

Usage: python experiments/remediation/proto_08_combo.py \
           --video <path> --pixel <i>,<j> [--pad 1.0] [--eta 0.125]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness                                          # noqa: E402
from proto_01_scale_search import ScaleSearchMosse      # noqa: E402


class ComboMosse(ScaleSearchMosse):
    """Scale search + constant-velocity pre-shift (+ optional padded init)."""

    name = "mosse-combo"

    VEL_EMA = 0.5

    def __init__(self, pad: float = 1.0, **kw):
        super().__init__(**kw)
        self._pad = pad

    def init(self, frame, box):
        if self._pad != 1.0:
            x, y, w, h = box
            cx, cy = x + w / 2, y + h / 2
            fh, fw = frame.shape[:2]
            w = min(int(round(w * self._pad)), fw)
            h = min(int(round(h * self._pad)), fh)
            box = (int(round(min(max(cx - w / 2, 0), fw - w))),
                   int(round(min(max(cy - h / 2, 0), fh - h))), w, h)
        self._vx = 0.0
        self._vy = 0.0
        super().init(frame, box)

    def update(self, frame):
        cx0, cy0 = self._cx, self._cy
        self._cx += self._vx
        self._cy += self._vy
        self._clamp_center(frame.shape)
        box, psr = super().update(frame)
        a = self.VEL_EMA
        self._vx = a * (self._cx - cx0) + (1 - a) * self._vx
        self._vy = a * (self._cy - cy0) + (1 - a) * self._vy
        return box, psr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--pad", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.125)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    harness.run(f"proto_08 combo pad={args.pad} eta={args.eta}",
                lambda: ComboMosse(pad=args.pad, learn_rate=args.eta),
                args.video, (i, j), csv_path=args.csv)
