"""Remediation baseline: instrument grip, PSR decay, drift and re-acquisition.

Runs the EXACT product pipeline (no modifications) on a video + target
pixel and logs per frame: PSR, box center, state, detector verdict-related
counters, and every re-acquisition attempt/outcome. Prints a summary and
writes a CSV so every candidate fix can be measured against the same
baseline.

Usage:
    python experiments/grip_baseline.py --video <path> --pixel <i>,<j> \
        [--csv out.csv]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_video import VideoSource            # noqa: E402
from src.loss_detect import LossDetector        # noqa: E402
from src.main import Pipeline                   # noqa: E402
from src.reacquire import Reacquirer            # noqa: E402
from src.tracker import box_center, make_tracker, pixel_to_box  # noqa: E402


class CountingReacquirer(Reacquirer):
    """Reacquirer that counts attempts and hits (instrumentation only)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attempts = 0
        self.hits = 0

    def attempt(self, frame):
        self.attempts += 1
        box = super().attempt(frame)
        if box is not None:
            self.hits += 1
        return box


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True, help="i,j (row,col) on frame 0")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))

    rows = []
    with VideoSource(args.video) as src:
        first = src.read()
        tracker = make_tracker("mosse")
        tracker.init(first, pixel_to_box(j, i, 128, src.width, src.height))
        reacq = CountingReacquirer()
        pipe = Pipeline(tracker, LossDetector(), reacq)
        t_start = time.perf_counter()
        for idx, frame in enumerate(src.frames(), start=1):
            box, conf = pipe.process(idx, frame)
            cx, cy = box_center(box) if box is not None else (-1, -1)
            rows.append((idx, f"{conf:.2f}" if conf is not None else "",
                         cx, cy, pipe.state))
        wall = time.perf_counter() - t_start

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["frame", "psr", "cx", "cy", "state"])
            w.writerows(rows)

    n = len(rows)
    first_lost = next((r[0] for r in rows if r[4] == "LOST"), None)
    print(f"frames: {n} | wall {wall:.1f}s ({n / wall:.0f} fps incl. decode)")
    print(f"grip held (init -> first LOST): "
          f"{(first_lost - 1) if first_lost else n} frames")
    print("transitions:", " -> ".join(f"{s}@f{k}" for k, s in pipe.transitions))
    print(f"re-acquisition: {reacq.attempts} attempts, {reacq.hits} accepted")
    if first_lost:
        lo = max(0, first_lost - 26)
        print(f"PSR f{lo + 1}..f{first_lost} (last 25 before LOST + LOST frame):")
        seg = [f"{r[0]}:{r[1]}" for r in rows[lo:first_lost]]
        print("  " + "  ".join(seg))


if __name__ == "__main__":
    main()
