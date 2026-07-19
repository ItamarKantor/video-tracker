"""Candidate (C): cv2 CSRT head-to-head on the remediation target.

CSRT has built-in scale estimation and a wider effective search than
MOSSE. It reports no usable confidence (v1 pipeline runs it in plain-
tracking fallback), so 'frames held' cannot come from a LOST event -
instead we log the box per frame and save checkpoint crops for visual
drift judgment against the known target path.

Usage: python experiments/remediation/proto_04_csrt.py \
           --video <path> --pixel <i>,<j> --csv out.csv [--snaps dir]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.io_video import VideoSource                    # noqa: E402
from src.tracker import box_center, make_tracker, pixel_to_box  # noqa: E402

CHECKPOINTS = (300, 400, 440, 470, 500, 530, 560, 600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--snaps", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))

    rows, update_ms = [], []
    with VideoSource(args.video) as src:
        first = src.read()
        tracker = make_tracker("csrt")
        tracker.init(first, pixel_to_box(j, i, 128, src.width, src.height))
        for idx, frame in enumerate(src.frames(), start=1):
            t0 = time.perf_counter()
            box, conf = tracker.update(frame)
            update_ms.append((time.perf_counter() - t0) * 1000)
            cx, cy = box_center(box) if box else (-1, -1)
            w = box[2] if box else -1
            rows.append((idx, cx, cy, w, conf))
            if args.snaps and idx in CHECKPOINTS and box:
                snap = frame.copy()
                x, y, bw, bh = box
                cv2.rectangle(snap, (x, y), (x + bw, y + bh), (0, 0, 255), 4)
                cv2.putText(snap, f"csrt f{idx} w={bw}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                cv2.imwrite(f"{args.snaps}/csrt_f{idx}.png", snap)

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["frame", "cx", "cy", "w", "ok"])
            w.writerows(rows)
    mean = sum(update_ms) / len(update_ms)
    print(f"[proto_04 csrt] {len(rows)} frames | update {mean:.2f} ms "
          f"({1000 / mean:.0f} fps) | reported-fail frames: "
          f"{sum(1 for r in rows if not r[4])}")
    print("centers at checkpoints:",
          {r[0]: (r[1], r[2], f"w={r[3]}") for r in rows if r[0] in CHECKPOINTS})


if __name__ == "__main__":
    main()
