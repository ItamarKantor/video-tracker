"""Render an annotated demo video of the full pipeline (offline).

Runs exactly the same pipeline as src/main.py (tracker + loss detector +
re-acquisition + overlay) and writes the annotated frames to an .mp4 at the
source fps, so the demo shows precisely what the live app shows - box,
trajectory, state banner (TRACKING / LOST / RE-ACQUIRED) and the measured
processing fps.

Output goes to demo/ by default, which is kept out of git (large media);
share the file as a GitHub release asset or a link. --end trims the
rendering to the first N frames (a clean single cycle demos better than a
long tail) and --caption burns a small label into every frame (e.g. to mark
a synthetic test clip as synthetic).

Usage:
    python experiments/render_demo.py --video input/synthetic_loop_DEV.mp4 \
        --pixel 857,1167 --end 800 --caption "SYNTHETIC TEST CLIP ..."
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_video import VideoSource                 # noqa: E402
from src.loss_detect import LossDetector             # noqa: E402
from src.main import FpsMeter, LOST, REACQUIRED, Pipeline  # noqa: E402
from src.reacquire import Reacquirer                 # noqa: E402
from src.tracker import box_center, make_tracker, pixel_to_box  # noqa: E402
from src.ui import draw_overlay                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True, help="i,j (row,col) on frame 0")
    ap.add_argument("--out", default=None,
                    help="output path (default: demo/<video-stem>_demo.mp4)")
    ap.add_argument("--tracker", default="mosse")
    ap.add_argument("--end", type=int, default=None,
                    help="stop rendering after this many frames")
    ap.add_argument("--caption", default=None,
                    help="small label burned into the bottom of every frame")
    args = ap.parse_args()

    i, j = (int(v) for v in args.pixel.split(","))
    out = Path(args.out) if args.out else \
        Path("demo") / f"{Path(args.video).stem}_demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    with VideoSource(args.video) as src:
        first = src.read()
        fps = src.fps if src.fps and src.fps > 0 else 30.0
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (src.width, src.height))
        if not writer.isOpened():
            raise SystemExit(f"cannot open writer for {out}")

        box = pixel_to_box(j, i, 128, src.width, src.height)
        tracker = make_tracker(args.tracker)
        tracker.init(first, box)
        pipe = Pipeline(tracker, LossDetector(), Reacquirer())

        trail = [box_center(box)]
        fps_meter = FpsMeter()
        prev_state, n, idx = None, 0, 0
        while True:
            t0 = time.perf_counter()          # decode+process+overlay
            frame = src.read()
            if frame is None:
                break
            idx += 1
            box, conf = pipe.process(idx, frame)
            if pipe.state == REACQUIRED and prev_state == LOST:
                trail.clear()
            prev_state = pipe.state
            if box is not None:
                trail.append(box_center(box))
                del trail[:-200]
            draw_overlay(frame, box, trail, pipe.state, fps_meter.value,
                         tracker.name, conf)
            if args.caption:
                cv2.putText(frame, args.caption,
                            (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            fps_meter.update((time.perf_counter() - t0) * 1000)
            writer.write(frame)
            n += 1
            if args.end is not None and n >= args.end:
                break
        writer.release()

    timeline = " -> ".join(f"{s}@f{k}" for k, s in pipe.transitions)
    print(f"wrote {out} ({n} frames @ {fps:.1f} fps) | timeline: {timeline}")


if __name__ == "__main__":
    main()
