"""Phase E: reproducible benchmark of the COMPLETE pipeline at 1080p.

Measures, per frame and at native resolution:
  * decode      -- VideoSource.read()
  * process     -- Pipeline.process() = tracker + loss detector (+ ORB
                   re-acquisition while LOST)
  * overlay     -- draw_overlay()
tagged by the state machine's state, so the worst-state cost (LOST, which
carries the full-res ORB search) is visible separately - that is the number
the >=30 fps claim must survive on a modest machine.

The full pipeline runs only with our MOSSE backend: it is the only backend
producing a real confidence signal (PSR), which the loss detector requires.
The other backends get tracker-update-only rows for context.

--threads pins cv2.setNumThreads() (numpy FFT work is unaffected, but ORB /
resize / cvtColor are OpenCV): --threads 1 is the conservative single-
thread proxy for a colder CPU than this machine.

Usage:
    python experiments/bench_pipeline.py --video input/synthetic_loop_DEV.mp4 \
        --pixel 857,1167 [--threads 1] [--others]
"""

import argparse
import json
import platform
import subprocess
import time

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_video import VideoSource            # noqa: E402
from src.loss_detect import LossDetector        # noqa: E402
from src.main import Pipeline                   # noqa: E402
from src.reacquire import Reacquirer            # noqa: E402
from src.tracker import make_tracker, pixel_to_box  # noqa: E402
from src.ui import draw_overlay                 # noqa: E402


def env_info() -> dict:
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor()
    return {"cpu": cpu, "python": platform.python_version(),
            "opencv": cv2.__version__, "numpy": np.__version__,
            "cv2_threads": cv2.getNumThreads()}


def _stats(ms: list) -> tuple[float, float]:
    return (float(np.mean(ms)), float(np.percentile(ms, 95)))


def bench_full_pipeline(video: str, pixel: tuple) -> list[dict]:
    """Stream the video through the complete pipeline, timing per component."""
    i, j = pixel
    rows = []
    per_state: dict = {}
    decode_ms, overlay_ms, e2e_ms = [], [], []
    with VideoSource(video) as src:
        first = src.read()
        tracker = make_tracker("mosse")
        tracker.init(first, pixel_to_box(j, i, 128, src.width, src.height))
        pipe = Pipeline(tracker, LossDetector(), Reacquirer())
        trail = []
        idx = 0
        while True:
            t0 = time.perf_counter()
            frame = src.read()
            t1 = time.perf_counter()
            if frame is None:
                break
            idx += 1
            box, conf = pipe.process(idx, frame)
            t2 = time.perf_counter()
            draw_overlay(frame, box, trail, pipe.state, 0.0, "mosse", conf)
            t3 = time.perf_counter()
            decode_ms.append((t1 - t0) * 1000)
            per_state.setdefault(pipe.state, []).append((t2 - t1) * 1000)
            overlay_ms.append((t3 - t2) * 1000)
            e2e_ms.append((t3 - t0) * 1000)

    for state, ms in sorted(per_state.items()):
        mean, p95 = _stats(ms)
        rows.append({"backend": "mosse", "scope": "process", "state": state,
                     "frames": len(ms), "mean": mean, "p95": p95})
    for scope, ms in (("decode", decode_ms), ("overlay", overlay_ms),
                      ("e2e (decode+process+overlay)", e2e_ms)):
        mean, p95 = _stats(ms)
        rows.append({"backend": "mosse", "scope": scope, "state": "all",
                     "frames": len(ms), "mean": mean, "p95": p95})
    # the number the real-time claim hangs on: worst state, end to end
    worst = max(per_state, key=lambda s: np.mean(per_state[s]))
    wmean = float(np.mean(per_state[worst])) + float(np.mean(decode_ms)) \
        + float(np.mean(overlay_ms))
    rows.append({"backend": "mosse", "scope": "e2e worst state", "state": worst,
                 "frames": len(per_state[worst]), "mean": wmean,
                 "p95": float(np.percentile(per_state[worst], 95))
                 + float(np.mean(decode_ms)) + float(np.mean(overlay_ms))})
    return rows


def bench_update_only(video: str, pixel: tuple, backend: str) -> dict:
    """Context row: raw tracker.update() cost for the non-PSR backends."""
    i, j = pixel
    ms = []
    with VideoSource(video) as src:
        first = src.read()
        tracker = make_tracker(backend)
        tracker.init(first, pixel_to_box(j, i, 128, src.width, src.height))
        for frame in src.frames():
            t0 = time.perf_counter()
            tracker.update(frame)
            ms.append((time.perf_counter() - t0) * 1000)
    mean, p95 = _stats(ms)
    return {"backend": backend, "scope": "update-only", "state": "-",
            "frames": len(ms), "mean": mean, "p95": p95}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True, help="i,j (row,col) on frame 0")
    ap.add_argument("--threads", type=int, default=0,
                    help="pin cv2.setNumThreads (0 = library default)")
    ap.add_argument("--others", action="store_true",
                    help="also bench update-only rows for mosse-cv2/kcf/csrt")
    args = ap.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)
    pixel = tuple(int(v) for v in args.pixel.split(","))

    env = env_info()
    # on some parallel backends (e.g. macOS GCD) getNumThreads() keeps
    # reporting core count even after a successful pin - record the pin too
    env["threads_pinned"] = args.threads or "default"
    print("env:", json.dumps(env))
    rows = bench_full_pipeline(args.video, pixel)
    if args.others:
        for backend in ("mosse-cv2", "kcf", "csrt"):
            rows.append(bench_update_only(args.video, pixel, backend))

    print("\n| backend | scope | state | frames | mean ms | p95 ms | fps @ mean |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['backend']} | {r['scope']} | {r['state']} | {r['frames']} "
              f"| {r['mean']:.2f} | {r['p95']:.2f} | {1000 / r['mean']:.0f} |")


if __name__ == "__main__":
    main()
