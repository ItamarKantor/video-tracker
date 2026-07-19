"""Phase B benchmark: fps of candidate OpenCV trackers on real 1080p frames.

Measures, per tracker and per init-box size:
  * tracker.update() time per frame (decode excluded), over N frames x R runs
  * video decode time per frame, measured separately (identical for all
    trackers, so isolating it keeps the comparison clean)
  * an estimated end-to-end fps = 1 / (decode + update)

Method notes:
  * Frames are used at native resolution (1920x1080) - no downscaling.
  * The first update after init is excluded from stats (one-time allocations).
  * Median of R runs is reported to smooth OS scheduling noise.
  * The fixed default target pixel is a benchmark-only convenience and
    must never leak into src/ - there, video and pixel are always runtime
    inputs.

Usage:
    python experiments/bench_trackers.py --video PATH [--pixel I,J]
                                         [--sizes 64,128] [--frames 300]
                                         [--runs 3]
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

# Benchmark-only default target (approved fixed pixel for apples-to-apples
# runs on the provided sample clip; the clip path is a runtime argument).
DEFAULT_PIXEL = (857, 1167)  # (i, j) = (row, col) on frame 0
SNAPSHOT_FRAMES = (50, 150, 299)  # checkpoints for the visual-hold check


def make_tracker(name: str):
    """Factory for the candidate trackers (fresh instance per run)."""
    factories = {
        "KCF": cv2.TrackerKCF_create,
        "CSRT": cv2.TrackerCSRT_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create,
    }
    return factories[name]()


def pixel_to_box(i: int, j: int, size: int, frame_shape) -> tuple:
    """Square box of side `size` centered on pixel (i,j), clamped into the frame."""
    h, w = frame_shape[:2]
    x = min(max(j - size // 2, 0), w - size)
    y = min(max(i - size // 2, 0), h - size)
    return (x, y, size, size)


def read_frames(video: str, n_frames: int):
    """Decode frame 0 + n_frames frames, timing decode cost per frame."""
    cap = cv2.VideoCapture(video)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame 0 of {video}")
    frames, decode_ms = [], []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        dt = (time.perf_counter() - t0) * 1000.0
        if not ok:
            break
        frames.append(frame)
        decode_ms.append(dt)
    cap.release()
    return first, frames, decode_ms


def bench_one(name: str, first, frames, box, snap_dir: Path | None):
    """One run: init tracker on frame 0, time update() on every later frame.

    Returns (per-frame update times in ms, tracker-reported-ok count,
    frame index of first reported failure or None).
    """
    tracker = make_tracker(name)
    tracker.init(first, box)
    update_ms, ok_count, first_fail = [], 0, None
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        ok, bbox = tracker.update(frame)
        update_ms.append((time.perf_counter() - t0) * 1000.0)
        if ok:
            ok_count += 1
        elif first_fail is None:
            first_fail = idx
        if snap_dir is not None and idx in SNAPSHOT_FRAMES:
            snap = frame.copy()
            if ok:
                x, y, w, h = (int(v) for v in bbox)
                cv2.rectangle(snap, (x, y), (x + w, y + h), (0, 0, 255), 3)
            label = f"{name} box={box[2]} frame={idx} ok={ok}"
            cv2.putText(snap, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)
            cv2.imwrite(str(snap_dir / f"{name}_{box[2]}_f{idx}.png"), snap)
    return update_ms[1:], ok_count, first_fail  # drop first update (warm-up)


def env_info() -> dict:
    """Record the machine/library state so the numbers are defensible."""
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor()
    return {
        "cpu": cpu,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cv2_threads": cv2.getNumThreads(),
    }


def stats_ms(samples: list) -> dict:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": float(np.percentile(samples, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True,
                    help="path to the provided sample clip (or any test video)")
    ap.add_argument("--pixel", default=f"{DEFAULT_PIXEL[0]},{DEFAULT_PIXEL[1]}",
                    help="init pixel as i,j (row,col) on frame 0")
    ap.add_argument("--sizes", default="64,128")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--snap-dir", default=None,
                    help="if set, save box snapshots at checkpoints (run 0 only)")
    args = ap.parse_args()

    i, j = (int(v) for v in args.pixel.split(","))
    sizes = [int(v) for v in args.sizes.split(",")]
    snap_dir = Path(args.snap_dir) if args.snap_dir else None
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)

    print("env:", json.dumps(env_info()))
    first, frames, decode_ms = read_frames(args.video, args.frames)
    print(f"video: {args.video}")
    print(f"frames timed: {len(frames)} at {first.shape[1]}x{first.shape[0]}, "
          f"decode {stats_ms(decode_ms)['mean']:.2f} ms/frame mean")

    results = []
    for name in ("KCF", "CSRT", "MOSSE"):
        for size in sizes:
            box = pixel_to_box(i, j, size, first.shape)
            runs = []
            for r in range(args.runs):
                sd = snap_dir if r == 0 else None
                update_ms, ok_count, first_fail = bench_one(
                    name, first, frames, box, sd)
                runs.append((stats_ms(update_ms), ok_count, first_fail))
            # median run by mean update time
            runs.sort(key=lambda t: t[0]["mean"])
            s, ok_count, first_fail = runs[len(runs) // 2]
            decode_mean = stats_ms(decode_ms)["mean"]
            row = {
                "tracker": name, "box": size,
                "update_mean": s["mean"], "update_median": s["median"],
                "update_p95": s["p95"],
                "update_fps": 1000.0 / s["mean"],
                "e2e_fps": 1000.0 / (s["mean"] + decode_mean),
                "reported_ok": f"{ok_count}/{len(frames)}",
                "first_fail": first_fail,
            }
            results.append(row)
            print(json.dumps(row))

    print("\n| Tracker | Box | update mean ms | median | p95 | update-only fps "
          "| est. e2e fps | reported ok | first fail |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['tracker']} | {r['box']} | {r['update_mean']:.2f} "
              f"| {r['update_median']:.2f} | {r['update_p95']:.2f} "
              f"| {r['update_fps']:.1f} | {r['e2e_fps']:.1f} "
              f"| {r['reported_ok']} | {r['first_fail']} |")


if __name__ == "__main__":
    main()
