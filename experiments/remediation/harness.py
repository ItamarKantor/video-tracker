"""Shared Stage-2 measurement harness.

Parameterized twin of experiments/grip_baseline.py: same loop, same
metrics, but the tracker / re-acquirer are injected so each prototype can
override exactly one behavior. src/ is the frozen v1 baseline and is only
imported, never modified.

Metrics per run: frames held (init -> first LOST), state transitions,
re-acquisition attempts/hits (+ first-hit frame), mean tracker-update ms
and implied fps, PSR trace CSV.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.io_video import VideoSource            # noqa: E402
from src.loss_detect import LossDetector        # noqa: E402
from src.main import Pipeline                   # noqa: E402
from src.reacquire import Reacquirer            # noqa: E402
from src.tracker import box_center, pixel_to_box  # noqa: E402


class CountingReacquirer(Reacquirer):
    """v1 re-acquirer + attempt/hit counters (instrumentation only)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attempts = 0
        self.hits = 0
        self.first_hit_frame = None
        self.frame_idx = 0  # set by the loop before each attempt

    def attempt(self, frame):
        self.attempts += 1
        box = super().attempt(frame)
        if box is not None:
            self.hits += 1
            if self.first_hit_frame is None:
                self.first_hit_frame = self.frame_idx
        return box


def run(label, tracker_factory, video, pixel, reacquirer_factory=None,
        observe_every=0, csv_path=None, box_size=128, verbose=True):
    """Run the v1 pipeline with injected components; return a metrics dict."""
    i, j = pixel
    rows = []
    update_ms = []
    with VideoSource(video) as src:
        first = src.read()
        tracker = tracker_factory()
        tracker.init(first, pixel_to_box(j, i, box_size, src.width, src.height))
        reacq = reacquirer_factory() if reacquirer_factory else CountingReacquirer()
        pipe = Pipeline(tracker, LossDetector(), reacq)
        for idx, frame in enumerate(src.frames(), start=1):
            if hasattr(reacq, "frame_idx"):
                reacq.frame_idx = idx
            t0 = time.perf_counter()
            box, conf = pipe.process(idx, frame)
            update_ms.append((time.perf_counter() - t0) * 1000)
            if (observe_every and pipe.state == "TRACKING"
                    and idx % observe_every == 0 and hasattr(reacq, "observe")):
                reacq.observe(tracker.last_good_patch)
            cx, cy = box_center(box) if box is not None else (-1, -1)
            rows.append((idx, f"{conf:.2f}" if conf is not None else "",
                         cx, cy, pipe.state))

    if csv_path:
        with open(csv_path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["frame", "psr", "cx", "cy", "state"])
            w.writerows(rows)

    first_lost = next((r[0] for r in rows if r[4] == "LOST"), None)
    mean_ms = statistics.fmean(update_ms)
    track_ms = [m for m, r in zip(update_ms, rows) if r[4] == "TRACKING"]
    metrics = {
        "label": label,
        "frames": len(rows),
        "frames_held": (first_lost - 1) if first_lost else len(rows),
        "transitions": " -> ".join(f"{s}@f{k}" for k, s in pipe.transitions),
        "attempts": getattr(reacq, "attempts", None),
        "hits": getattr(reacq, "hits", None),
        "first_hit": getattr(reacq, "first_hit_frame", None),
        "mean_ms": mean_ms,
        "tracking_ms": statistics.fmean(track_ms) if track_ms else float("nan"),
        "fps": 1000.0 / mean_ms,
    }
    if verbose:
        print(f"[{label}] held {metrics['frames_held']} frames | "
              f"transitions: {metrics['transitions'] or 'none'} | "
              f"reacq {metrics['attempts']}/{metrics['hits']} "
              f"(first hit f{metrics['first_hit']}) | "
              f"process {mean_ms:.2f} ms mean "
              f"(TRACKING-state {metrics['tracking_ms']:.2f} ms) "
              f"= {metrics['fps']:.0f} fps")
    return metrics
