"""Exploratory tracker variant: higher template learning rate.

The Stage-2 diagnosis showed the box stays ON target through the entire
PSR collapse - the appearance (blur + zoom + rotation burst) simply
changes faster than eta=0.125 can absorb. The direct lever is eta itself:
adapt faster, so the template follows the burst. Risk to measure later on
other clips: faster adaptation also poisons faster (mitigated by the
existing freeze-on-SUSPECT policy).

Usage: python experiments/remediation/proto_07_fast_adapt.py \
           --video <path> --pixel <i>,<j> [--eta 0.25]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mosse import MosseTracker              # noqa: E402
import harness                                  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pixel", required=True)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    i, j = (int(v) for v in args.pixel.split(","))
    harness.run(f"proto_07 eta={args.eta}",
                lambda: MosseTracker(learn_rate=args.eta), args.video, (i, j),
                csv_path=args.csv)
