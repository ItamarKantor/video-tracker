"""Build the SYNTHETIC exit-and-return test clip (forward + reversed).

The real sample clip contains a clean loss event (target fully exits around
f455) but the target never re-enters, so re-acquisition cannot be exercised
end-to-end on it. Playing frames [0..end] forward and then the same frames
reversed guarantees the target exits and re-enters symmetrically - a
controlled harness for developing and testing loss detection +
re-acquisition.

SYNTHETIC - development/testing only, never the graded demo. The output is
written into input/ (gitignored: large media stays out of the public repo).

Frames are buffered JPEG-compressed (~0.5 MB/frame instead of ~6 MB raw) so
the reversed pass needs neither gigabytes of RAM nor slow codec seeking.

Usage:
    python experiments/make_synthetic_loop.py --video PATH [--end 560]
                                              [--out input/synthetic_loop_DEV.mp4]
"""

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = str(ROOT / "input" / "synthetic_loop_DEV.mp4")
JPEG_QUALITY = 95


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True,
                    help="path to the provided sample clip")
    ap.add_argument("--end", type=int, default=560,
                    help="last source frame of the forward pass (default 560: "
                         "~100 frames after the sample clip's loss event)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer for {args.out}")

    encoded = []
    for _ in range(args.end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise SystemExit("JPEG encode failed")
        encoded.append(buf)
    cap.release()

    for buf in reversed(encoded[:-1]):  # skip the pivot frame (no duplicate)
        writer.write(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    writer.release()

    total = len(encoded) * 2 - 1
    print(f"wrote {args.out}: {total} frames ({len(encoded)} forward + "
          f"{len(encoded) - 1} reversed) at {fps:.1f} fps, {w}x{h}")


if __name__ == "__main__":
    main()
