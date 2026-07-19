# Video Tracker

Real-time single-object video tracker in Python. Give it a video and one
pixel on the first frame. It tracks that object, detects when the object is
lost, and re-acquires it when it re-enters the frame. CPU only, no GPU. On
an Apple M2 it stays well above 30 fps at 1080p, with a conservative
slower-CPU projection in the Performance section below.

## Demo

**Watch the demo:** [tracking_demo_delivery.mp4](https://github.com/ItamarKantor/video-tracker/releases/download/v1.0/tracking_demo_delivery.mp4)
(16 MB, on the [v1.0 release](https://github.com/ItamarKantor/video-tracker/releases/tag/v1.0)).
Rendered by the pipeline on the provided sample clip, unmodified, init pixel
`i=750, j=1259`. Timeline:

- `TRACKING` from frame 1, about 7.6 s of clean tracking.
- `LOST@456`, fast camera motion breaks the correlation.
- `RE-ACQUIRED@464`, verified feature match re-seeds the tracker.
- Two more loss/recovery cycles through the spin (`LOST@496 → RE-ACQUIRED@504`,
  `LOST@519 → RE-ACQUIRED@527`).
- `LOST@537`, then honest LOST to the end. The camera flies away and
  descends over a building, the target is not matchable again, and the
  system draws no box rather than inventing one.

The first recovery was checked independently of the tracker: the target's
world point was optical-flow-tracked across the whole loss and landed 42 px
from the recovered box on a 128 px box, an ORB+RANSAC homography agreed at
44 px, and the patches matched with 191 inliers.

Render the same output on any clip:

```bash
python experiments/render_demo.py --video <video> --pixel <i>,<j>
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# select the target by mouse (click, then ENTER/SPACE to confirm):
python -m src.main path/to/video.mp4

# or give the target pixel directly as [i],[j] = row,col on frame 0:
python -m src.main path/to/video.mp4 --pixel 410,650
```

Keys: `q` / `ESC` quit. Flags: `--tracker {mosse,mosse-cv2,kcf,csrt}`,
`--box-size N`, `--fast` (no throttle), `--no-display` (headless),
`--reacq-every N`. Run `-h` for all.

## Requirements

- Python 3.13 (tested on 3.13.12)
- Pinned in [`requirements.txt`](requirements.txt):

| Package | Version | Used for |
|---|---|---|
| opencv-contrib-python | 4.13.0.92 | video I/O, UI, ORB, reference trackers |
| numpy | 2.5.1 | the MOSSE tracker math (FFT, statistics) |

Dev extras (tests): [`requirements-dev.txt`](requirements-dev.txt)
(pytest 9.1.1).

## How it works

Three parts behind one state machine. It starts in **TRACKING**, from the
chosen pixel.

- **TRACKING**: healthy confidence (PSR) adapts the template, a suspect
  frame keeps tracking but freezes it. Drops to **LOST** when confidence
  stays low for several frames, or when the motion guard finds the box
  stuck on a static overlay.
- **LOST**: tracking stops, and the full frame is searched every N-th
  frame. A verified ORB + RANSAC match goes to **RE-ACQUIRED** and re-seeds
  a fresh tracker at the found location.
- **RE-ACQUIRED**: a short probation. Healthy confidence returns to
  **TRACKING**. A collapse (a re-seed on background) or the motion guard
  drops it back to **LOST**.

**1. Tracking: our own MOSSE** ([`src/mosse.py`](src/mosse.py), Bolme et
al., CVPR 2010). OpenCV's binding exposes only `init`/`update`, with no
response map, no confidence, no template control. Our ~150-line FFT
implementation exposes:
  - **PSR** (peak-to-sidelobe ratio) of every response, the confidence
    signal everything else uses.
  - a **template-freeze switch**. Without it the filter keeps learning as
    the target leaves and poisons itself onto the background (measured: PSR
    2900 on a dead frame-edge patch).
  - a quality-gated **appearance buffer**, the sharpest recent patch, kept
    for re-acquisition.

The cv2 trackers (`mosse-cv2`, `kcf`, `csrt`) stay available for A/B. On the
sample clip, MOSSE-128 was the only tracker that survived the pan, zoom and
~177° rotation, at ~0.7 ms/frame. CSRT drifted silently while reporting
success, which is why the state machine trusts PSR, not backend flags.

**2. Loss detection: relative PSR with hysteresis**
([`src/loss_detect.py`](src/loss_detect.py)). Healthy PSR varies per scene
(~50 on desert aerials, 30-60 indoors), so a fixed threshold fires late on
high-PSR scenes and never on low-PSR ones. The detector keeps a running
median of healthy PSR and flags a frame low when
`psr < max(0.35 × median, floor)`, with an absolute override (PSR ≥ 20 is
never low) and two-sided hysteresis (3 low frames confirm LOST, one lucky
frame does not reset). First suspicion freezes the template, which is cheap
and reversible where poisoning is not.

**3. Re-acquisition: context patches + ORB + RANSAC**
([`src/reacquire.py`](src/reacquire.py)). The stored appearance is the
target plus 2x context, with the target box recorded inside. On repetitive
texture a window patch loses every match to the ratio test (0 of 116),
while the 2x context gives 27 inliers on the same frames. While LOST the
full-res frame is ORB-detected once per attempt and matched against the
stored contexts. At least 8 RANSAC-consistent matches plus size checks
accept a hit, and the homography maps the inner box back, restoring current
scale for free. The memory is sticky: only the first-loss appearance is
matched, because a later re-seed can be imperfect and poison it. A re-seed
on background self-corrects (PSR collapses, back to LOST). A re-seed on a
similar real structure does not (see limitations).

**4. Static-overlay guard** ([`src/main.py`](src/main.py), `StaticGuard`).
Burned-in HUD graphics correlate perfectly forever, so PSR cannot flag
them. Each frame the guard estimates the global similarity transform (sparse
optical flow on a downscaled pair) and predicts how a point at the box's
position should move. A box that stays far below that prediction for 8
frames is stuck to the screen and reverts to LOST. Referencing the
prediction to the box's own position makes it abstain for a target at the
rotation/zoom center. About 2 ms/frame, always on.

The trajectory overlay shows the last ~200 points. On a moving camera,
older image-space points no longer map to world positions, so an unbounded
trail would be a meaningless smear.

## Generality: any clear object, no per-target tuning

Validated on six targets on the sample clip (structure corner, vegetation
cluster, settlement structures, open ground, an alternate corner,
near-featureless desert) plus a synthetic exit-and-return harness, all with
identical code and constants, every run ending in an honest state. Also
tested cold on held-out clips (indoor, a different camera). Generality here
means the tracker locks any clear target and always ends honestly. Recovery
identity on repeated look-alike structures is a separate, measured limit
(see below). Per-target timelines:
[`experiments/remediation/results.md`](experiments/remediation/results.md).

## Architecture

```
src/
  main.py         CLI, main loop, the state machine (Pipeline)
  io_video.py     streaming video input (path/URL), metadata
  tracker.py      BaseTracker interface + OpenCV backend adapters
  mosse.py        our MOSSE: FFT correlation, PSR, freeze, appearance buffer
  loss_detect.py  relative-PSR loss detector with hysteresis
  reacquire.py    ORB+RANSAC global re-detection while LOST
  ui.py           overlay (box, trajectory, state, fps) + mouse selection
experiments/      benchmark & demo scripts + measured results
tests/            pytest unit tests (synthetic frames, no video needed)
```

## Performance (1920×1080, CPU only, no GPU)

Full pipeline on an Apple M2, pinned to one thread. Worst state is LOST (it
carries the context search). Full tables in
[`experiments/bench_results.md`](experiments/bench_results.md):

| State (1 thread) | mean ms/frame | implied fps |
|---|---|---|
| TRACKING (incl. ~2 ms guard) | 3.0 | ~333 |
| LOST, amortized (search every 8th frame) | 10.7 | ~94 |
| projected worst state, 3× slower CPU | ~32 | **~31 (still ≥ 30)** |

The single frame that runs a search spikes to ~95-103 ms single-threaded.
The cadence (`--reacq-every`, default 8) amortizes it while the tracker is
idle, so 30 fps is sustained throughput, not a per-frame guarantee.
Reproduce with `python experiments/bench_pipeline.py --video <clip>
--pixel <i>,<j> [--threads 1]`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

30 unit tests (fixed seeds, synthetic frames): MOSSE lock, PSR collapse,
the freeze switch, the context buffer, loss-detector transitions and
hysteresis, re-acquisition on planted/featureless/unrelated targets plus
the sticky-memory and inner-box invariants, the overlay guard's
fire/abstain, the fps-readout bounds, and the pinned defaults.

## Known limitations

- **Violent zoom-plus-rotation bursts** defeat the whole correlation-tracker
  family (measured: ~0.3 %/frame zoom and up to ~1.5°/frame rotation,
  reaching +128° and 1.5× within ~90 frames). Eleven variants (motion
  prediction, padded windows, scale search, higher learning rates,
  combinations, CSRT) all failed the same segment. The answer is honest
  LOST plus fast re-acquisition (recoveries 4-8 frames after loss), not
  heroic tracking.
- **No scale adaptation while tracking.** MOSSE uses a fixed window, so a
  strong zoom degrades correlation until an honest LOST. Re-acquisition
  restores the box at current scale via the homography.
- **Re-acquisition needs textured context and lands approximately under a
  large viewpoint change.** A featureless surrounding cannot be re-found
  (the app warns at loss time). Under a very different viewpoint the target
  is recovered onto its content region, not pixel-perfectly. The memory is
  sticky by choice: correctness over adaptivity.
- **Wrong-instance re-acquisition on repeated near-identical structures.**
  Re-acquisition can lock onto the wrong instance and track it stably.
  Measured: 4 of 8 init pixels gave false or suspect recoveries (up to
  ~925 px off, one accepted at the minimum inlier count). Neither safeguard
  catches it by construction: RANSAC proves geometric consistency and PSR
  proves a sharp lock, and a wrong-but-real structure satisfies both. Same
  class as the overlay lock. The fix (not built) is a motion-plausibility
  gate on re-seeds: the guard already estimates the global affine each
  frame, so accumulated camera motion during LOST predicts where the target
  should reappear, and a candidate landing far outside that region is
  rejected. The prediction drifts over long losses and would need a
  widening tolerance, which is why it is future work.
- **Burned-in overlays** are perfect correlation targets. The guard reverts
  a box stuck on one, but initializing on an overlay is the user's
  responsibility.
- **Per-frame spikes while LOST** (~95-103 ms single-threaded). The mean
  holds ≥ 30 fps, but a strict per-frame deadline would need an async
  search.
- **Re-acquisition frame indices vary across OpenCV builds** (same version,
  different compilation): marginal matches resolve differently. The
  behaviour (honest LOST, genuine recovery, no overlay lock) is stable
  across the two builds tested.
- **Detector thresholds are fixed values**, validated cold on the aerial
  sample at 60 fps, the synthetic harness, and held-out indoor clips. They
  count in frames, which is the safer (slower to fire) direction at lower
  frame rates.
