# ASIO Video Tracker

Real-time single-object video tracker in Python: give it a video and **one
pixel** on the first frame, and it tracks that object through the video,
detecting when the object is lost and **re-acquiring it when it re-enters
the frame**, all on CPU at well over 30 fps on full-HD footage.

## Demo

**Watch the demo:** [tracking_demo_delivery.mp4](https://github.com/ItamarKantor/video-tracker/releases/download/v1.0/tracking_demo_delivery.mp4)
(16 MB, attached to the [v1.0 release](https://github.com/ItamarKantor/video-tracker/releases/tag/v1.0),
since large media is kept out of git). It is rendered by the pipeline
itself on the **provided sample clip, unmodified** (init pixel
`i=750, j=1259`). Exact timeline, so you can scrub to each event:

- `TRACKING` from frame 1, ~7.6 s of clean tracking (green box +
  trajectory)
- `LOST@456`, fast camera motion destroys the correlation (red banner)
- `RE-ACQUIRED@464`, verified feature match re-seeds the tracker (yellow)
- two more honest loss/recovery cycles through the camera spin
  (`LOST@496 → RE-ACQUIRED@504`, `LOST@519 → RE-ACQUIRED@527`)
- `LOST@537` → **honest LOST to the end of the clip**. This long tail is
  designed behavior, not a failure: the camera flies away and descends
  over a building, the target is never matchable again, and the system
  refuses to draw a box rather than inventing one.

The demo's first recovery was independently verified to land within
**~42 px of the true target on a 128 px box, with 191 appearance inliers**:
the target's world point was optical-flow-tracked frame-by-frame across
the whole loss (pyramidal Lucas-Kanade, discarding any lost-status
result), and cross-checked with an ORB+RANSAC homography between the
pre-loss and post-recovery frames (44 px agreement), because "the box
came back" is not evidence it came back to the *right* place.

To render the same annotated output on any clip:

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

Keys during playback: `q` / `ESC` quits. Useful flags: `--tracker
{mosse,mosse-cv2,kcf,csrt}` (backend A/B), `--box-size N`, `--fast`
(no real-time throttling), `--no-display` (headless benchmark mode),
`--reacq-every N` (re-detection cadence while LOST). Run `-h` for all.

## Requirements

- Python 3.13 (developed and tested on 3.13.12)
- Pinned in [`requirements.txt`](requirements.txt):

| Package | Version | Used for |
|---|---|---|
| opencv-contrib-python | 4.13.0.92 | video I/O, UI, ORB, reference trackers |
| numpy | 2.5.1 | the MOSSE tracker math (FFT, statistics) |

Dev extras (tests): [`requirements-dev.txt`](requirements-dev.txt)
(pytest 9.1.1).

## How it works

Three cooperating parts behind one explicit state machine (hybrid
tracking-by-detection). The machine has three states and starts in
**TRACKING**, initialised from the chosen pixel:

- In **TRACKING**, a healthy confidence (PSR) lets the template keep
  adapting, while a suspect frame keeps tracking but freezes the template.
  The state falls to **LOST** when confidence stays low for several
  consecutive frames, or when the motion guard finds the box glued to a
  static overlay.
- In **LOST**, the tracker stops and the full frame is searched for the
  target every N-th frame. A verified ORB + RANSAC match promotes the state
  to **RE-ACQUIRED** and re-seeds a fresh tracker at the found location.
- In **RE-ACQUIRED**, a short probation follows. If confidence stays
  healthy it returns to **TRACKING**. If it collapses (a re-seed that
  landed on background) or the motion guard fires, it drops back to
  **LOST**.

**1. Tracking: our own MOSSE correlation filter**
([`src/mosse.py`](src/mosse.py), Bolme et al., CVPR 2010). We implemented
MOSSE ourselves instead of using `cv2.legacy.TrackerMOSSE` for a functional
reason: the OpenCV binding exposes only `init`/`update`, with no response
map, no confidence, no control over template adaptation. Our ~150-line FFT
implementation exposes:
  - **PSR** (peak-to-sidelobe ratio) of every correlation response, the
    confidence signal everything else is built on.
  - a **template-freeze switch**. Without it, the online filter keeps
    learning while the target leaves the frame and poisons itself onto the
    background (we measured a poisoned template reporting PSR ≈ 2900 while
    pinned to a dead patch of frame edge).
  - a quality-gated **appearance buffer**, the sharpest recent patch of
    the target, kept for re-acquisition.

The cv2 trackers (`mosse-cv2`, `kcf`, `csrt`) remain available behind the
same interface for A/B comparison, benchmarks below. On the aerial sample
clip, our benchmark found MOSSE-128 was the only tracker that survived the
camera's pan + zoom + ~177° rotation, at ~0.7 ms/frame. CSRT (the usual
"accurate" choice) drifted silently while still reporting success, which
is exactly why the state machine trusts PSR, not backend success flags.

**2. Loss detection: relative PSR with hysteresis**
([`src/loss_detect.py`](src/loss_detect.py)). Healthy PSR levels vary per
scene (we measured ~50 on desert aerials, 30-60 indoors), so a fixed
threshold is wrong twice: it fires late on high-PSR scenes and never on
low-PSR ones. The detector keeps a running median of recent healthy PSR
and flags a frame LOW when `psr < max(0.35 × median, floor)`, with two
absolute guards (a frame in Bolme's strong-lock band ≥ 20 is never LOW)
and two-sided hysteresis (3 consecutive LOW frames confirm LOST, and one
lucky frame mid-collapse does not reset the evidence). On the first
suspicion the template freezes, because freezing is cheap and reversible
while poisoning is neither.

**3. Re-acquisition: CONTEXT patches + ORB + ratio test + RANSAC**
([`src/reacquire.py`](src/reacquire.py)). The stored appearance is the
target plus 2× surrounding context, with the target box recorded inside.
This is a measured decision: on locally repetitive texture a window-sized
patch is annihilated by the ratio test (every descriptor has near-twins,
and 0/116 matches survived), while the 2× context captures unique larger
structure (27 vs 7 RANSAC inliers on the same frames). While LOST, the
full-resolution frame is ORB-detected once per attempt and matched against
the stored contexts. At least 8 RANSAC-consistent matches plus size sanity
checks accept a hit, and the homography maps the INNER target box back,
which also restores the target's current scale for free. The matching
memory is **sticky**: only the appearance stored at the *first* loss is
ever matched, because a later re-seed can be imperfect and would poison the
memory (measured). A re-seed that lands on background self-corrects, as PSR
collapses within frames and the machine returns to LOST. A re-seed that
lands on a similar *real* structure does not (see Known limitations).

**4. Static-overlay guard** ([`src/main.py`](src/main.py), `StaticGuard`).
Burned-in screen furniture (HUD text, telemetry) is the one failure PSR
can never flag: static content correlates perfectly forever. Each frame we
estimate the global similarity transform (sparse optical flow on a
downscaled pair) and predict how a point at the box's own position should
move. A box that stays far below that prediction for 8 straight frames is
glued to the screen, not the scene, so it reverts to LOST. Referencing the
prediction to the box's position makes the guard correctly abstain for a
target at the rotation/zoom center. Cost ~2 ms/frame, always on.

A note on the trajectory overlay: it shows the most recent ~200 points by
design. On a moving camera, older image-space points no longer correspond
to world positions, so an unbounded trail would draw a meaningless smear
rather than the object's motion.

## Generality: any clear object, no per-target tuning

The pipeline was validated on **six different targets on the sample clip**
(the primary structure corner, a vegetation cluster, settlement
structures, open ground with tracks, an alternate structure corner, and
near-featureless desert) plus a synthetic exit-and-return harness, all with
identical code and constants, every run ending in honest states. It was
also validated on held-out clips not included in this repo, covering indoor
scenes and a different camera. Generality here means the tracker locks any
clear target and always resolves to an honest state. Recovery **identity**
on repeated look-alike structures is a separate, measured limitation (see
Known limitations below). Full per-target timelines:
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

## Performance (real 1920×1080 frames, CPU only, no GPU)

Full pipeline on an Apple M2, worst state is LOST (it carries the context
search). Complete tables and method in
[`experiments/bench_results.md`](experiments/bench_results.md):

| State (pinned to 1 thread) | mean ms/frame | implied fps |
|---|---|---|
| TRACKING (incl. ~2 ms guard) | 3.0 | ~333 |
| LOST, amortized (search every 8th frame) | 10.7 | ~94 |
| projected worst state on a 3× slower CPU | ~32 | **~31 (still ≥ 30)** |

The individual frame that runs a context-matching attempt spikes to
~95-103 ms single-threaded, and the cadence (`--reacq-every`, default 8)
amortizes it while the tracker is idle. Reproduce with
`python experiments/bench_pipeline.py --video <clip> --pixel <i>,<j>
[--threads 1]`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

30 deterministic unit tests (fixed seeds, synthetic frames): MOSSE locking
onto a known translation, PSR collapse on unrelated content, the freeze
switch, the quality-gated context buffer, all loss-detector transitions
and hysteresis, re-acquisition on planted/featureless/unrelated targets
plus the sticky-memory and inner-box mapping invariants, the
static-overlay guard's fire/abstain semantics, the fps-readout sanity
bounds, and the measured default constants.

## Known limitations

- **Violent appearance transients defeat the correlation family**: a burst
  of simultaneous zoom + rotation (measured on the sample footage:
  ~0.3 %/frame zoom with up to ~1.5°/frame rotation, accumulating to
  +128°/1.5× within ~90 frames) collapses the correlation no matter the
  search geometry. We measured eleven tracker variants (motion
  prediction, padded windows, scale search, higher learning rates, their
  combinations, and CSRT) against the same segment and none rode it out.
  The system's answer is honest LOST plus fast re-acquisition (recoveries
  observed 4-8 frames after the loss), not heroic tracking.
- **No scale adaptation while tracking**: MOSSE uses a fixed-size window,
  so a strong zoom degrades correlation until an honest LOST. Re-acquisition
  restores the box at the target's current scale via the homography.
- **Re-acquisition needs a textured context and lands approximately under
  a large viewpoint change**: a featureless surrounding (blank surfaces)
  cannot be re-found, and the app warns explicitly at loss time. A target
  returning under a very different viewpoint is recovered onto its content
  region, not pixel-perfectly. The matching memory is deliberately sticky
  (first-loss appearance only), choosing correctness over adaptivity.
- **Wrong-instance re-acquisition on repeated near-identical structures.**
  On scenes full of look-alike structures, re-acquisition can lock onto
  the wrong instance and then track it stably and confidently: measured on
  the sample clip, **4 of 8 tested init pixels** produced false or suspect
  recoveries (offsets up to ~925 px from the true target, and one accept
  passed at exactly the minimum inlier count). The insight is *why* this
  is invisible to the existing safeguards: the RANSAC gate proves a match
  is geometrically self-consistent, and PSR proves the tracker has a sharp
  lock, yet a wrong-yet-real structure satisfies both perfectly. Neither
  signal can express "this is the wrong instance." It is the same class of
  error as the burned-in-overlay lock, a sensor that cannot see the
  failure mode by construction, and like it, it requires a different sensor
  rather than a better threshold. Proposed future fix (not built):
  a motion-plausibility gate on re-seeds. The guard already estimates the
  global affine each frame, so accumulated camera motion during LOST
  predicts where the target should reappear, and candidates implausibly
  far outside that region could be rejected or demoted. Honestly, the
  prediction drifts over long LOST stretches and would need a widening
  tolerance, which is why it is future work rather than a late change.
- **Burned-in screen overlays** (HUD text, telemetry) are perfect
  correlation targets. The motion guard detects and reverts a box that
  gets glued to them, but initializing ON an overlay is still the user's
  responsibility.
- **Per-frame spikes while LOST**: a context-matching attempt takes
  ~95-103 ms single-threaded, and the every-8th-frame cadence keeps mean
  throughput ≥ 30 fps, but a strict per-frame deadline would need the
  search sliced asynchronously.
- **Exact re-acquisition frame indices vary across OpenCV builds**
  (packaging/compilation level, same version): marginal-inlier candidates
  near the acceptance threshold resolve differently. The qualitative
  behavior (honest LOST, genuine recovery, no overlay lock) is stable
  across the two builds we audited.
- Detector thresholds are fixed values validated cold on very different
  footage (the aerial desert sample at 60 fps, a synthetic exit-and-return
  harness, plus held-out clips covering indoor scenes and a different
  camera). They are counted in frames, which acts in the safer (slower to
  fire) direction at lower frame rates.