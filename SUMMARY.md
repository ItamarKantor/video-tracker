# Task summary

Answers to the six questions from the assignment.

## 1. The steps I took

1. **Research before architecture.** I mapped the problem into three
   sub-problems (frame-to-frame tracking, loss detection, re-acquisition)
   and listed candidate algorithms for each, deciding to trust nothing
   without measuring it on real 1080p frames.
2. **Benchmark.** I measured KCF, CSRT and MOSSE on the sample clip at
   native resolution (speed AND whether the box actually stayed on the
   target, verified against ground truth I established with ORB matching).
   Surprise result: MOSSE-128 was the only tracker that held through the
   clip's pan/zoom/rotation, while CSRT drifted silently and still
   reported success.
3. **Working MVP.** Video input, pixel selection (mouse + CLI), tracking
   box, trajectory and fps overlay, with every backend behind one interface
   so the tracker stays swappable.
4. **Wrote my own MOSSE.** The OpenCV MOSSE exposes no response map, no
   confidence and no control over template adaptation, and all three are
   required for honest loss detection. Implementing Bolme's algorithm
   (~150 lines of FFT) gave me PSR confidence and a template-freeze switch.
5. **Loss detection + re-acquisition.** A relative-PSR detector with
   hysteresis (tuned on the real loss event in the sample clip), and an
   ORB+RANSAC searcher that runs only while LOST. I built a synthetic
   exit-and-return test clip (sample clip played forward then reversed) to
   develop against, because in the original clip the target never returns.
6. **Real-time hardening.** Profiled per state, pinned to one thread as a
   cold-CPU proxy, and set the re-detection cadence so the worst state
   holds ≥30 fps even on a CPU ~3× slower than my machine.
7. **Generalization test.** Ran everything cold (no re-tuning) on
   held-out clips not included in the repo, covering indoor scenes and a
   different camera, including a real exit-and-return scene at 1080p
   where it detected the loss and re-acquired correctly. Logged the
   failures honestly (see question 5).
8. **Review, robustness and tests.** Input validation with clean errors,
   deterministic unit tests, and a final pass for readability.
9. **A hard second look at the sample clip, and a remediation round.**
   Before publishing I re-verified the sample clip frame by frame and
   found the tracker losing the target while it was still visible (a
   violent zoom+rotation burst) and re-acquisition recovering nothing. I
   rebuilt the evidence from a measured baseline: eleven tracker variants
   failed the same segment (so honest LOST is the right outcome there),
   and the re-acquisition fix turned out to be storing the target WITH 2x
   spatial context (repetitive texture kills window-scale matching, so the
   ratio test rejected 116/116 matches, while context-scale matching
   verified 27 inliers). Two safeguards came out of the same investigation:
   a sticky appearance memory (only the pre-loss appearance is matched) and
   a motion-based guard that catches a box glued to burned-in screen
   overlays, which confidence measures structurally cannot see. Final
   validation covered five venues in closed loop, with zero false losses
   and zero false re-acquisitions, backed by 30 unit tests.

## 2. Guiding principles

- **Measure, don't assume.** Every major choice (tracker, box size, search
  resolution, detection cadence) was decided by an experiment on real
  frames, and the numbers are in `experiments/bench_results.md`.
- **Honest state over pretty state.** The system must never show TRACKING
  while it isn't tracking. The entire design flows from this: PSR
  confidence, template freezing, verified re-acquisition, and a UI that
  says LOST loudly.
- **Real-time is a budget, not a hope.** ~33 ms per 1080p frame. Every
  component was measured against it, and the only expensive one
  (re-detection) is amortized and runs only when tracking is idle.
- **Small explainable functions.** I can walk through every line,
  including the FFT math.
- **No overfitting to the sample clip.** Nothing is hardcoded to any
  video, and thresholds are relative where possible and were validated
  cold on unseen footage.

## 3. Algorithms researched and conclusions

- **KCF**: fast (~2 ms/frame) but brittle here. It reported loss almost
  immediately under the sample clip's rotation, and "re-locked" falsely
  after failures. Rejected.
- **CSRT**: the literature's accuracy favorite. At ~72 fps end-to-end it
  fits the budget, but it drifted silently off-target while reporting
  success 300/300, which makes it unusable without an external confidence
  measure. Kept as a documented fallback backend.
- **MOSSE**: ~0.5 ms/frame, and the only one that held through
  pan/zoom/rotation (its fast-adapting plain-intensity template tolerates
  gradual appearance change better than HOG-based filters). Chosen, but
  reimplemented by hand, because the decisive features (PSR, update
  control) aren't exposed by OpenCV's binding.
- **Template matching (NCC) for re-acquisition**: fails under rotation
  (best score ~0.3 at a wrong location on rotated frames). Rejected in
  favor of ORB.
- **ORB + ratio test + RANSAC homography**: rotation/scale tolerant and
  verifiable (inlier count is a real evidence measure). Three empirical
  findings shaped the final design. First, matching must run at full
  resolution, because downscaling kills weak features on low-contrast
  footage. Second, the stored patch must include 2x spatial CONTEXT: on
  repetitive texture a window-sized patch loses every match to the ratio
  test (0/116), while the context patch verifies 27 inliers. Third, the
  matching memory must be sticky (only the pre-loss appearance), because an
  imperfect re-seed poisons everything captured after it.
- **A motion-based static-overlay guard**: burned-in HUD graphics are
  static in screen space while real scenery flows, and a correlation
  tracker glued to them reports perfect confidence forever. Comparing the
  box's motion to the scene-predicted motion at its own position (from a
  per-frame similarity estimate) catches this failure class that no
  appearance confidence can.
- **A particle filter (Condensation) I had built previously** for a
  university project: considered as the tracking core (native scale
  handling, built-in confidence), but its strongest measurement cue there
  was a foreground mask that doesn't exist in this task. On desert footage
  a color-histogram likelihood is nearly information-free, so it would
  need a new gradient-based likelihood plus heavy vectorization to reach
  real time. Estimated 1-2 days with real risk of still underperforming
  MOSSE, so I declined it and the effort went into the custom MOSSE
  instead.

## 4. The biggest challenge and how I overcame it

**Making loss detection honest**, specifically defeating template
poisoning. An adaptive correlation filter keeps learning every frame, so
when the target leaves, the filter learns the background. I measured a
poisoned template stuck on a frame edge reporting PSR ≈ 2900, with the
tracker maximally confident and completely wrong. Fixing this took three
linked pieces: freeze the template at the first sign of trouble (cheap and
reversible, unlike poisoning), detect loss from a *relative* PSR drop with
hysteresis (a fixed threshold fired ~34 frames late on the real loss
event, while the relative detector fires within ~4 frames with zero false
triggers), and store the appearance for re-acquisition from *before* the
degradation. The naive "last patch" was a blurred, half-out-of-frame
sliver that could never be re-found, so the tracker keeps a quality-gated
buffer and searches for the sharpest recent patch instead.

The challenge had a second act: burned-in HUD overlays are the one place
confidence can never be trusted. Static screen graphics correlate
perfectly forever, so a box glued to them looks *maximally healthy* (I
measured confidence 40x the normal level on one). The answer had to come
from a different physical signal entirely: motion. A genuine target moves
with the scene, while screen furniture does not. That guard, plus
context-scale matching and the sticky appearance memory, is what finally
made the system honest end to end on the hardest footage.

## 5. Where the software is limited / fails

- Violent appearance transients (simultaneous zoom + fast rotation) defeat
  the whole correlation-tracker family. Measured: eleven tracker variants
  and CSRT all failed the same burst. The system answers with honest LOST
  and fast re-acquisition, not heroic tracking.
- On scenes containing repeated near-identical structures, re-acquisition
  can lock onto the WRONG instance and then track it stably: measured on
  the sample clip, 4 of 8 tested init pixels produced false or suspect
  recoveries (up to ~925 px off the true target). This failure is
  invisible to the existing safeguards by construction. The RANSAC gate
  proves a match is geometrically self-consistent and PSR proves the lock
  is sharp, yet a wrong-yet-real structure satisfies both perfectly, and
  neither signal can express "this is the wrong instance". It is the same
  class of error as the burned-in-overlay lock: it needs a different
  sensor (a motion-plausibility gate on re-seeds is the proposed future
  fix), not a better threshold.
- No scale adaptation while tracking: a strong zoom degrades correlation
  until an honest LOST, after which re-acquisition restores the box at the
  target's current scale.
- Re-acquisition requires a textured surrounding context (featureless
  surroundings cannot be re-found, and the app warns explicitly), and
  under a very different viewpoint it recovers the target's content region
  approximately, not pixel-perfectly. The appearance memory is
  deliberately sticky (pre-loss appearance only), choosing correctness
  over adaptivity.
- While LOST, a context-matching attempt spikes to ~100 ms single-threaded.
  The every-8th-frame cadence keeps mean throughput ≥30 fps, but a hard
  per-frame deadline would need an asynchronous search.
- Burned-in HUD overlays are perfect correlation targets. The motion guard
  detects and reverts a box glued to them, but the initial pixel should
  still be chosen on the object itself, not on screen graphics.
- Exact re-acquisition frame indices vary across OpenCV builds (same
  version, different compilation) because marginal matches near the
  acceptance threshold resolve differently. The qualitative behavior is
  stable across the builds we audited.

## 6. Personal experience

The most satisfying moment came near the end, when I played with the inputs
and outputs of the finished version and watched pieces I had built
separately actually connect into one working system. The hardest moment
came just before it. I had a version that looked done, and then saw I had
handled rotation and zoom wrong: the tracker was dropping a target that
was still clearly visible. That was uncomfortable to admit, but working out
*why* it failed, and finding a fix that held (re-acquiring the target from
a larger patch of surrounding context), turned into the part I'm most proud
of. My electrical-engineering background helped more than I expected. The
signal-processing side (the FFT correlation, the PSR confidence, the
RANSAC geometry) felt like familiar ground, and benchmarking and digging
into algorithms was something I was used to. What was new was carrying that
theory all the way into a complete real-time program with a UI and its own
tests, and doing it alongside an AI coding agent that I directed and
corrected while checking every result against my own measurements. The
assumption I had to give up was that our first analysis had understood the
problem deeply enough. It hadn't, and being made to go back and question it
properly was the real lesson of the project.
