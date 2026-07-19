# Task summary

Answers to the six assignment questions.

## 1. Steps

1. Split the problem into three parts: frame-to-frame tracking, loss
   detection, re-acquisition. Decided to measure candidates, not trust
   reputation.
2. Benchmarked KCF, CSRT and MOSSE on real 1080p frames, checking both
   speed and whether the box stayed on target (ground truth from ORB
   matching). MOSSE-128 was the only one that held through the clip's pan,
   zoom and rotation. CSRT drifted off-target while reporting success.
3. Built the MVP: video input, pixel selection (mouse and CLI), box,
   trajectory and fps overlay, every backend behind one interface.
4. Wrote my own MOSSE. OpenCV's exposes no response map, no confidence and
   no template control, all of which loss detection needs. About 150 lines
   of FFT gave me PSR and a freeze switch.
5. Added loss detection (relative PSR with hysteresis) and re-acquisition
   (ORB + RANSAC, only while LOST). Built a synthetic forward-then-reversed
   clip to test recovery, since the target never returns in the original.
6. Hardened for real time: per-state profiling, single-thread cold-CPU
   proxy, and a re-detection cadence chosen so the worst state projects
   above 30 fps even on a CPU 3x slower than mine.
7. Tested cold on held-out clips (indoor, different camera), including a
   real exit-and-return at 1080p. Logged the failures.
8. Input validation, unit tests, cleanup.
9. Re-checked the sample clip frame by frame before publishing and found
   the tracker losing a visible target under a zoom-and-rotation burst,
   with re-acquisition recovering nothing. Rebuilt from a measured
   baseline: eleven tracker variants failed the same segment, so honest
   LOST is the right outcome there. The re-acquisition fix was storing the
   target with 2x context (a plain 128 px patch loses every match to the
   ratio test, 0 of 116, where context gives 27 inliers). Two safeguards came out
   of it: a sticky appearance memory and a motion guard for boxes stuck on
   burned-in overlays. Final: five venues, zero false losses, zero false
   re-acquisitions, 30 tests.

## 2. Principles

- Measure, don't assume. Every choice came from an experiment on real
  frames.
- Never show TRACKING when not tracking. PSR confidence, template freezing,
  verified re-acquisition, a UI that says LOST.
- Real time is a budget. 33 ms per 1080p frame, and the one expensive step
  runs only while tracking is idle.
- Small functions I can explain, including the FFT math.
- No overfitting. Nothing hardcoded to a clip, thresholds relative,
  validated cold on unseen footage.

## 3. Algorithms

- **KCF**: fast (~2 ms) but brittle here. Lost the target early under
  rotation and false-relocked. Rejected.
- **CSRT**: the accuracy favorite, fits the budget at ~72 fps, but drifted
  off-target while reporting success 300/300. Unusable without external
  confidence. Kept as a fallback backend.
- **MOSSE**: ~0.5 ms and the only tracker that held through the rotation.
  Chosen, and reimplemented by hand because OpenCV hides the response map
  and update control.
- **NCC template matching** for re-acquisition: fails under rotation (best
  score ~0.3 at the wrong place). Rejected for ORB.
- **ORB + ratio test + RANSAC**: rotation and scale tolerant, and the
  inlier count is real evidence. Three findings shaped it: search at full
  resolution (downscaling kills weak features), store 2x context (a window
  patch loses every match to the ratio test, 0 of 116, where context gives 27),
  and keep the memory sticky (first-loss appearance only), since a bad
  re-seed poisons everything after it.
- **Motion-based overlay guard**: burned-in HUD graphics are static in
  screen space while the scene flows. Comparing the box's motion to the
  scene-predicted motion at its position catches a lock no appearance
  confidence can.
- **Particle filter (Condensation)** I had from a course: considered for
  the core, but its main cue was a foreground mask this task doesn't have,
  and a colour histogram is near useless on desert footage. Would need a
  new likelihood and heavy vectorisation to hit real time, 1-2 days with
  real risk of still losing to MOSSE. Declined.

## 4. Biggest challenge

Honest loss detection, specifically template poisoning. An adaptive filter
learns every frame, so once the target leaves it learns the background. I
measured a poisoned template on a frame edge reporting PSR 2900: maximum
confidence, completely wrong. The fix had three parts. Freeze the template
at the first low-confidence frame. Detect loss from a relative PSR drop with
hysteresis, not a fixed threshold (the fixed one fired 34 frames late, the
relative one within 4, no false triggers). Store the re-acquisition patch
from before the collapse, since the last pre-freeze patch is a blurred,
half-off-frame sliver ORB cannot match.

A second failure needed a different sensor. A box glued to a burned-in HUD
overlay correlates perfectly forever, so confidence reads maximal (I
measured 40x normal) while it tracks nothing. Confidence cannot see this by
construction, so I used motion: a real target moves with the scene, an
overlay does not. That guard, the context patches and the sticky memory
made the system honest on the hard footage.

## 5. Where it fails

- Violent zoom-plus-rotation bursts defeat the whole correlation-tracker
  family. I measured eleven variants and CSRT failing the same burst. The
  answer is honest LOST plus fast re-acquisition, not heroic tracking.
- On scenes full of near-identical structures, re-acquisition can lock onto
  the wrong instance and track it confidently. Measured: 4 of 8 init pixels
  gave false or suspect recoveries, up to 925 px off. Neither safeguard
  catches it by construction. RANSAC proves geometric consistency and PSR
  proves a sharp lock, and a wrong-but-real structure satisfies both. Same
  class as the overlay lock. It needs a new sensor (a motion-plausibility
  gate on re-seeds), not a tighter threshold.
- No scale adaptation while tracking. A strong zoom degrades correlation
  until an honest LOST, then re-acquisition restores the box at current
  scale.
- Re-acquisition needs textured context and lands approximately under a
  large viewpoint change. A featureless target cannot be re-found (the app
  warns). The memory is sticky by choice: correctness over adaptivity.
- A re-acquisition attempt spikes to ~100 ms single-threaded while LOST.
  The every-8th-frame cadence keeps the mean above 30 fps, but a hard
  per-frame deadline would need an async search.
- Burned-in overlays are perfect correlation targets. The guard reverts a
  box stuck on one, but the initial pixel should be on the object, not on
  screen graphics.
- Re-acquisition frame indices vary across OpenCV builds (same version,
  different compilation). The behaviour is stable, the exact frames are not.

## 6. Personal experience

The best moment was near the end, running the finished version and watching
parts I had built separately work as one system. The worst came just before
it: a version that looked done, until I saw I had handled rotation and zoom
wrong and the tracker was dropping a target still in plain view. Fixing it,
once I understood why it failed, is the part I'm most proud of. My EE
background helped more than I expected. The signal-processing side, FFT
correlation, PSR, RANSAC, was familiar, and so was benchmarking and reading
up on algorithms. What was new was taking the theory all the way to a
working real-time program with a UI and tests, and doing it with an AI
coding agent I directed and checked against my own measurements. What I got
wrong at the start was assuming the first analysis had understood the
problem well enough. It hadn't, and going back to question it was the real
lesson.
