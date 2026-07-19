# Remediation Stage 2: measured comparison

Baseline (frozen v1, `experiments/grip_baseline.py`, ASIO sample clip,
target i=857 j=1167): **held 451 frames (LOST@f452), re-acquisition
100 attempts / 0 accepted, 190 fps incl. decode.** Ground truth: target
fully visible at LOST, visible (mid-spin) until the true exit ~f550-570,
never returns. Camera during the segment: 15-21 px/frame translation,
+0.31 %/frame zoom, 0.8-1.5°/frame rotation (accumulating to +128° and
1.5× by f540). Frames are SHARP (Laplacian var 374-396 vs 294 at f300).

## Tracker levers (goal: hold past 451)

| candidate | held | transitions | fps (process) | verdict |
|---|---|---|---|---|
| baseline v1 | 451 | LOST@452 | 190 (e2e) | reference |
| ① motion predict | 449 | LOST@450 | 269 | no effect |
| ② padded window ×1.5 | 450 | LOST@451 | 242 | no effect |
| ③ scale search {0.95,1,1.05} | 451 | LOST@452 | 231 | no effect |
| ⑦ η=0.25 | 453 | LOST@454 | 274 | +2 frames, negligible |
| ⑦ η=0.4 | 454 | LOST@455 | 273 | +3 frames, negligible |
| ⑧ combo ①+③ | 447 | LOST@448 | 234 | no effect |
| ⑧ combo ①+②+③ (pad 1.5) | 449 | junk re-locks | **15, R4 FAIL** | degenerate, out |
| ⑧ combo ①+③, η=0.25 | 448 | LOST@449 | 226 | no effect |
| ⑧ combo ①+③, η=0.4 | 453 | LOST@454 | 235 | no effect |
| (C) cv2 CSRT | n/a (no LOST signal) | silent drift: ~500 px off by f300, box degenerates to 22 px, 0 reported failures | 87 | disqualified |

**Conclusion:** the grip loss at ~f452 is caused by an appearance transient
(simultaneous zoom + fast rotation burst) that no search-geometry or
adaptation-rate lever can ride out. Every correlation variant and CSRT
fail the same segment. Honest LOST at ~452 is effectively the achievable
outcome for the tracking layer, so remediation must come from
re-acquisition.

## Re-acquisition levers (goal: verified recovery while the target is visible, no false locks)

Root-cause measurements first:

- v1 (single 128 px stored patch): Lowe ratio test annihilated, median
  best/second distance ratio **0.97**, ratio survivors **0/116** even with
  a sharp patch. The 128 px window is locally repetitive (near-identical
  pen walls).
- Motion-compensated NCC probe (integrate global affine while LOST, warp
  the patch, search near the propagated position): dead-reckoning drifts
  badly over 50+ frames (predicted position leaves the frame while the
  target is mid-frame), best NCC 0.09-0.16 = noise. Rejected.
- **Context-patch finding: storing the same target with 2× spatial context
  (256 px) defeats the repetitiveness**, 27 RANSAC inliers at f300 (vs 7
  with 128 px). Sweep of the LOST window: 9-13 inliers at f456-476 with
  hit centers moving consistently with the compound through f540, and
  **zero hits ≥5 inliers after the true exit** (no HUD/roof false
  matches). Verification honesty: region-level correctness was confirmed
  visually (box on the compound's wall band) plus by trajectory
  consistency. Per-pen identity is NOT establishable on this repetitive
  texture, so an independent geometric/appearance check on one re-seed was
  accordingly inconclusive (183 px predicted-position error, 0 appearance
  inliers), consistent with region-correct but pen-ambiguous recovery.

| candidate | recovery on ASIO clip | false locks | fps | verdict |
|---|---|---|---|---|
| v1 single patch | 0/100 | none | ok | dead (ratio annihilation) |
| ④ gallery (128 px patches) | 0/100 | none | 220 | dead (same funnel) |
| ⑤ gallery × multi-scale variants | 1 hit | **FALSE, locked onto static HUD overlay, conf 328, to end of clip** | 229 | worse than baseline |
| ⑨ context patches (2×) + inner-box re-seed | **RE-ACQUIRED@f456 (4 frames after LOST, 13 inliers)**, held-out clip A improves (f79 vs f83), harness recovers @f459 | **one remaining path:** a genuine accept at f515 near the frame corner later POISONS onto the static HUD during the RE-ACQUIRED probation (box frozen at fixed coords, conf 70-108 to clip end) | 450 mean, heaviest frames ~41 ms | **winner, needs the static-guard companion** |

Accept-time static veto (region motion vs global motion) was implemented
and does NOT catch the remaining path, because the f515 acceptance itself
is on moving ground. The HUD lock develops ~20 frames later as the scene
sweeps out from under the re-seeded box. A static-vs-global-motion check
DURING tracking/probation is required (state-machine adjacent → approval
needed).

## Recommended combination for Stage 3

1. **Integrate context-patch re-acquisition** (proto_09 core: 2× context
   snapshots, init + quality-gated refresh, ≤3 entries. The frame is
   ORB-detected once, the inner target box is mapped through the verified
   homography, and min_inliers is unchanged at 8).
2. **Add a static-scenery guard**: "box has near-zero temporal motion while
   the frame moves" ⇒ not our target. At accept time (cheap, already
   prototyped) AND during the RE-ACQUIRED probation (the actual fix for
   the f515→HUD chain, which requires a small, approved extension beside
   the PSR verdict in the state machine).
3. **Keep the v1 MOSSE tracker unchanged**, since all grip levers measured
   ineffective here. Recovery-in-4-frames makes grip extension moot, and
   simplicity preserves the validated behavior on every other clip.

Expected ASIO end state: track to ~f451 → honest LOST → RE-ACQUIRED f456 →
honest LOST/RE-ACQUIRED cycling through the spin while the target is
visible → final honest LOST at the true exit (~f550-570), no false locks
to the end. R4 headroom preserved (heaviest re-acq frame ~41 ms, amortized
under REACQ_EVERY_N=4 as before).

---

# Stage 3: shipped configuration validation (v2, REACQ_EVERY_N=8)

Integrated: context-patch re-acquisition (sticky pristine entries) +
affine-referenced StaticGuard (always on). Cadence N=8 ratified after
measuring the context attempt at ~103 ms single-threaded (amortized
10.7 ms measured → ~31 fps LOST-state even on a 3x-slower CPU).

Cross-build reproducibility (two audit builds, both OpenCV 4.13.0, one
opencv-contrib-python==4.13.0.92, one base opencv-python, i.e. a
packaging/compilation difference, not a version difference): at N=8 the
sample-clip, held-out clip A and held-out clip B timelines reproduce
identically, while the harness and held-out clip C drift 8-11 frames in
re-acquisition indices (RA@650 vs 642, RA@836 vs 825, and clip C's
genuine-return recovery lands on BOTH builds). At N=4 the builds disagreed
on whole marginal accepts (ASIO: 3 vs 2 recoveries).
The state machine's qualitative behavior (honest LOST on the tail, genuine
recovery when matchable, no HUD/roof lock) is stable across builds, but
exact re-acquisition frame indices are not, because marginal-inlier
candidates near the acceptance threshold resolve differently under
different OpenCV compilations. N=8 reduces, but does not eliminate, this
sensitivity by dropping the most marginal accepts.

## Reproduction: init pixels (i,j on frame 0), public venues

| venue | file | pixel i,j |
|---|---|---|
| ASIO sample | the provided sample clip (path is a runtime argument) | 857,1167 |
| synthetic harness | input/synthetic_loop_DEV.mp4 (built from the sample clip by make_synthetic_loop.py) | 857,1167 |

Three additional held-out clips (personal footage, not published) were
used for generalization testing, and their timelines are summarized below
without identifying detail.

## Five-venue closed-loop timelines at N=8

- ASIO: LOST@452 → RA@460 → LOST@469 → honest LOST to end (verified
  frame-by-frame: f515/529/545/600/700/850 all boxless honest LOST, no
  HUD/roof lock, the conf-4838 pathology is gone).
- harness: LOST@455 → RA@463 → LOST@473 → RA@521 → LOST@530 → RA@650 →
  TRACKING@695 → LOST@1022 → RA@1062 → TRACKING@1107.
- held-out clip A (indoor, 1080p, genuine exit-and-return):
  LOST@15 → RA@79 → TRACKING@124 (tracks to end).
- held-out clip B (indoor pan): LOST@39 → RA@47 → TRACKING@92 →
  honest LOST@270.
- held-out clip C (indoor pan): LOST@186 → RA@202 → LOST@236 → RA@836,
  the genuine indoor return IS recovered at the shipped cadence.

Zero false LOSTs and zero false re-acquisitions on all venues. Costs at
1080p (default threads): TRACKING ~3 ms/frame (guard ~2 ms of it), LOST
amortized ~4.8 ms, worst attempt frame ~103 ms single-threaded.

## R5 generality: six targets on the sample clip, zero per-target tuning

The primary target plus five additional targets of deliberately different
character, all run with identical code and constants. Every run tracks its
target and ends in honest LOST (the camera ultimately leaves every ground
feature behind), and the weak-texture desert target cycles with early
honest losses rather than faking confidence.

| target (i,j) | character | measured timeline |
|---|---|---|
| 857,1167 | structure corner (primary) | held 451 → RA@460 → LOST@469 → honest LOST to end |
| 700,1345 | vegetation cluster | held 437 → RA@446 → RA@466 → RA@483 → honest LOST@504 |
| 990,320 | settlement structures (leave view early) | LOST@45 → RA@181 → RA@216 → honest LOST@227 |
| 980,700 | open ground with tracks | LOST@48 → RA@56 → TRACKING@101 → LOST@341 → RA@397 → TRACKING@442 → honest LOST@506 |
| 800,1000 | alternate structure corner | held 533 → RA@542 → honest LOST@577 |
| 300,1500 | near-featureless desert | weak-texture cycling (LOST@57 → RA@65 → … ) with honest states throughout, final LOST@334 |

An independent audit measured five equivalent target categories on the
same clip with the same qualitative outcome.

## Wrong-instance re-acquisition (audit finding, owned limitation)

Recovery-identity verification protocol (validated on a known-good
recovery, where it returned 11 px vs the homography's 3 px): take the
pipeline's box center ~30 frames before the confirmed LOST, LK-track that
world point frame-by-frame (winSize 31, maxLevel 4, discarding any
status-0 result) to ~8 frames after RE-ACQUIRED, and compare with the
pipeline's recovered center, then cross-check with an ORB+RANSAC homography
between the same two frames. On a 128 px box: under ~70 px is genuine,
over ~130 px is a different object.

Applying it across eight init pixels on the sample clip: **four produced
false or suspect recoveries**, where the box re-appeared on a different but
near-identical structure and then tracked it stably and confidently.
Worst cases: (401,970) recovered 371 px off (455 px by homography, 82
appearance inliers) then held confident TRACKING for 376 frames.
(529,967) 377 px off, accepted at exactly min_inliers=8. (780,980)
909 px off with 50 inliers. (652,1425) 925 px. The demo target
(750,1259) was verified genuine on two builds: 42-46 px, 44 px homography
agreement, 191 appearance inliers (256-context measure).

Why the existing safeguards cannot see this: RANSAC proves a match is
geometrically self-consistent, PSR proves the lock is sharp, and the
static guard proves the box moves with the scene. A wrong-yet-real
structure satisfies all three perfectly, so no signal in the system can
express "this is the wrong instance". Same error class as the
burned-in-overlay lock: a sensor blind to the failure mode by
construction, requiring a different sensor rather than a better threshold.

### Proposed future fix (recorded, deliberately not built)

A motion-plausibility gate on re-seeds: StaticGuard already estimates the
global affine transform every frame, so integrating it across the LOST
stretch predicts the region where the target should reappear, and a
candidate landing implausibly far outside that region is rejected or
demoted. Honest caveat: the integrated prediction drifts over long LOST
stretches (measured earlier in this file: dead-reckoning left the frame
entirely within ~50 frames on the spin segment), so the gate needs a
tolerance that widens with LOST duration, a design-and-measure exercise of
its own, which is why it is future work rather than a late change to the
frozen v2.
