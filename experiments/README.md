# Experiments

Benchmark scripts and measured results (fps of candidate trackers on real
1920x1080 CPU frames in Phase B, and the real-time hardening numbers in Phase E).
Results recorded here feed the tables in the main README and the written summary.

**Phase legend.** The phase labels used below and across these logs refer to
the v1 build stages: A scaffold, B benchmark, C loss detection and template
poisoning, D re-acquisition design, E real-time hardening, F generalization,
G delivery. The later v2 remediation (the sample-clip failure and its fix) is
tracked separately as Stages 1 to 3 in `remediation/results.md`.

## Folder & version convention (remediation)

`src/` is the FROZEN v1 baseline, and remediation prototypes never modify it.
All candidate fixes live in `experiments/remediation/` as
`proto_NN_<candidate>.py`, each importing from `src/` and overriding only
the behavior it tests (subclass/wrap). Every prototype is measured with the
same instrument (`experiments/grip_baseline.py` metrics via
`experiments/remediation/harness.py`) against the same baseline
(451 frames held, 100/0 re-acquisition). Results accumulate in
`experiments/remediation/results.md`.

## Findings log (feeds the Phase G write-up)

- **Template poisoning (measured, Phase C):** with always-on adaptation,
  cv2-style MOSSE poisons onto background/edges after the target exits. On
  the sample clip it ended pinned to a static edge patch reporting
  PSR ≈ 2900. Freezing the template on low confidence eliminates this
  (PSR then stays honestly low, ~3-4, until re-acquisition).
- **Loss-detector tuning (Phase D, sample-clip loss event at ~f455):**
  relative-PSR detector (window=45, rel_drop=0.35, n_enter=3, n_exit=5,
  abs_healthy=20) → zero false triggers, detection at f452 vs f489 for a
  fixed PSR<8 threshold (~37 frames earlier). abs_healthy guards against
  the inflated post-init reference, and window=45 sets the regime-adaptation
  timescale (90 was too slow for the fast-pan regime change).
- **Re-acquisition design evidence (Phase D):** 0.5× downscale destroys
  the low-contrast aerial features (0-2 ORB matches vs 10 RANSAC inliers
  at full res). The last-learned patch is already degraded at freeze time
  (delayed appearance buffer fixes this). min_inliers=8 works with
  ~60-keypoint patches. Background re-seeds self-correct via the PSR
  detector within frames (wrong-instance re-seeds on repeated structures
  do NOT, see the wrong-instance limitation in the main README).
- **Phase F watch item, frame-unit parameters at 30 fps:** window /
  n_enter / n_exit / APPEARANCE_DELAY are counted in FRAMES and were tuned
  on a ~60 fps clip, so on 30 fps footage their timescales effectively
  halve. If they misfire there, the fix is normalizing to time units
  (seconds), not re-tuning constants to a second clip.
- **Generalization results (3 unseen 30 fps clips from a different domain,
  indoor and high-contrast, cold provisional parameters, nothing
  re-tuned):** (1) full exit-and-return success on the 1080p
  re-acquisition clip. LOST during the motion-blur swing-away, zero false
  re-locks while the camera visited unrelated scenery, correct RE-ACQUIRED
  on the designated target ~0.2 s after return, 130-145 fps e2e.
  (2) The relative-PSR thresholds transferred as-is (healthy PSR 30-60 on
  the new domain vs ~49 on the desert sample), and the frame-unit timescale
  halving at 30 fps caused NO premature triggers (it acts in the
  safe/slower direction).
  (3) Two real gaps found, both tracker-architecture class, documented as
  limitations: MOSSE loses grip under fast handheld pans (gradual PSR
  decay 48→15 while the target is still in frame, an honest LOST but a
  premature one), and single-template ORB re-acquisition fails under
  strong viewpoint change / weak texture (a featureless low-texture patch
  = 0 keypoints, so re-acquisition is structurally inactive, and
  blur-degraded low-texture patches = 0 RANSAC inliers at return). False
  re-locks: zero across ~2400 LOST frames in five runs, so the
  verification bar fails silent, never wrong.
- **KNOWN FAILURE MODE, no scale adaptation (documented, not fixed):**
  MOSSE tracks with a fixed-size window, so a strong zoom (e.g. the drone
  descending onto the building in the sample clip's second half) degrades
  the correlation until honest LOST. The system detects the loss and
  recovers via re-acquisition when the appearance matches again (observed
  on the synthetic harness: zoom-driven LOST at f1022, RE-ACQUIRED at
  f1046), but continuous scale tracking would need a scale-search layer or
  a CSRT fallback, deliberately deferred as candidate future work.
