"""Our own MOSSE correlation tracker (Bolme et al., CVPR 2010).

Why our own implementation instead of cv2.legacy.TrackerMOSSE: the OpenCV
Python API exposes only init/update - no response map, no PSR, no control
over template adaptation. The loss detector needs exactly those:
PSR as the confidence signal, and the ability to FREEZE template updates
when confidence drops so the filter is not poisoned by background just as
the target leaves the frame.

The algorithm, in short:
  * The filter H is defined in the Fourier domain so that correlating it
    with the target patch F yields a narrow Gaussian peak G at the target
    center:  F (*) h ~ g.
  * Closed-form ridge-regression solution, kept as separate numerator /
    denominator for cheap online updates:
        H* = A / (B + eps),  A = G . conj(F),  B = F . conj(F)
    (all products element-wise in the Fourier domain).
  * Per frame: crop the search window at the last position, preprocess,
    correlate:  r = irfft2( rfft2(f) . H* ).  The response peak gives the
    displacement; PSR grades how trustworthy that peak is.
  * Online update (only while not frozen), exponential moving average with
    learning rate eta:  A <- eta*G.conj(F) + (1-eta)*A   (same for B).

PSR (peak-to-sidelobe ratio) = (peak - mean(sidelobe)) / std(sidelobe),
where the sidelobe is the response map excluding an 11x11 window around the
peak. Bolme's reference values: healthy tracking typically 20+, and PSR
falling under ~7 indicates occlusion / loss - the loss detector chooses its
thresholds empirically (see loss_detect.py).
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .tracker import BaseTracker, Box

EPS = 1e-5


def _preprocess(patch: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Bolme's preprocessing: log -> zero-mean/unit-var -> cosine window.

    The log transform compresses lighting variation; normalization removes
    gain/offset; the Hann window kills the artificial edges the FFT's
    circular boundary would otherwise inject into the spectrum.
    """
    p = np.log1p(patch.astype(np.float32))
    p = (p - p.mean()) / (p.std() + EPS)
    return p * window


def _gaussian_peak(w: int, h: int, sigma: float = 2.0) -> np.ndarray:
    """Desired correlation output: Gaussian with the peak at the window center."""
    ys, xs = np.mgrid[0:h, 0:w]
    g = np.exp(-((xs - w // 2) ** 2 + (ys - h // 2) ** 2) / (2.0 * sigma ** 2))
    return g.astype(np.float32)


def _random_small_warp(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small random rotation/scale around the patch center (training augmentation).

    Bolme initializes the filter from ~8 perturbed copies of the first patch
    so it tolerates small pose changes instead of overfitting to one frame.
    """
    h, w = patch.shape
    angle = rng.uniform(-5.0, 5.0)          # degrees
    scale = rng.uniform(0.95, 1.05)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(patch, m, (w, h), borderMode=cv2.BORDER_REFLECT)


class MosseTracker(BaseTracker):
    """MOSSE with PSR confidence and a controllable template-update policy.

    Attributes the loss-detection / re-acquisition layer relies on:
      frozen          -- while True, update() still tracks but never adapts
                         the filter (protects the template near a loss).
      last_good_patch -- BGR crop of the target window from the HIGHEST-PSR
                         entry among the last APPEARANCE_DELAY learned
                         frames. The sliding window dodges the pre-loss
                         frames (measured on the sample clip: PSR slid
                         38 -> 13 while the box ran off the frame edge, so
                         the final learned patch was a letterbox-
                         contaminated sliver ORB could never re-find), and
                         the quality gate inside the window prefers sharp
                         frames over motion-blurred ones, which score
                         visibly lower PSR (measured on handheld footage).
                         The init patch enters with +inf quality: the
                         user-chosen view is trusted until APPEARANCE_DELAY
                         real measurements displace it.
      reacq_contexts  -- what re-acquisition actually matches: CONTEXT
                         crops (CONTEXT_FACTOR x the window, target box
                         recorded inside). Measured on the aerial sample:
                         a window-sized patch is locally repetitive and the
                         ratio test annihilates every match (0/116), while
                         the 2x context captures unique larger structure
                         (27 vs 7 RANSAC inliers). Returns the best recent
                         context plus the never-evicted init-view context.
    """

    name = "mosse"
    provides_confidence = True  # PSR is a real graded signal

    APPEARANCE_DELAY = 30  # learned frames kept; oldest is the re-acq patch
    CONTEXT_FACTOR = 2.0   # context side / window side for re-acq patches

    def __init__(self, learn_rate: float = 0.125, psr_exclude: int = 11,
                 n_train_warps: int = 8, seed: int = 0):
        self._eta = learn_rate
        self._exclude = psr_exclude
        self._n_warps = n_train_warps
        self._rng = np.random.default_rng(seed)
        self.frozen = False
        # (quality, context_patch, inner_box) triples; quality = the PSR of
        # the frame the patch was learned from, inner_box = the target
        # window's (x, y, w, h) inside the context crop
        self._patches: deque[tuple[float, np.ndarray, Box]] = deque(
            maxlen=self.APPEARANCE_DELAY)
        self._init_context: tuple[np.ndarray, Box] | None = None

    def _best_entry(self):
        return max(self._patches, key=lambda e: e[0]) if self._patches else None

    @property
    def last_good_patch(self) -> np.ndarray | None:
        """Target-window crop of the best buffered entry (inner region)."""
        entry = self._best_entry()
        if entry is None:
            return None
        _, ctx, (ix, iy, iw, ih) = entry
        return ctx[iy:iy + ih, ix:ix + iw]

    @property
    def reacq_contexts(self) -> list[tuple[np.ndarray, Box]]:
        """(context_patch, inner_box) entries for re-acquisition: the best
        recent appearance plus the permanent init view."""
        out = []
        entry = self._best_entry()
        if entry is not None:
            out.append((entry[1], entry[2]))
        if self._init_context is not None and (
                not out or self._init_context[0] is not out[0][0]):
            out.append(self._init_context)
        return out

    # -- geometry helpers ----------------------------------------------------

    def _clamp_center(self, frame_shape) -> None:
        """Keep the search window fully inside the frame."""
        fh, fw = frame_shape[:2]
        w, h = self._size
        self._cx = float(np.clip(self._cx, w / 2, fw - w / 2))
        self._cy = float(np.clip(self._cy, h / 2, fh - h / 2))

    def _crop_gray(self, frame) -> np.ndarray:
        """Grayscale search window at the current center (crop first: a full
        1080p cvtColor would cost more than the whole tracking step)."""
        w, h = self._size
        x0 = int(round(self._cx - w / 2))
        y0 = int(round(self._cy - h / 2))
        patch = frame[y0:y0 + h, x0:x0 + w]
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    def _box(self) -> Box:
        w, h = self._size
        return (int(round(self._cx - w / 2)), int(round(self._cy - h / 2)), w, h)

    # -- BaseTracker interface -----------------------------------------------

    def init(self, frame, box: Box) -> None:
        x, y, w, h = box
        self._size = (w, h)
        self._cx, self._cy = x + w / 2, y + h / 2
        self._clamp_center(frame.shape)
        self._window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        self._G = np.fft.rfft2(_gaussian_peak(w, h))
        self.frozen = False

        gray = self._crop_gray(frame)
        self._A = np.zeros_like(self._G)
        self._B = np.zeros(self._G.shape, dtype=np.float64)
        for i in range(self._n_warps + 1):  # identity + n random warps
            g = gray if i == 0 else _random_small_warp(gray, self._rng)
            f = np.fft.rfft2(_preprocess(g, self._window))
            self._A += self._G * np.conj(f)
            self._B += (f * np.conj(f)).real
        self._A /= (self._n_warps + 1)
        self._B /= (self._n_warps + 1)
        self._patches.clear()
        self._init_context = None
        self._remember_appearance(frame, float("inf"))
        entry = self._best_entry()
        self._init_context = (entry[1].copy(), entry[2])

    def update(self, frame) -> tuple[Box | None, float]:
        """Track into `frame` -> (box, PSR). Never returns None: MOSSE always
        has an argmax - deciding 'lost' from the PSR is the loss detector's
        job."""
        w, h = self._size
        f = np.fft.rfft2(_preprocess(self._crop_gray(frame), self._window))
        response = np.fft.irfft2(f * self._A / (self._B + EPS), s=(h, w))

        py, px = np.unravel_index(np.argmax(response), response.shape)
        psr = self._psr(response, px, py)
        self._cx += px - w // 2
        self._cy += py - h // 2
        self._clamp_center(frame.shape)

        if not self.frozen:
            f2 = np.fft.rfft2(_preprocess(self._crop_gray(frame), self._window))
            self._A = self._eta * self._G * np.conj(f2) + (1 - self._eta) * self._A
            self._B = self._eta * (f2 * np.conj(f2)).real + (1 - self._eta) * self._B
            self._remember_appearance(frame, float(psr))
        return self._box(), float(psr)

    # -- confidence / appearance ----------------------------------------------

    def _psr(self, response: np.ndarray, px: int, py: int) -> float:
        """Peak-to-sidelobe ratio of the correlation response.

        A sharp, isolated peak (high PSR) means the filter found one clearly
        best alignment; a flat/multi-modal response (low PSR) means it is
        guessing - the signature of occlusion or a vanished target.
        """
        peak = response[py, px]
        half = self._exclude // 2
        side = response.copy()
        side[max(py - half, 0):py + half + 1, max(px - half, 0):px + half + 1] = np.nan
        side = side[~np.isnan(side)]
        if side.size == 0:  # degenerate: window barely larger than the exclusion
            return 0.0
        return (peak - side.mean()) / (side.std() + EPS)

    def _remember_appearance(self, frame, quality: float) -> None:
        """Buffer the CONTEXT crop around the window the filter just learned
        from, tagged with the frame's PSR (re-acquisition uses the highest-
        quality entry - see reacq_contexts)."""
        w, h = self._size
        fh, fw = frame.shape[:2]
        cw = min(int(round(w * self.CONTEXT_FACTOR)), fw)
        ch = min(int(round(h * self.CONTEXT_FACTOR)), fh)
        x0 = int(round(min(max(self._cx - cw / 2, 0), fw - cw)))
        y0 = int(round(min(max(self._cy - ch / 2, 0), fh - ch)))
        wx = int(round(self._cx - w / 2))
        wy = int(round(self._cy - h / 2))
        inner = (min(max(wx - x0, 0), cw - w), min(max(wy - y0, 0), ch - h),
                 w, h)
        self._patches.append(
            (quality, frame[y0:y0 + ch, x0:x0 + cw].copy(), inner))
