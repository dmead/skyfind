"""Focus by star size, for a camera that records its own clip (the phone).

Autofocus has nothing to hold on a dark sky, and a phone lens's "infinity"
is a number, not a stop (its focus scale is uncalibrated, the lens sags when
pointed up and shifts with temperature). So the lens is stepped through the
far end of its travel, the stars measured at each stop, and the sharpest
position taken, as an astronomer's autofocuser does:

- `measure`: the stars' half-flux radius (HFR: the flux-weighted mean
  distance from each star's centre, the median over stars) and their
  sharpness (the share of a star's light in its brightest pixel).
- `Sweep`: which positions to visit and what they showed. A coarse pass
  over the range fits the V-curve (a hyperbola: HFR^2 is quadratic in the
  lens position); a fine pass about its minimum takes the peak of the
  sharpness, because a phone's stars are a pixel or two wide at best and
  their HFR goes flat there. The lens is last moved to the best position
  from the same side the passes came (voice coils have hysteresis).
- `drifted`: during the recording, whether the stars have grown since.

Positions are whatever the lens takes (Android: dioptres, 0 = infinity as
the maker has it); only their order matters here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

RADIUS = 12                 # px: a star's window (and the largest blur seen)
MIN_STARS = 5               # fewer measured: the stop says nothing
COARSE_STEPS = 13
FINE_STEPS = 9
DRIFT = 1.3                 # stars this much larger than at focus: refocus


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


def measure(frame: np.ndarray, *, max_stars: int = 60, k_sigma: float = 5.0,
            radius: int = RADIUS) -> dict:
    """The stars' size in one frame (grey or BGR, 8-bit levels): dict(hfr,
    sharp, n, flux). `hfr` px and `sharp` (0..1) are medians over the `n`
    brightest unsaturated stars; None with fewer than `MIN_STARS`.

    Stars are found in a smoothed copy (a defocused star is faint per
    pixel: its disc stands out only summed), then each is measured in the
    frame itself over its window, less the window's edge median."""
    g = _gray(frame)
    h, w = g.shape
    # the sky's gradient out, at a scale wider than any blur disc
    small = cv2.resize(g, (max(1, w // 8), max(1, h // 8)),
                       interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), 7)
                    .astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    d = g - bg
    sm = cv2.GaussianBlur(d, (0, 0), 2.0)
    sub = sm[::4, ::4]
    med = float(np.median(sub))
    sig = 1.4826 * float(np.median(np.abs(sub - med))) or 1.0
    peaks = (sm > med + k_sigma * sig) & \
        (sm >= cv2.dilate(sm, np.ones((2 * radius + 1,) * 2, np.uint8)))
    ys, xs = np.nonzero(peaks)
    keep = (xs >= radius) & (xs < w - radius) & (ys >= radius) & \
        (ys < h - radius)
    xs, ys = xs[keep], ys[keep]
    order = np.argsort(-sm[ys, xs])[:max_stars * 3]

    oy, ox = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rim = np.hypot(ox, oy) >= radius - 1.5
    # a star's own pixels: where the smoothed copy stands clear of the sky
    # (the whole window's noise would pull the HFR out to its edge)
    lit = sm > med + 3 * sig
    hfr, sharp, flux = [], [], []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        win = (slice(y - radius, y + radius + 1),
               slice(x - radius, x + radius + 1))
        raw = g[win]
        if raw.max() >= 250:                   # saturated: flat-topped
            continue
        p = np.where(lit[win], raw - float(np.median(raw[rim])), 0.0)
        f = float(p.sum())
        if f <= 0:
            continue
        cx, cy = float((p * ox).sum() / f), float((p * oy).sum() / f)
        if abs(cx) > 2.5 or abs(cy) > 2.5:     # a neighbour, or a trail
            continue
        hfr.append(float((p * np.hypot(ox - cx, oy - cy)).sum() / f))
        sharp.append(float(p.max() / f))
        flux.append(f)
        if len(hfr) >= max_stars:
            break
    if len(hfr) < MIN_STARS:
        return dict(hfr=None, sharp=None, n=len(hfr), flux=None)
    return dict(hfr=float(np.median(hfr)), sharp=float(np.median(sharp)),
                n=len(hfr), flux=float(np.median(flux)))


def v_curve(x, hfr) -> tuple[float, float] | None:
    """The minimum of the hyperbola HFR^2 = a (x - x0)^2 + c through the
    stops (x, hfr): (x0, HFR there), or None when they don't make a V (no
    curvature: the range never passed through focus)."""
    x, y = np.asarray(x, float), np.asarray(hfr, float) ** 2
    if len(x) < 3:
        return None
    s = float(np.ptp(x)) or 1.0
    u = (x - x.mean()) / s
    a, b, c = np.polyfit(u, y, 2)
    if a <= 0:
        return None
    u0 = -b / (2 * a)
    if abs(u0) > 1.0:                 # a slope's curvature, not a V
        return None
    return float(u0 * s + x.mean()), float(np.sqrt(max(c - b * b / (4 * a),
                                                       0.0)))


def _peak(x, y) -> float | None:
    """Where a parabola through (x, y) is highest, when it has a top inside
    the stops."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    s = float(np.ptp(x)) or 1.0
    u = (x - x.mean()) / s
    a, b, _ = np.polyfit(u, y, 2)
    if a >= 0:
        return None
    x0 = -b / (2 * a) * s + x.mean()
    return float(x0) if x.min() <= x0 <= x.max() else None


@dataclass
class Sweep:
    """An autofocus run over lens positions `lo`..`hi` (`lo` the far end:
    Android's 0). The caller moves the lens to `next()`, waits for it to
    settle, hands a frame (or several) to `add`, and repeats until `next()`
    is None; `best` is then where the lens stays (`next()`'s last two
    positions put it there from the `hi` side)."""
    lo: float
    hi: float
    coarse: int = COARSE_STEPS
    fine: int = FINE_STEPS
    stops: list = field(default_factory=list)     # dict(x, hfr, sharp, n, pass)
    best: float | None = None
    hfr: float | None = None                      # the stars' size at `best`
    why: str = ""
    _queue: list = field(default_factory=list)
    _stage: str = "coarse"

    def __post_init__(self):
        # from the near side to the far: every stop approached the same way
        self._queue = list(np.linspace(self.hi, self.lo, self.coarse))

    @property
    def step(self) -> float:
        return (self.hi - self.lo) / (self.coarse - 1)

    @property
    def done(self) -> bool:
        return self._stage == "done"

    def next(self) -> float | None:
        if not self._queue:
            self._advance()
        return float(self._queue[0]) if self._queue else None

    def add(self, frames) -> dict:
        """What the lens showed at `next()`'s position: one frame or a
        list of them (their measures' medians). -> that stop."""
        x = float(self._queue.pop(0))
        if isinstance(frames, np.ndarray):
            frames = [frames]
        ms = [m for m in (measure(f) for f in frames) if m["hfr"] is not None]
        stop = dict(x=x, hfr=None, sharp=None, n=0, stage=self._stage)
        if ms:
            stop.update(hfr=float(np.median([m["hfr"] for m in ms])),
                        sharp=float(np.median([m["sharp"] for m in ms])),
                        n=int(np.median([m["n"] for m in ms])))
        if self._stage != "settle":
            self.stops.append(stop)
        elif not self._queue and stop["hfr"] is not None:
            self.hfr = stop["hfr"]
        return stop

    def _seen(self, stage):
        return [s for s in self.stops if s["stage"] == stage
                and s["hfr"] is not None]

    def _advance(self) -> None:
        if self._stage == "coarse":
            seen = self._seen("coarse")
            if len(seen) < 3:
                self._finish(None, "no stars to focus on (cloud, a lens "
                             "cap, or the exposure too short)")
                return
            xs = np.array([s["x"] for s in seen])
            hf = np.array([s["hfr"] for s in seen])
            i = int(np.argmin(hf))
            # the V through the stops about the smallest stars (far from
            # focus the faint stars are lost and the rest measure small)
            near = np.abs(xs - xs[i]) <= 3.5 * self.step
            fit = v_curve(xs[near], hf[near])
            x0 = xs[i] if fit is None else fit[0]
            x0 = float(np.clip(x0, xs[i] - self.step, xs[i] + self.step))
            self._coarse = float(np.clip(x0, self.lo, self.hi))
            a = max(self.lo, self._coarse - 1.5 * self.step)
            b = min(self.hi, self._coarse + 1.5 * self.step)
            self._stage = "fine"
            self._queue = list(np.linspace(b, a, self.fine))
        elif self._stage == "fine":
            seen = self._seen("fine")
            if len(seen) < 3:
                self._finish(self._coarse, "from the coarse pass (the fine "
                             "one lost the stars)")
                return
            xs = np.array([s["x"] for s in seen])
            sh = np.array([s["sharp"] for s in seen])
            x0 = _peak(xs, sh)
            if x0 is None:                         # no top inside: an end
                x0 = float(xs[int(np.argmax(sh))])
            self._finish(x0, "at the far end of the lens's travel"
                         if x0 <= self.lo + 1e-9 else "")
        else:
            self._stage = "done"

    def _finish(self, x, why: str) -> None:
        self.best, self.why = x, why
        if x is None:
            self._stage, self._queue = "done", []
            return
        # back out to the near side, then in: the passes' own direction
        self._stage = "settle"
        self._queue = [min(self.hi, x + 2 * self.step), x]

    def summary(self) -> str:
        if self.best is None:
            return "focus: " + self.why
        size = "" if self.hfr is None else f", stars {self.hfr:.2f} px (HFR)"
        return (f"focus: lens at {self.best:.3f}{size}"
                + (f" ({self.why})" if self.why else ""))


def drifted(hfr_now: float | None, hfr_focused: float | None,
            ratio: float = DRIFT) -> bool:
    """During the recording: the stars have grown `ratio` times since the
    sweep (the lens sagged, or cooled): time to refocus."""
    return hfr_now is not None and hfr_focused is not None and \
        hfr_now > ratio * max(hfr_focused, 0.8)
