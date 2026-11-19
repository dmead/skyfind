"""A camera that moves: how it turned against the ground, and so how long
between frames.

A motion-control rig pans, tilts or slides the camera through a clip. The
stars still give each frame's orientation exactly (a plate solve per frame),
but that orientation is the Earth's turn and the camera's together, and a
steady pan looks just like a faster or slower sky: from the stars alone the
frame interval — which times every satellite — is lost (a dolly clip: 32.5 s per
frame from a fixed-camera fit; really about 43).

The landscape splits them. Far land (a ridge kilometres away) doesn't move
when the camera slides a metre or two, so its motion in the image is the
camera's turn alone. With the frame interval s, the stars' orientations Q_k
give the camera's turn from the Earth-fixed pose at k0,

    G_k = Q_k R(a, s * sidereal * (k - k0))^T Q_k0^T

and the right s is the one that makes the land keep still: G_j G_i^T is how
the land should turn from frame i to frame j, and the skyline says how it
did. The skyline is followed as a profile, per column the row where the
bright sky gives way to dark land (a silhouette at night), matched between
frames by a sideways shift and then a rotation of its rays; trees and near
ground (moved by the slide) fall out as outliers.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from .camera import SIDEREAL_DEG_PER_DAY, Camera, kabsch, rotate

EDGE = 4                   # rows either side of a skyline edge


def skyline_rows(gray: np.ndarray, top: np.ndarray, *, span: float = 0.12,
                 min_drop: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """(rows, ok) per column: where the sky gives way to land, searched
    within `span` of the frame height around `top` (the foreground mask's
    first land row per column: the land moves, so the search is wide)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    h, w = g.shape
    band = int(span * h)
    d = np.full((h, w), -np.inf, np.float32)
    d[EDGE:h - EDGE] = g[:h - 2 * EDGE] - g[2 * EDGE:]   # above minus below
    rows = np.arange(h)[:, None]
    t = np.asarray(top, float)[None, :]
    d[(rows < t - band) | (rows > t + band)] = -np.inf
    y = np.argmax(d, axis=0)
    ok = (d[y, np.arange(w)] > min_drop) & np.isfinite(t[0]) & (t[0] < h)
    return y.astype(float), ok


def _line(t: np.ndarray, ok: np.ndarray, h: int, *, prefer_low: bool
          ) -> np.ndarray | None:
    """RANSAC: the straight line (row per column) most of the columns `ok`
    agree on; with `prefer_low`, ties go to the lower line."""
    w = len(t)
    x = np.arange(w, dtype=float)
    if ok.sum() < w // 10:
        return None
    rng = np.random.default_rng(0)
    xs, ts = x[ok], t[ok]
    tol = max(10.0, 0.02 * h)
    best = None
    for _ in range(500):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        m = (ts[j] - ts[i]) / (xs[j] - xs[i])
        if abs(m) > 0.5:
            continue
        line = ts[i] + m * (x - xs[i])
        n = int((np.abs(ts - line[ok]) < tol).sum())
        score = (n, float(np.mean(line)) if prefer_low else 0.0)
        if best is None or score > best[0]:
            best = (score, line)
    if best is None:
        return None
    near = np.abs(ts - best[1][ok]) < tol
    return np.polyval(np.polyfit(xs[near], ts[near], 1), x)


def land_top(fg_mask: np.ndarray) -> np.ndarray:
    """Where to look for the skyline in each column: a straight line
    through the foreground mask's first land rows, the lowest most columns
    agree on (a tree or a mast pokes far above it; columns the mask missed
    have no land). The land moves with the camera, the mask doesn't, so
    this is only the middle of a wide first search. None when the mask
    has too little land to say (a turning camera's land doesn't stay put,
    and a mask of what stays put may find none)."""
    land = fg_mask > 0
    h, w = land.shape
    top = np.where(land.any(0), land.argmax(0), -1).astype(float)
    return _line(top, (top > 0) & (top < h - 1), h, prefer_low=True)


def _match(ya, oka, yb, okb, max_shift: int):
    """Sideways shift of profile b against a: (dx, median error). The
    camera rolls and tilts as well, so what's left after the shift is a
    tilted line (a 2 deg roll is 70 px across a 1080p frame), not a
    constant: that line is taken out before judging the fit."""
    w = len(ya)
    x = np.arange(w)
    xc = (x - w / 2) / w
    best = (np.inf, 0)
    for dx in range(-max_shift, max_shift + 1):
        src = x - dx
        ok = okb & (src >= 0) & (src < w)
        ok[ok] &= oka[src[ok]]
        if ok.sum() < w // 8:
            continue
        r = yb[ok] - ya[src[ok]]
        u = xc[ok]
        for _ in range(2):                     # a line, then refit robustly
            A = np.column_stack([np.ones(len(u)), u])
            sel = slice(None) if _ == 0 else                 np.abs(r - A @ c) <= np.percentile(np.abs(r - A @ c), 75)
            c = np.linalg.lstsq(A[sel], r[sel], rcond=None)[0]
        e = np.median(np.abs(r - A @ c))
        if e < best[0]:
            best = (e, dx)
    return best[1], best[0]


def land_turn(cam: Camera, ya, oka, yb, okb, *, max_shift: int | None = None
              ) -> tuple[np.ndarray, float, int] | None:
    """How the land turned from profile a's frame to profile b's: the
    rotation R (camera rays, a -> b), the median ray misfit (px) and how
    many columns agreed; None when they don't match."""
    w = len(ya)
    max_shift = max_shift or w // 5
    dx, err = _match(ya, oka, yb, okb, max_shift)
    if not np.isfinite(err):
        return None
    x = np.arange(w, dtype=float)
    R = np.eye(3)
    ra_all = cam.px_to_ray(np.column_stack([x, ya]))
    for it in range(4):
        # frame a's columns carried into frame b by R (first: the shift)
        if it == 0:
            xb = x + dx
        else:
            pb, _ = cam.ray_to_px(ra_all @ R.T)
            xb = pb[:, 0]
        ok = oka & (xb >= 0) & (xb <= w - 1)
        ok[ok] &= okb[np.clip(np.rint(xb[ok]).astype(int), 0, w - 1)]
        if ok.sum() < w // 8:
            return None
        ybi = np.interp(xb[ok], x, yb)
        a, b = ra_all[ok], cam.px_to_ray(np.column_stack([xb[ok], ybi]))
        keep = np.ones(len(a), bool)
        for _ in range(3):                 # trees, near ground: outliers
            R = kabsch(a[keep], b[keep])
            e = np.linalg.norm(a @ R.T - b, axis=1)
            keep = e <= np.percentile(e, 75)
    e = np.linalg.norm(a @ R.T - b, axis=1)[keep]
    return R, float(np.median(e) * cam.f), int(keep.sum())


def profiles(frames: dict[int, np.ndarray], fg_mask: np.ndarray) -> dict:
    """{k: (rows, ok)} skyline profiles of BGR or grey frames: a wide
    search about the mask's land line, then a narrow one about the line
    the frame's own skyline follows (the camera moved it)."""
    top = land_top(fg_mask)
    span = 0.2
    if top is None:              # no mask to go by: the lower 60% of the frame
        h, w = fg_mask.shape
        top, span = np.full(w, 0.7 * h), 0.3
    out = {}
    for k, f in frames.items():
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        y, ok = skyline_rows(g, top, span=span)
        line = _line(y, ok, g.shape[0], prefer_low=False)
        out[k] = (y, ok) if line is None else             skyline_rows(g, line, span=0.06)
    return out


def land_pairs(cam: Camera, prof: dict, ks, *, gap=(2, 6),
               max_err_px: float = 3.0) -> dict:
    """{(i, j): R} the land's turn between sampled frames `gap` samples
    apart (long enough for the Earth to have turned well past the
    measurement's noise, short enough for the views to overlap)."""
    ks = sorted(ks)
    out = {}
    for a in range(len(ks)):
        for b in range(a + gap[0], min(a + gap[1] + 1, len(ks))):
            i, j = ks[a], ks[b]
            t = land_turn(cam, *prof[i], *prof[j])
            if t is not None and t[1] <= max_err_px:
                out[(i, j)] = t[0]
    return out


def camera_turns(Q: dict, s: float, k0: int, cap, *, axis=(0, 0, 1.0),
                 sign: float = 1.0) -> dict:
    """{k: G_k} the camera's turn from its Earth-fixed pose at k0, for a
    frame interval of `s` seconds (Q: GCRS -> camera per frame; `cap`: the
    capture index of a frame)."""
    axis = np.asarray(axis, float)
    w = sign * np.radians(SIDEREAL_DEG_PER_DAY / 86400.0 * s)
    out = {}
    for k, q in Q.items():
        a = -w * (float(cap(k)) - float(cap(k0)))
        Rk = np.stack([rotate(e, axis, a) for e in np.eye(3)], 1)
        out[k] = q @ Rk.T @ Q[k0].T
    return out


def interval(Q: dict, pairs: dict, k0: int, cap, *, axis=(0, 0, 1.0),
             s_range=(1.0, 3600.0)) -> tuple[float, float, float]:
    """The frame interval (s) that keeps the land still: (s, sign, rms
    misfit in deg of the best four fifths of the pairs). `pairs`: the
    land's measured turns (`land_pairs`)."""
    def cost(s, sign):
        G = camera_turns(Q, s, k0, cap, axis=axis, sign=sign)
        e = np.sort([np.linalg.norm(Rotation.from_matrix(
            R.T @ G[j] @ G[i].T).as_rotvec()) for (i, j), R in pairs.items()])
        # the worst fifth left out: a pair matched on a tree, a plane
        return float(np.sqrt(np.mean(np.square(e[:max(1, int(0.8 * len(e)))]))))

    best = None
    for sign in (1.0, -1.0):
        grid = np.geomspace(*s_range, 400)
        c = [cost(s, sign) for s in grid]
        i = int(np.argmin(c))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        r = minimize_scalar(lambda s: cost(s, sign), bounds=(lo, hi),
                            method="bounded", options=dict(xatol=1e-3))
        if best is None or r.fun < best[2]:
            best = (float(r.x), sign, float(r.fun))
    s, sign, e = best
    return s, sign, float(np.degrees(e))


# ------------------------------------------------ cleaning for a moving camera
def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    h, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64),
                              0)
    return h / h[2, 2]


def _grid(cam: Camera, n: int = 12) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(0, cam.width - 1, n + 4),
                         np.linspace(0, cam.height - 1, n))
    return np.column_stack([xs.ravel(), ys.ravel()])


def sky_track(cam: Camera, n_frames: int):
    """The sky's motion frame to frame (remove-satellites' `Track`) from a
    fitted camera: a grid of pixels carried to the sky at frame k and back
    to frame k+1, one homography per step. A turning camera throws the
    tracker (it takes the camera's turn for bad or dropped frames); the
    stars have already said exactly how the sky moved."""
    from remove_satellites.core.track import Track
    xy = _grid(cam)
    steps = np.empty((n_frames - 1, 3, 3))
    for k in range(n_frames - 1):
        v = cam.unproject(xy, k)
        p, z = cam.project(v, k + 1)
        ok = z > 0.05
        steps[k] = _homography(xy[ok], p[ok])
    return Track(steps=steps, inliers=np.full(n_frames - 1, len(xy)),
                 residual_px=0.0)


def land_steps(cam: Camera, i: int, j: int) -> np.ndarray:
    """Homography carrying far land from frame i to frame j: the camera's
    own turn between them (the sky's turn left out)."""
    xy = _grid(cam)
    r = cam._turned(cam._unturned(cam.px_to_ray(xy), i), j)
    p, z = cam.ray_to_px(r)
    ok = z > 0.05
    return _homography(xy[ok], p[ok])


STAR_PEAK = 6.0           # a star: a local peak this far over the background
STAR_HELD = 0.35          # ... that the sky's motion puts back within this


def _looks(gray, cam: Camera, track, k: int, gaps, *, stars: bool = False):
    """(score, stars) of frame k against a few frames earlier. `score`:
    how much better it matches them carried the land's way (the camera's
    turn) than the sky's way, the median over `gaps` (a flicker or a
    trail doesn't count): > 0 on land. Not < 0 only on sky, though: near
    ground a sliding rig moves by parallax matches neither way, and scores
    anything. `stars` (when asked for): the peaks the sky's motion put
    back where they are in frame k, in every gap (to STAR_HELD of their
    height): sky, then, and nothing else is. None, None before any gap.
    `gray(j)`: frame j prepared (`_prep`: background off)."""
    w, h = cam.width, cam.height
    b = gray(k)
    votes, held = [], []
    for g in gaps:
        if k - g < 0:
            continue
        a = gray(k - g)
        sky = cv2.warpPerspective(a, track.between(k - g, k), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        ground = cv2.warpPerspective(a, land_steps(cam, k - g, k), (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
        ds = np.abs(sky - b)
        votes.append(ds - np.abs(ground - b))
        held.append(ds)
    if not votes:
        return None, None
    score = cv2.GaussianBlur(np.median(np.stack(votes), axis=0), (0, 0), 2)
    if not stars:
        return score, None
    peak = (b > STAR_PEAK) & (b >= cv2.dilate(b, np.ones((5, 5), np.uint8)))
    kept = np.max(np.stack(held), axis=0) < STAR_HELD * b
    return score, (peak & kept).astype(np.uint8)


def _reader(path):
    from remove_satellites.core import video
    from remove_satellites.core.foreground import _prep
    return lambda j: _prep(video.read_frame(path, j))


def _sky_rule(land: np.ndarray) -> np.ndarray:
    """Land -> foreground as for a fixed camera: sky is what's connected to
    the top row, the rest land; specks in the sky (a star the turn left a
    pixel off) aren't land: left in, each would punch a hole in the trails
    across it."""
    h, w = land.shape
    sky = 1 - land
    _, lab = cv2.connectedComponents(sky, connectivity=4)
    seeds = np.unique(lab[0][sky[0] == 1])
    fg = (~np.isin(lab, seeds)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    small = np.flatnonzero(st[:, cv2.CC_STAT_AREA] < 1e-3 * w * h)
    fg[np.isin(lab, small[small > 0])] = 0
    return fg


def _disc(px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def land_mask(path, cam: Camera, track, frames, *, gaps=(4, 8, 12),
              thresh: float = 3.0, close_px: int = 15) -> np.ndarray:
    """uint8 mask, 1 = land anywhere it went through the clip. At each of
    `frames` (ones the stars were followed through: elsewhere the camera's
    turn is only held, not known) the land is what `_looks` like land,
    specks opened away. One mask for the clip, so it's the union over it
    (`fgmask.png`: for the cleaned copy, the skyline, the sky's rays); the
    trails are found against each frame's own (`FrameLand`). Sky =
    connected to the top row, as for a fixed camera."""
    gray = _reader(path)
    land = np.zeros((cam.height, cam.width), np.uint8)
    for k in frames:
        sc, _ = _looks(gray, cam, track, int(k), gaps)
        if sc is not None:
            land |= cv2.morphologyEx((sc > thresh).astype(np.uint8),
                                     cv2.MORPH_OPEN, _disc(9))
    if close_px > 1:
        land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, _disc(close_px))
    return _sky_rule(land)


NEAR = (-3, 0, 3)          # frames whose land a frame's near land gathers
SKY_REACH = 31             # px: sky around a star it shows moving with the sky


class FrameLand:
    """Each frame's own land for a turning camera. The union over the clip
    (`land_mask`) hides sky wherever the land went: a dolly clip's tree, close to
    a sliding rig, sweeps across a third of the frame, while any one frame
    shows 15-20% land. Frame k gets back, of the union, what shows stars
    moving with the sky in frame k (`_looks`' stars, reaching `SKY_REACH`
    around each), less its near land (what looks like land in frames
    k-3..k+3, carried the land's way). Ground shows no stars, so it stays
    land however little of it looks like land; a smooth twilight sky
    shows none either, and stays as the union has it."""

    def __init__(self, path, cam: Camera, track, union: np.ndarray, *,
                 gaps=(4, 8, 12), thresh: float = 3.0, keep: int = 48):
        from remove_satellites.core import video
        self.path, self.cam, self.track = path, cam, track
        self.union = union.astype(np.uint8)
        self.gaps, self.thresh, self.keep = gaps, thresh, keep
        self.n = video.probe(path).frame_count
        self._gray: dict[int, np.ndarray] = {}
        self._here: dict[int, tuple] = {}
        self._land: dict[int, np.ndarray] = {}
        self._read = _reader(path)

    def feed(self, j: int, frame: np.ndarray) -> None:
        """Frame j as read in order (saves reading it again)."""
        from remove_satellites.core.foreground import _prep
        if j not in self._gray:
            self._gray[j] = _prep(frame)
        self._prune(self._gray, j)

    def _prune(self, d: dict, j: int) -> None:
        for old in [i for i in d if i < j - self.keep]:
            del d[old]

    def gray(self, j: int) -> np.ndarray:
        if j not in self._gray:
            self._gray[j] = self._read(j)
        return self._gray[j]

    def here(self, j: int):
        """(land, stars) of frame j by its own look, or None."""
        if j not in self._here:
            sc, st = _looks(self.gray, self.cam, self.track, j, self.gaps,
                            stars=True)
            self._here[j] = None if sc is None else (
                cv2.morphologyEx((sc > self.thresh).astype(np.uint8),
                                 cv2.MORPH_OPEN, _disc(9)), st)
            self._prune(self._here, j)
        return self._here[j]

    def __call__(self, k: int) -> np.ndarray:
        if k in self._land:
            return self._land[k]
        w, h = self.cam.width, self.cam.height
        near = np.zeros((h, w), np.uint8)
        for d in NEAR:
            j = k + d
            hj = self.here(j) if 0 <= j < self.n else None
            if hj is not None:
                near |= cv2.warpPerspective(hj[0], land_steps(self.cam, j, k),
                                            (w, h), flags=cv2.INTER_NEAREST)
        hk = self.here(k)
        land = self.union
        if hk is not None:
            shown = cv2.dilate(hk[1], _disc(SKY_REACH)) & \
                (1 - cv2.dilate(near, _disc(15)))
            land = _sky_rule(self.union & (1 - shown))
        self._land[k] = land
        self._prune(self._land, k)
        return land


def turning_frames(path, track, land_of, radius: int, lo: int, hi: int, *,
                   threads: int = 4, lead: int = max(NEAR)):
    """(k, original, sky median, land) for frames lo..hi-1, in order: the
    median the cleaner takes (the frames around k carried the sky's way;
    a sample left out, 0 and 255 alternately, where its frame showed land
    or nothing), but each frame's own land left out (`land_of(k)`), not
    one mask for the clip: the sky the turn swept the land across is
    searched too. One sequential read through a ring of the window
    (`lead` frames ahead: a frame's land looks at the next few), each
    frame handed to `land_of.feed` when it has one."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    from remove_satellites.core import video
    info = video.probe(path)
    n, w, h = info.frame_count, info.width, info.height
    r = radius
    back = r + max(gaps_of(land_of))
    start, stop = max(0, lo - back), min(n, hi + r + lead)
    reader = enumerate(video.iter_frames(path, start, stop), start)
    buf: deque = deque(maxlen=2 * r + 1 + lead)
    feed = getattr(land_of, "feed", None)
    lands: dict[int, np.ndarray] = {}

    def land(k):
        if k not in lands:
            lands[k] = land_of(k)
            for old in [j for j in lands if j < k - 2 * r]:
                del lands[old]
        return lands[k]

    def aligned(k, f, c):
        if k == c:
            return f
        H = track.between(k, c)
        out = cv2.warpPerspective(f, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        ok = cv2.warpPerspective(1 - land(k), H, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[ok == 0] = 0 if (k - c) % 2 == 0 else 255
        return out

    strips = [x for x in np.array_split(np.arange(h), max(1, threads))
              if x.size]
    with ThreadPoolExecutor(max(1, threads)) as ex:
        for c in range(lo, hi):
            top = min(n - 1, c + r)
            while not buf or buf[-1][0] < min(n - 1, top + lead):
                k, f = next(reader)
                buf.append((k, f))
                if feed is not None:
                    feed(k, f)
            window = [(k, f) for k, f in buf if c - r <= k <= top]
            stack = np.stack(list(ex.map(lambda kf: aligned(*kf, c),
                                         window)))
            mid = len(stack) // 2

            def strip(rs, stack=stack, mid=mid):
                return np.partition(stack[:, rs[0]:rs[-1] + 1], mid,
                                    axis=0)[mid]

            med = np.concatenate(list(ex.map(strip, strips)), axis=0)
            yield c, dict(window)[c], med, land(c)


def gaps_of(land_of) -> tuple:
    return tuple(getattr(land_of, "gaps", (0,)))
