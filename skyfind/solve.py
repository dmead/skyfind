"""Blind site/time solve from satellite trails, then identify every trail.

Knowns: each trail segment as two sky directions (camera fit), the site's
inertial position per frame up to (lat, dlst, h) (`site.Site`), the frame
interval (sky rotation rate), a catalogue of orbital elements, and a nominal
time `t_ref` for frame 0 (planets, or the user) with a search half-window
`twin` around it. Unknowns: geodetic latitude, sidereal correction dlst,
clock offset T (frame 0's mid-exposure is t_ref + T), exposure E, height h.

1. Per track (one object followed over >= 3 frames): every catalogued object
   that passes along it at some T within +-twin is fitted with (lat, dlst,
   T, E) — 4 unknowns against 4n end coordinates. A correct identification
   fits; chance can't fit many frames. But one high-orbit object can fit
   many (site, time) pairs over +-12 h, so every good fit is kept.
2. Cluster: the answer is where fits of *distinct* objects coincide.
3. Joint fit of the cluster, joined by other tracks that agree.
4. Identify every segment at the solution; count matches again with the
   clock deliberately wrong to estimate how many IDs are chance.

Segment end times: the object is at one end at t_k + T - E/2 and at the
other at t_k + T + E/2, t_k = frame k's nominal mid-exposure. An end cut
short by the frame edge, the skyline or the Earth's shadow isn't at an
exposure boundary; such ends (`clip_a`, `clip_b`) don't enter the fits.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import data, native
from .camera import Camera, rotate, unit
from .orbits import Orbits
from .site import A_WGS84, F_WGS84, WE, Site

SITE_GRID = [(dla, dls) for dla in (-4, -2, 0, 2, 4)
             for dls in (-6, -4, -2, 0, 2, 4, 6)]
STEPS = [0.002, 0.002, 0.1, 0.1, 0.02]      # lat, dlst deg; T, E s; h km
GOOD_FIT = 0.25                             # deg, median end error
# a moving camera's interval searched +-8% about the land's: low, smooth
# hills pin the land's sideways shift poorly (the synthetic rig's came out
# 3% short with no parallax at all)
INTERVAL_SPAN = 0.08


@dataclass
class Seg:
    k: int
    a: np.ndarray                 # GCRS unit vector of end 1 (x1, y1)
    b: np.ndarray                 # end 2 (x2, y2)
    clip_a: bool = False
    clip_b: bool = False
    xy: tuple = ()                # (x1, y1, x2, y2) pixels
    length: float = 0.0


def segs_from_pixels(camera: Camera, rows) -> list[Seg]:
    """Trail segments (dicts with k, x1, y1, x2, y2[, clip_a, clip_b, len])
    -> sky directions through the camera model."""
    out = []
    for s in rows:
        xy = np.array([[s["x1"], s["y1"]], [s["x2"], s["y2"]]], float)
        v = camera.unproject(xy, s["k"])
        out.append(Seg(int(s["k"]), v[0], v[1], bool(s.get("clip_a", False)),
                       bool(s.get("clip_b", False)),
                       (s["x1"], s["y1"], s["x2"], s["y2"]),
                       float(s.get("len", np.hypot(s["x2"] - s["x1"],
                                                   s["y2"] - s["y1"])))))
    return out


def _deg(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ====================================================================== problem
@dataclass
class Problem:
    orbits: Orbits
    names: list
    norad: np.ndarray
    site: Site
    t_ref: float                  # TT jd: nominal mid-exposure of frame 0
    spf: float                    # s per frame
    twin: float                   # s: clock search half-window
    lat0: float                   # deg: start latitude (skyline)
    h0: float = 1.0               # km: assumed height (+-1) unless fitted
    camera: Camera | None = None
    n_frames: int = 0
    log: object = field(default=print, repr=False)
    # camera-frame rays of the frame's sky when nothing says which way is
    # up (pure sky): the site is then free (`free_passes`), `site.up_cam`
    # only a reference for dlst
    sky: np.ndarray | None = None

    @property
    def free(self) -> bool:
        return self.sky is not None

    def s(self, k) -> np.ndarray:
        """Capture index of frame(s) k (frames a re-timed clip dropped are
        skipped; see `Camera.clock`)."""
        return self.camera.s(k) if self.camera is not None else \
            np.asarray(k, float)

    def tk(self, k) -> np.ndarray:
        return self.t_ref + self.s(k) * self.spf / 86400

    def exposure_turn(self) -> tuple[np.ndarray, float]:
        """(axis, w): a point recorded `off` s from mid-exposure lands
        w * off rad about `axis` (GCRS) from where the frame's model puts
        it — the sky's turn, and a moving camera's own (`Camera.own_spin`)
        added to it."""
        cam = self.camera
        if cam is None:
            return np.zeros(3), 0.0
        w = -math.radians(cam.omega) / self.spf
        if not cam.moving:
            return cam.axis, w
        v = w * cam.axis + cam.own_spin(self.spf)
        n = float(np.linalg.norm(v))
        return v / n, n

    def with_interval(self, spf: float) -> "Problem":
        """This problem for a moving camera at another frame interval: the
        same orbit table, the camera's turn and the observer's positions
        redone (`Camera.with_interval`)."""
        cam = self.camera.with_interval(spf)
        n_cap = float(cam.s(self.n_frames - 1)) + 1
        t_mid = self.t_ref + n_cap * spf / 2 / 86400
        site = Site(cam, self.site.up_cam, t_mid, self.n_frames)
        return replace(self, camera=cam, spf=spf, site=site)

    def dirs(self, j, ks, offs, T, lat, dlst, h=None) -> np.ndarray:
        """Unit directions of object(s) j seen from the site at frames `ks`,
        `offs` s from mid-exposure, clock offset T. Only `offs` moves the
        observer: the stars fixed its inertial position at each frame's
        actual mid-exposure; T moves the satellites alone.

        Returned as the frame's mid-exposure camera model sees them: the
        camera turns with the Earth, so a point recorded `off` s from
        mid-exposure unprojects `off` s of sky rotation away from its true
        direction (+-0.05 deg at the ends of a 25 s exposure)."""
        h = self.h0 if h is None else h
        ks = np.asarray(ks, float)
        offs = np.broadcast_to(np.asarray(offs, float), ks.shape)
        rs = self.site.gcrs(ks, lat, dlst, h)
        vs = self.site.velocity(rs)
        p = self.orbits.at_many(self.tk(ks) + (T + offs) / 86400,
                                np.broadcast_to(j, ks.shape))
        d = unit(p - rs - vs * offs[..., None])
        if self.camera is None:
            return d
        if self.camera.moving:
            ax, w = self.exposure_turn()
            return rotate(d, ax, w * offs)
        return rotate(d, self.camera.axis,
                      -np.radians(self.camera.omega) * offs / self.spf)

    def utc(self, T: float, k: float = 0.0) -> str:
        t = data.timescale().tt_jd(float(self.tk(k)) + T / 86400)
        return t.utc_strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------------------- persistence
    def save(self, folder) -> None:
        """Orbit table + everything else a worker process needs."""
        folder = Path(folder)
        self.orbits.save(folder)
        (folder / "problem.json").write_text(json.dumps(dict(
            names=list(self.names), norad=[int(n) for n in self.norad],
            camera=self.camera.to_dict(), up_cam=self.site.up_cam.tolist(),
            t_ref=self.t_ref, spf=self.spf, twin=self.twin, lat0=self.lat0,
            h0=self.h0, n_frames=self.n_frames,
            sky=None if self.sky is None else np.asarray(self.sky).tolist())))

    @classmethod
    def load(cls, folder, log=print) -> "Problem":
        folder = Path(folder)
        d = json.loads((folder / "problem.json").read_text())
        cam = Camera.from_dict(d["camera"])
        n_cap = float(cam.s(d["n_frames"] - 1)) + 1
        t_mid = d["t_ref"] + n_cap * d["spf"] / 2 / 86400
        site = Site(cam, d["up_cam"], t_mid, d["n_frames"])
        return cls(Orbits.load(folder), d["names"], np.array(d["norad"]),
                   site, d["t_ref"], d["spf"], d["twin"], d["lat0"],
                   d["h0"], cam, d["n_frames"], log,
                   None if d.get("sky") is None else np.array(d["sky"]))


def build_problem(camera: Camera, up_cam, elements, t_ref: float, twin: float,
                  n_frames: int, *, lat0: float | None = None,
                  sky=None, h0: float = 1.0, grid_dt: float | None = None,
                  folder=None, workers: int = 1, pool=None,
                  shared: bool = False, on_block=None,
                  log=print) -> Problem:
    """Propagate the catalogue over t_ref +- twin plus the clip's length
    (in shared memory with `shared`, or into `folder`; by `workers`
    processes, or `pool`'s). `sky`: camera rays of the frame's sky when
    `up_cam` is only a reference (pure sky: the site is free)."""
    from .tles import nearest_epoch
    spf = camera.seconds_per_frame
    n_cap = float(camera.s(n_frames - 1)) + 1          # captures spanned
    t_mid = t_ref + n_cap * spf / 2 / 86400
    els = nearest_epoch(elements, t_mid - 69.184 / 86400)   # ~UTC
    lo = t_ref - (twin + 180) / 86400
    # a moving camera's interval is searched about the land's
    # (`with_interval`): room for the clip to run that much longer
    pad = 0.0
    if camera.moving:
        land = camera.meta.get("moving", {}).get("interval_from_land", spf)
        # a tracking mount's interval is searched over the whole range
        # satellites allow (`pipeline._interval_search`)
        land = max(land, camera.meta.get("tracked", {}).get("interval_max",
                                                            0.0))
        pad = max(0.0, max(spf, land) * (1 + INTERVAL_SPAN) - spf) * n_cap
    hi = t_ref + (n_cap * spf + twin + 180 + pad) / 86400
    if grid_dt is None:           # SKYFIND_GRID_DT: coarser on a phone
        grid_dt = float(os.environ.get("SKYFIND_GRID_DT", 20.0))
    orb = Orbits.propagate([(e.line1, e.line2) for e in els], lo, hi,
                           grid_dt=grid_dt, folder=folder, workers=workers,
                           pool=pool, shared=shared, on_block=on_block,
                           log=log)
    site = Site(camera, up_cam, t_mid, n_frames)
    return Problem(orb, [e.name for e in els],
                   np.array([e.norad for e in els]), site, t_ref, spf, twin,
                   site.horizon_lat if lat0 is None else lat0, h0, camera,
                   n_frames, log, None if sky is None else np.asarray(sky))


# ====================================================================== fits
def fdjac(fun, p, steps):
    """Forward-difference Jacobian with absolute steps.

    least_squares' default relative steps (~1e-8) move the site by
    centimetres and the clock by microseconds, below the float32 orbit
    table's noise: the gradient is noise and fits never leave the start."""
    f0 = fun(p)
    J = np.empty((len(f0), len(p)))
    for i, h in enumerate(steps[:len(p)]):
        q = np.array(p, float)
        q[i] += h
        J[:, i] = (fun(q) - f0) / h
    return J


@dataclass
class Assoc:
    """A track (its segments) assigned to one object, with end order."""
    j: int
    segs: list
    a_first: list                 # per segment: end a is the earlier one
    cut: np.ndarray | None = None  # (n, 2) ends found cut short by the fit

    def observed(self):
        obs, free = [], []
        for s, af in zip(self.segs, self.a_first):
            if af:
                obs.append((s.a, s.b)); free.append((not s.clip_a, not s.clip_b))
            else:
                obs.append((s.b, s.a)); free.append((not s.clip_b, not s.clip_a))
        free = np.array(free)
        if self.cut is not None:
            free &= ~self.cut
        return np.array(obs), free

    def end_errors(self, P: "Problem", p) -> np.ndarray:
        """(n, 2) angular error of each end (nan where not free)."""
        _, free = self.observed()
        out = np.full(free.shape, np.nan)
        out[free] = np.linalg.norm(residuals(P, [self], p).reshape(-1, 3), axis=1)
        return out


def residuals(P: Problem, assocs: list[Assoc], p) -> np.ndarray:
    """Degrees, 3 per free end: predicted minus observed direction."""
    lat, dlst, T, E = p[:4]
    h = p[4] if len(p) > 4 else P.h0
    out = []
    for a in assocs:
        ks = np.repeat([s.k for s in a.segs], 2)
        offs = np.tile([-E / 2, E / 2], len(a.segs))
        d = P.dirs(a.j, ks, offs, T, lat, dlst, h).reshape(-1, 2, 3)
        obs, free = a.observed()
        out.append(np.degrees(d - obs)[free].ravel())
    return np.concatenate(out) if out else np.zeros(0)


def fit(P: Problem, assocs: list[Assoc], p0, *, height: bool = False,
        f_scale: float = 0.15, x_scale=(0.5, 0.5, 20, 5, 0.5)):
    """Least squares over (lat, dlst, T, E[, h]); -> (p, sigma, errs)."""
    p0 = list(p0[:4]) + ([P.h0 if len(p0) < 5 else p0[4]] if height else [])
    n = len(p0)
    # dlst within 12 deg of its start (a free site starts anywhere); short
    # exposures allowed
    lo = [-89.9, p0[1] - 12, -P.twin - 120, 0.2, -0.5][:n]
    hi = [89.9, p0[1] + 12, P.twin + 120, P.spf, 9.0][:n]
    p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    f = lambda q: residuals(P, assocs, q)                   # noqa: E731
    sol = least_squares(f, p0, jac=lambda q: fdjac(f, q, STEPS),
                        x_scale=list(x_scale[:n]), bounds=(lo, hi),
                        loss="soft_l1", f_scale=f_scale)
    r = f(sol.x)
    errs = np.linalg.norm(r.reshape(-1, 3), axis=1)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * np.sum(r ** 2) / \
            max(1, len(r) - n)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(n, np.nan)
    return sol.x, sig, errs


def end_order(P: Problem, j, segs, p, span=60.0) -> list[bool]:
    """Per segment: does the object pass end a before end b?"""
    ss = np.arange(-span, span + 0.25, 0.5)
    out = []
    for s in segs:
        d = P.dirs(j, np.full(len(ss), s.k), ss, p[2], p[0], p[1],
                   p[4] if len(p) > 4 else None)
        out.append(int(np.argmax(d @ s.a)) < int(np.argmax(d @ s.b)))
    return out


# ====================================================================== per track
def track_motion(P: Problem, segs: list[Seg]) -> tuple[float, np.ndarray]:
    """(angular rate deg/s, unit direction of motion at the middle segment)
    of a track: its exposures are `spf` s apart, so the drift of the segment
    midpoints over the track is the object's apparent motion."""
    m0, m1 = unit(segs[0].a + segs[0].b), unit(segs[-1].a + segs[-1].b)
    mid = unit(segs[len(segs) // 2].a + segs[len(segs) // 2].b)
    dt = float(P.s(segs[-1].k) - P.s(segs[0].k)) * P.spf
    rate = float(_deg(m0 @ m1)) / dt
    u = m1 - m0
    u -= (u @ mid) * mid
    return rate, unit(u)


def _near_objects(P: Problem, seg: Seg, *, near_deg=25.0, quantum=40.0,
                  motion=None, rate_factor=1.5, dir_deg=40.0,
                  part=(0, 1)) -> dict[int, set]:
    """Objects passing within `near_deg` of the segment's ends from the
    nominal site at some clock offset within +-twin: {object: {T rounded to
    `quantum` s}}. The sweep steps along the orbit table's own rows (20 s;
    no interpolation). With the track's `motion` (rate, direction), the
    object must also move at a rate within `rate_factor` and in a direction
    within `dir_deg` — loose enough for the parallax of the site grid, tight
    enough to drop most of the sky. `part` = (i, n) sweeps only the i-th of
    n slices of the clock window (parallel jobs)."""
    tk = float(P.tk(seg.k))
    rc = P.site.gcrs(seg.k, P.lat0, 0.0, P.h0)
    ends = np.stack([seg.a, seg.b])
    cosn = math.cos(math.radians(near_deg))
    cosd = math.cos(math.radians(dir_deg))
    orb = P.orbits
    step = orb.dt
    lo = max(0, int(np.ceil(((tk - P.twin / 86400) - orb.t0) * 86400 / step)))
    hi = min(len(orb.r) - 1,
             int(np.floor(((tk + P.twin / 86400) - orb.t0) * 86400 / step)) + 1)
    rows = np.arange(lo, hi + 1)
    cut = np.linspace(0, len(rows) - 1, part[1] + 1).astype(int)
    rows = rows[cut[part[0]]:cut[part[0] + 1] + 1]
    Ts = (orb.t0 + rows * step / 86400 - tk) * 86400
    near: dict[int, set] = {}

    rc32 = rc.astype(np.float32)             # float32 is plenty for a cone
    ends = ends.astype(np.float32)

    def look(i):
        d = orb.r[i] - rc32
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    d = look(rows[0])
    for T, i2 in zip(Ts[:-1], rows[1:]):
        d2 = look(i2)
        m = (d @ ends.T).max(1)
        hit = m > cosn
        if motion is not None and hit.any():
            j = np.flatnonzero(hit)
            dv = d2[j] - d[j]
            rate = np.degrees(np.linalg.norm(dv, axis=1)) / step
            u = dv - (dv * d[j]).sum(1, keepdims=True) * d[j]
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            ok = (rate < motion[0] * rate_factor) & \
                (rate > motion[0] / rate_factor) & (u @ motion[1] > cosd)
            hit[j[~ok]] = False
        for j in np.flatnonzero(hit):
            near.setdefault(int(j), set()).add(round(T / quantum) * quantum)
        d = d2
    return near


def near_objects(P: Problem, segs: list[Seg], *, part=(0, 1),
                 near_deg=25.0) -> dict[int, set]:
    """A track's (object, clock quanta) worth checking, from its middle
    segment and its motion (`_near_objects`)."""
    return _near_objects(P, segs[len(segs) // 2], near_deg=near_deg,
                         motion=track_motion(P, segs), part=part)


def candidates(P: Problem, segs: list[Seg], **kw):
    """(object, T, E, end order, lat, dlst) for every pass along a track
    (`check_passes` over `near_objects`; `free_passes` for a free site)."""
    if P.free:
        return free_passes(P, segs, **kw)
    return check_passes(P, segs, near_objects(P, segs), **kw)


def _ground(p: np.ndarray, d: np.ndarray, R: float):
    """Where lines from points p (N, 3) back along unit direction d meet a
    sphere of radius R: (site (N, 3), range (N,), hit (N,))."""
    pd = p @ d
    disc = R * R - ((p * p).sum(1) - pd * pd)
    root = np.sqrt(np.maximum(disc, 0))
    rho = pd - root
    return p - rho[:, None] * d, rho, (disc > 0) & (rho > 0)


def _ground_wgs84(p: np.ndarray, d: np.ndarray, h: float):
    """`_ground` onto the ellipsoid (+ h km): a sphere puts low lines of
    sight tens of km off."""
    x, _, _ = _ground(p, d, 6371.0 + h)
    z2 = unit(x)[:, 2] ** 2
    e2 = F_WGS84 * (2 - F_WGS84)
    return _ground(p, d, A_WGS84 * np.sqrt((1 - e2) / (1 - e2 * (1 - z2)))
                   + h)


def _mid(g: Seg) -> np.ndarray:
    return unit(g.a + g.b)


def free_passes(P: Problem, segs: list[Seg], *, part=(0, 1),
                rate_factor=1.25, dir_deg=12.0, tol=0.3, refine_s=2.0,
                fine_s=0.25,
                merge_s=60.0, sun_max_alt=-6.0):
    """(object, T, E, end order, lat, dlst) for passes along a track with
    nothing saying which way is up (pure sky).

    The stars put each segment's direction in inertial space, so for an
    object at a time t the site is where the line from the object back
    along the middle segment's direction meets the Earth. There the frame's
    sky must be above the horizon and the Sun down; the object must move as
    the track does there (rate within `rate_factor`, direction within
    `dir_deg`); and seen from there (turned with the Earth) it must be
    within `tol` deg of two other segments when they were exposed —
    interior ones where the track is long enough, as the Earth's shadow can
    cut a track's first or last segment short, moving its midpoint. Steps
    along the orbit table's rows like `_near_objects`, checking each row's
    survivors every `refine_s` s across the row, then every `fine_s` s
    around the best; `part` as there. Passes
    of one object within `merge_s` collapse to the best."""
    from .date import sun_vectors
    n = len(segs)
    m = n // 2
    k0, d0 = segs[m].k, _mid(segs[m])
    tk0 = float(P.tk(k0))
    ga, gb = segs[max(m - 1, 0)], segs[min(m + 1, n - 1)]
    span = float(P.s(gb.k) - P.s(ga.k)) * P.spf
    rate = float(_deg(_mid(ga) @ _mid(gb))) / span        # deg/s, locally
    u0 = _mid(gb) - _mid(ga)
    u0 = unit(u0 - (u0 @ d0) * d0)
    checks = [segs[1], segs[-2]] if n >= 5 else \
        [g for i, g in enumerate(segs) if i != m][:1] + segs[-1:]
    checks = [(g.k, _mid(g), float(P.s(g.k) - P.s(k0)) * P.spf)
              for g in checks]
    sky = P.camera.frame_to_gcrs(np.asarray(P.sky, float), k0)
    sun = sun_vectors(np.array([tk0]))[0]
    dark = math.sin(math.radians(sun_max_alt))
    cosd = math.cos(math.radians(dir_deg))
    cost = math.cos(math.radians(tol))
    lens = [float(_deg(g.a @ g.b)) for g in segs]
    fwd = []
    for i, g in enumerate(segs):                  # each segment's own motion
        v = _mid(segs[min(i + 1, n - 1)]) - _mid(segs[max(i - 1, 0)])
        fwd.append(bool((g.b - g.a) @ v > 0))
    orb = P.orbits
    step = orb.dt
    lo = max(0, int(np.ceil(((tk0 - P.twin / 86400) - orb.t0) * 86400 / step)))
    hi = min(len(orb.r) - 1,
             int(np.floor(((tk0 + P.twin / 86400) - orb.t0) * 86400 / step)))
    rows = np.arange(lo, hi + 1)
    cut = np.linspace(0, len(rows) - 1, part[1] + 1).astype(int)
    rows = rows[cut[part[0]]:cut[part[0] + 1] + 1]
    best: dict[int, list] = {}
    for i in rows:
        p = orb.r[i].astype(float)
        x, rho, hit = _ground(p, d0, 6371.0 + P.h0)
        j = np.flatnonzero(hit)
        if not j.size:
            continue
        up = unit(x[j])
        ok = ((up @ sky.T).min(1) > 0) & (up @ sun < dark)
        j, up = j[ok], up[ok]
        if not j.size:
            continue
        x, rho, _ = _ground_wgs84(p[j], d0, P.h0)
        w = orb.v[i, j].astype(float) - WE * np.cross(P.site.pole, x)
        w -= (w @ d0)[:, None] * d0
        wn = np.linalg.norm(w, axis=1)
        r_app = np.degrees(wn / rho)
        ok = (r_app < rate * rate_factor) & (r_app > rate / rate_factor) & \
            ((w / np.maximum(wn, 1e-12)[:, None]) @ u0 > cosd)
        j, x = j[ok], x[ok]
        if not j.size:
            continue
        # the pass is up to half a row from the row's time: try the times
        # around it (the site slides along the line of sight with the object)
        # (then finer around the best: over a long track, a second's error
        # in the object's motion along the line of sight moves it past the
        # tolerance by the time of a check segment minutes away)
        t = orb.t0 + i * step / 86400
        top = np.full(len(j), -2.0)
        top_dt = np.zeros(len(j))
        coarse = np.arange(-step / 2, step / 2 + 1e-9, refine_s)
        fine = np.arange(-refine_s / 2, refine_s / 2 + 1e-9, fine_s)
        for grid, base in ((coarse, None), (fine, "best")):
            start = top_dt.copy()
            for dl in grid:
                dls = start + dl if base else np.full(len(j), dl)
                xd, _, hit = _ground_wgs84(
                    orb.at_many(t + dls / 86400, j), d0, P.h0)
                worst = np.where(hit, 2.0, -2.0)
                for kc, dc, dt in checks:
                    pc = orb.at_many(t + (dls + dt) / 86400, j)
                    worst = np.minimum(worst,
                                       unit(pc - P.site.turn(xd, k0, kc)) @ dc)
                better = worst > top
                top[better], top_dt[better] = worst[better], dls[better]
                x[better] = xd[better]
        ok = top > cost
        T0 = (t - tk0) * 86400              # the object at frame k0 at t
        for jj, cc, dl, xx in zip(j[ok], top[ok], top_dt[ok], x[ok]):
            best.setdefault(int(jj), []).append((-float(cc), T0 + float(dl),
                                                 xx))
    E = float(np.median(lens)) / rate if rate > 0 else P.spf / 2
    out = []
    for j, found in best.items():
        found.sort(key=lambda f: f[0])
        kept: list = []
        for f in found:
            if all(abs(f[1] - g[1]) > merge_s for g in kept):
                kept.append(f)
        for _, T, x in kept:
            lat, dl = P.site.cell(k0, x)
            out.append((int(j), float(T), min(max(E, 0.5), P.spf), fwd,
                        float(lat), float(dl)))
    return out


def _coarse_ok(P, j, s, Ts, RS, VS, tol, coarse):
    """Conservative first look at one segment on a `coarse` s grid: a
    (clock, site) cell survives unless even the closest coarse sample is
    further than `tol` plus the distance the object moves in half a coarse
    step (x1.5). The fine check can't pass anything this rejects."""
    ss = np.arange(-60, 60 + coarse / 2, coarse)
    tt = float(P.tk(s.k)) + (Ts[:, None] + ss[None]) / 86400
    p = P.orbits.at_many(tt, np.full(tt.shape, j))
    d = p[:, None] - RS[None, :, None] - VS[None, :, None] * ss[None, None, :, None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)          # (T, ng, n, 3)
    step = _deg((d[..., 1:, :] * d[..., :-1, :]).sum(-1)) / coarse   # deg/s
    ok = np.ones(d.shape[:2], bool)
    for end in (s.a, s.b):
        c = np.nan_to_num(d @ end, nan=-2)
        i = c.argmax(-1)
        rate = np.take_along_axis(step, np.clip(i, 0, step.shape[-1] - 1)
                                  [..., None], -1)[..., 0]
        e = _deg(np.take_along_axis(c, i[..., None], -1)[..., 0])
        ok &= e <= tol + 1.5 * rate * coarse / 2 + 0.01
    return ok


def check_passes(P: Problem, segs: list[Seg], near: dict[int, set], *,
                 tol=3.0, step=1.0, merge_s=60.0, coarse=4.0):
    """(object, T, E, end order, lat, dlst) for passes along every segment
    of a track within `tol` deg, with one consistent T, at some site on a
    grid around the skyline's start values (blind: those are only rough).

    Vectorised per object over all its clock quanta and grid sites; the
    middle segment (the one the objects were preselected on) goes first,
    with a conservative `coarse` look before the fine one, so most
    hypotheses die cheaply. Quanta of one pass that agree within `merge_s`
    collapse to the best."""
    mid = len(segs) // 2
    grid = np.array(SITE_GRID, float) + [P.lat0, 0.0]
    ss = np.arange(-60, 60 + step / 2, step)
    RS = [P.site.gcrs(s.k, grid[:, 0], grid[:, 1], P.h0) for s in segs]
    VS = [P.site.velocity(r) for r in RS]
    order_segs = [mid] + [i for i in range(len(segs)) if i != mid]
    ng, ns, nss = len(grid), len(segs), len(ss)
    out = []
    for j in sorted(near):
        Ts = np.array(sorted(near[j]), float)
        nT = len(Ts)
        alive = _coarse_ok(P, j, segs[mid], Ts, RS[mid], VS[mid], tol,
                           coarse) if coarse else np.ones((nT, ng), bool)
        errs = np.zeros((nT, ng, ns))
        tcs = np.zeros((nT, ng, ns))
        spans = np.zeros((nT, ng, ns))
        first = np.zeros((nT, ng, ns), bool)
        for i in order_segs:
            s = segs[i]
            cT, cg = np.nonzero(alive)             # only the live cells
            if not cT.size:
                break
            rows, inv = np.unique(cT, return_inverse=True)
            tt = float(P.tk(s.k)) + (Ts[rows, None] + ss[None]) / 86400
            p = P.orbits.at_many(tt, np.full(tt.shape, j))[inv]  # (C, nss, 3)
            d = p - RS[i][cg, None] - VS[i][cg, None] * ss[None, :, None]
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            ca, cb = d @ s.a, d @ s.b
            ia = np.nan_to_num(ca, nan=-2).argmax(-1)
            ib = np.nan_to_num(cb, nan=-2).argmax(-1)
            ea = _deg(np.take_along_axis(ca, ia[..., None], -1)[..., 0])
            eb = _deg(np.take_along_axis(cb, ib[..., None], -1)[..., 0])
            e = np.maximum(ea, eb)
            ok = (e <= tol) & (ia > 0) & (ia < nss - 1) & (ib > 0) & \
                (ib < nss - 1)
            alive[cT, cg] = ok
            errs[cT, cg, i] = e
            tcs[cT, cg, i] = Ts[cT] + (ss[ia] + ss[ib]) / 2
            spans[cT, cg, i] = np.abs(ss[ib] - ss[ia])
            first[cT, cg, i] = ia < ib
        alive &= np.ptp(tcs, axis=2) < 40
        if not alive.any():
            continue
        med = np.where(alive, np.median(errs, axis=2), np.inf)
        found = []                                   # (err, Tc, iT, g)
        for iT in np.flatnonzero(alive.any(1)):
            g = int(np.argmin(med[iT]))
            found.append((med[iT, g], float(np.median(tcs[iT, g])), iT, g))
        found.sort()
        kept: list = []
        for e, tc, iT, g in found:
            if all(abs(tc - k[1]) > merge_s for k in kept):
                kept.append((e, tc, iT, g))
        for e, tc, iT, g in kept:
            out.append((int(j), tc, float(np.median(spans[iT, g])),
                        [bool(o) for o in first[iT, g]], float(grid[g, 0]),
                        float(grid[g, 1])))
    return out


def fit_candidate(P: Problem, segs: list[Seg], cand) -> dict | None:
    """One candidate (object, T, E, end order, lat, dlst) fitted to a track."""
    j, T, E, af, la, dl = cand
    a = Assoc(j, segs, af)
    if not a.observed()[1].any():
        return None
    try:
        p, _, errs = fit(P, [a], [la, dl, T, min(max(E, 0.5), P.spf)])
    except (ValueError, np.linalg.LinAlgError):
        return None
    lat, dlst, T2, E2 = p
    k1 = segs[0].k
    glat, glon = P.site.geodetic(k1, lat, dlst, P.h0,
                                 float(P.tk(k1)) + T2 / 86400)
    return dict(obj=P.names[j], norad=int(P.norad[j]), j=int(j), lat=glat,
                lon=glon, slat=float(lat), dlst=float(dlst), T=float(T2),
                E=float(E2), med_err=float(np.median(errs)),
                max_err=float(np.max(errs)))


def track_result(segs: list[Seg], ncand: int, fits: list, *, keep=GOOD_FIT
                 ) -> dict:
    """Dedupe (several starts converge on one fit) and keep the good fits.
    Ordered by error, then object and clock, so the outcome doesn't depend
    on the order the fits finished in."""
    fits = sorted((f for f in fits if f),
                  key=lambda f: (f["med_err"], f["j"], f["T"]))
    uniq: list = []
    kept: dict[int, list] = {}              # object -> clocks kept for it
    for f in fits:
        ts = kept.setdefault(f["j"], [])
        if all(abs(t - f["T"]) >= 5 for t in ts):
            ts.append(f["T"])
            uniq.append(f)
    good = [f for f in uniq if f["med_err"] < keep]
    return dict(frames=[s.k for s in segs], n=len(segs), ncand=ncand,
                fits=good or uniq[:1])


def fit_many(P: Problem, segs: list[Seg], cands) -> list:
    """`fit_candidate` for many candidates of one track: all at once
    (`batchfit`), or one SciPy fit each with SKYFIND_SCIPY_FITS=1."""
    if os.environ.get("SKYFIND_SCIPY_FITS") == "1":
        return [fit_candidate(P, segs, c) for c in cands]
    from .batchfit import fit_candidates
    return fit_candidates(P, segs, cands)


def fit_track(P: Problem, segs: list[Seg]) -> dict:
    """Every candidate object fitted to one track; all good fits kept (the
    answer is where several tracks' fits coincide)."""
    cands = candidates(P, segs)
    return track_result(segs, len(cands), fit_many(P, segs, cands))


# ---------------------------------------------------------------- workers
_W: dict = {}

_ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS")}


def default_workers() -> int:
    """All but two logical cores: the fits are pure CPU, one thread each."""
    return max(1, (os.cpu_count() or 2) - 2)


def _worker_init(folder):
    _W.clear()
    _W["folder"] = folder


def _problem() -> "Problem":
    """This worker's problem, loaded on its first job that needs it (the
    pool starts before there is one: `Workers.attach`)."""
    P = _W.get("P")
    if _W.get("folder") is not None:
        # re-attached (a moving camera's interval search): load it again
        st = os.stat(Path(_W["folder"]) / "problem.json")
        stamp = (st.st_mtime_ns, st.st_size)
        if P is not None and _W.get("stamp") != stamp:
            P = None
        _W["stamp"] = stamp
    if P is None:
        P = _W["P"] = Problem.load(_W["folder"], log=None)
    return P


class Workers:
    """Worker processes sharing one problem: each loads it once (the orbit
    table shared, or memory-mapped, so there's one copy) and then takes
    small jobs. Numpy in each worker runs one thread, so N workers use N
    cores rather than fighting over them.

    The processes start at once and import while the caller carries on;
    they can build the problem's orbit table (`build_problem(pool=...)`)
    before `attach` hands them the problem, which each loads on its first
    job that needs it."""

    def __init__(self, P: Problem | None = None, folder=None,
                 n: int | None = None):
        self.n = n or default_workers()
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        saved = {k: os.environ.get(k) for k in _ONE_THREAD}
        os.environ.update(_ONE_THREAD)
        try:
            self.ex = ProcessPoolExecutor(self.n, initializer=_worker_init,
                                          initargs=(str(self.folder),))
            # one job each spawns them all now, while the one-thread
            # environment is set; no waiting for them to finish importing
            for i in range(self.n):
                self.ex.submit(_noop, i)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        if P is not None:
            self.attach(P)

    def attach(self, P: Problem) -> None:
        """The problem the jobs work on (saved where the workers look)."""
        P.save(self.folder)

    def map(self, fn, jobs, chunksize=1):
        return self.ex.map(fn, jobs, chunksize=chunksize)

    def close(self):
        self.ex.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _noop(_):
    return None


class Threads:
    """`Workers` where a program can't start more of itself (Android:
    `par.processes()` false): threads in this process sharing the problem
    itself. The same jobs, the same answers; they overlap where numpy lets
    go of the GIL."""

    folder = None

    def __init__(self, P: Problem | None = None, n: int | None = None):
        from concurrent.futures import ThreadPoolExecutor
        self.n = n or default_workers()
        self.ex = ThreadPoolExecutor(self.n)
        if P is not None:
            self.attach(P)

    def attach(self, P: Problem) -> None:
        _W["P"] = P

    def map(self, fn, jobs, chunksize=1):
        return self.ex.map(fn, jobs)

    def close(self):
        self.ex.shutdown()
        _W.pop("P", None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pool(prefix: str, n: int | None = None):
    """The solve's workers: processes sharing the problem through `folder`
    (a new temp dir), or threads where there can't be processes."""
    from . import par
    if not par.processes():
        return Threads(n=n)
    import tempfile
    return Workers(folder=tempfile.mkdtemp(prefix=prefix), n=n)


def gpu_wanted() -> bool:
    """A GPU to use: there's one CuPy can drive, and SKYFIND_GPU isn't 0."""
    if os.environ.get("SKYFIND_GPU") == "0":
        return False
    from . import gpu
    return gpu.available()


def _gpu_orbits(P: Problem, log=print):
    """The orbit table on the GPU, when there is one with room for it (and
    SKYFIND_GPU isn't 0); else None."""
    if not gpu_wanted():
        return None
    from . import gpu
    import cupy as cp
    need = 2 * P.orbits.r.nbytes
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < need + (1 << 30):
        if log:
            log(f"  GPU: {free / 2**30:.1f} GB free, the orbit table needs "
                f"{need / 2**30:.1f} GB: searching on the CPU")
        return None
    return gpu.DeviceOrbits(P.orbits)


def _job_free(args):
    i, segs, part = args
    return i, free_passes(_problem(), segs, part=part)


def _job_near(args):
    i, segs, part = args
    return i, near_objects(_problem(), segs, part=part)


def _job_check(args):
    i, segs, near = args
    return i, check_passes(_problem(), segs, near)


def _object_batches(tracks, near: dict[int, dict], quanta: int) -> list:
    """(track, segments, {object: quanta}) jobs of about `quanta` clock
    hypotheses each, biggest tracks first."""
    jobs = []
    for i in sorted(near, key=lambda i: -sum(map(len, near[i].values()))):
        batch, size = {}, 0
        for j in sorted(near[i]):
            batch[j] = near[i][j]
            size += len(near[i][j])
            if size >= quanta:
                jobs.append((i, tracks[i], batch))
                batch, size = {}, 0
        if batch:
            jobs.append((i, tracks[i], batch))
    return jobs


def _job_fits(args):
    i, segs, cands = args
    return i, fit_many(_problem(), segs, cands)


def fit_tracks(P: Problem, tracks: list[list[Seg]], *, pool=None,
               batch: int | None = None, slices: int = 8, quanta: int = 400,
               on_track=None, dev=None, log=print) -> list[dict]:
    """fit_track for every track. With a `pool` (`Workers`) the work is cut
    into small jobs so one busy track can't keep the others waiting: the
    catalogue sweep in `slices` of each track's clock window, the pass
    checks in batches of ~`quanta` (object, clock) hypotheses, the fits in
    batches of `batch` candidates. Same result as without a pool.
    `on_track(i, result)` is called as each track finishes. `dev`: the
    orbit table already on the GPU (`_gpu_orbits`); without one, the GPU
    is used when there is one, for this call alone."""
    tic = time.time()
    if batch is None:               # SciPy fits are slow: small batches
        batch = 8 if os.environ.get("SKYFIND_SCIPY_FITS") == "1" else 1000

    def note(i, r):
        if on_track:
            on_track(i, r)
        if not log:
            return
        b = r["fits"][0] if r["fits"] else None
        desc = (f"{b['obj']:24s} med {b['med_err']:.3f} deg  lat "
                f"{b['lat']:6.2f} lon {b['lon']:7.2f} T {b['T']:+6.0f}s "
                f"E {b['E']:4.1f}s" if b else "no fit")
        log(f"  track {i:3d} frames {r['frames'][0]}-{r['frames'][-1]} "
            f"({r['n']}): {r['ncand']:4d} cands, "
            f"{len([f for f in r['fits'] if f['med_err'] < GOOD_FIT])} "
            f"good -> {desc} ({time.time() - tic:.0f}s)")

    if pool is None:
        out = []
        for i, t in enumerate(tracks):
            out.append(fit_track(P, t))
            note(i, out[-1])
        return out

    cands: dict[int, list] = {i: [] for i in range(len(tracks))}
    jobs = [(i, t, (p, slices)) for i, t in enumerate(tracks)
            for p in range(slices)]
    # the GPU, when there is one with room for the orbit table: the pass
    # search and every track's fits (not SciPy's, kept for checking)
    scipy = os.environ.get("SKYFIND_SCIPY_FITS") == "1"
    own = dev is None
    if own:
        dev = _gpu_orbits(P, log)
    try:
        if dev is not None:
            # on the GPU: each track's whole clock window at once (a free
            # site's, every track at once)
            from . import gpu
            if P.free:
                cands = dict(enumerate(gpu.free_passes_all(P, dev, tracks)))
            else:
                for i, t in enumerate(tracks):
                    cands[i] = gpu.skyline_passes(P, dev, t)
        elif P.free and native.available():
            # no GPU, but the same kernels compiled for this CPU (the phone)
            cands = dict(enumerate(native.free_passes_all(P, tracks)))
        elif P.free:
            # a free site: lines of sight to the ground, in slices
            for i, part in pool.map(_job_free, jobs):
                cands[i] += part
        else:
            # pass 1a: sweep the catalogue, each track's clock window in
            # slices
            near: dict[int, dict] = {i: {} for i in range(len(tracks))}
            for i, part in pool.map(_job_near, jobs):
                for j, Ts in part.items():
                    near[i].setdefault(j, set()).update(Ts)
            # pass 1b: check those (object, clock) hypotheses in batches
            for i, part in pool.map(_job_check,
                                    _object_batches(tracks, near, quanta)):
                cands[i] += part
        n = sum(len(c) for c in cands.values())
        gpu_fits = dev is not None and not scipy
        if log:
            log(f"  {n} candidate passes over {len(tracks)} tracks "
                f"({time.time() - tic:.0f}s"
                + (", on the GPU" if dev is not None else "")
                + "); fitting them "
                + ("on the GPU" if gpu_fits else f"on {pool.n} workers"))
        if gpu_fits:
            from . import gpu
            res = gpu.fit_all(P, dev, dict(enumerate(tracks)), cands)
            out = []
            for i in range(len(tracks)):
                out.append(track_result(tracks[i], len(cands[i]), res[i]))
                note(i, out[-1])
            return out
    finally:
        if own and dev is not None:
            dev.free()
    jobs = [(i, tracks[i], cands[i][a:a + batch])
            for i in range(len(tracks))
            for a in range(0, len(cands[i]), batch)]
    # the biggest tracks first, so their batches start early
    jobs.sort(key=lambda j: -len(cands[j[0]]))
    fits: dict[int, list] = {i: [] for i in range(len(tracks))}
    left = {i: -(-len(cands[i]) // batch) for i in range(len(tracks))}
    out: list = [None] * len(tracks)
    for i in range(len(tracks)):
        if not left[i]:
            out[i] = track_result(tracks[i], 0, [])
            note(i, out[i])
    for i, part in pool.map(_job_fits, jobs):
        fits[i] += part
        left[i] -= 1
        if not left[i]:
            out[i] = track_result(tracks[i], len(cands[i]), fits[i])
            note(i, out[i])
    return out


# ====================================================================== cluster
def orbit_families(P: Problem) -> dict[int, tuple]:
    """{norad: (orbit-plane normal, radius km)} at the middle of the clock
    window. Objects in one plane at one height (a Starlink shell's plane,
    a train) pass the same places at shifted times, so for a shifted clock
    and site they fit each other's tracks: their agreement isn't
    independent evidence."""
    i = len(P.orbits.r) // 2
    r = P.orbits.r[i].astype(float)
    n = np.cross(r, P.orbits.v[i].astype(float))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    rad = np.linalg.norm(r, axis=1)
    return {int(k): (n[j], float(rad[j])) for j, k in enumerate(P.norad)
            if np.isfinite(rad[j])}


class _Distinct:
    """How many independent objects among some: distinct catalogue numbers,
    and with `orbits` (`orbit_families`) distinct planes-and-heights (in
    catalogue order, each counted unless its plane and height match one
    already counted). Remembered by the set of objects: the fits of one
    cluster share their members."""

    def __init__(self, orbits: dict | None, plane_deg=1.5, height_km=25.0):
        self.orbits = orbits
        self.cos = math.cos(math.radians(plane_deg))
        self.height_km = height_km
        self.memo: dict = {}
        self.plain: dict = {}                    # norad -> (nx, ny, nz, r)

    def _orbit(self, k):
        o = self.plain.get(k)
        if o is None and k not in self.plain:
            got = self.orbits.get(k)
            o = None if got is None else \
                (float(got[0][0]), float(got[0][1]), float(got[0][2]),
                 float(got[1]))
            self.plain[k] = o
        return o

    def __call__(self, norads) -> int:
        key = tuple(sorted(set(norads)))
        n = self.memo.get(key)
        if n is not None:
            return n
        if not self.orbits:
            n = len(key)
        else:
            cos, hk = self.cos, self.height_km
            seen: list = []
            n = 0
            for k in key:
                o = self._orbit(k)
                if o is not None and any(
                        o[0] * s[0] + o[1] * s[1] + o[2] * s[2] > cos and
                        abs(o[3] - s[3]) < hk for s in seen):
                    continue
                n += 1
                if o is not None:
                    seen.append(o)
        self.memo[key] = n
        return n


def _distinct(members, orbits: dict | None, plane_deg=1.5, height_km=25.0
              ) -> int:
    """How many independent objects: distinct catalogue numbers, and with
    `orbits` (`orbit_families`) distinct planes-and-heights."""
    return _Distinct(orbits, plane_deg, height_km)(
        m["norad"] for m in members)


def _median(xs) -> float:
    """np.median's value for a short list, without its overhead."""
    s = sorted(xs)
    h = len(s) // 2
    return s[h] if len(s) % 2 else (s[h - 1] + s[h]) / 2


def cluster(track_fits: list[dict], *, keep=GOOD_FIT, dT=40.0, dlat=0.6,
            dlon=0.8, n_max=6, orbits: dict | None = None) -> list[dict]:
    """Where fits of distinct objects coincide in (T, lat, lon) — distinct
    orbits when `orbits` (`orbit_families`) is given. Each good fit
    gathers, from each track, the best fit near it (within dT, dlat,
    dlon); the fits are ranked by how many distinct objects, then tracks,
    they gather (then their median error), and the best not near a better
    one are the clusters. In arrays: the fits binned in (dT, dlat, dlon)
    cells, each meeting only its neighbours' (a free site leaves 10^5 good
    fits)."""
    fits = [(ti, f) for ti, r in enumerate(track_fits) for f in r["fits"]
            if f["med_err"] < keep]
    n = len(fits)
    if not n:
        return []
    T = np.array([f["T"] for _, f in fits], float)
    LA = np.array([f["lat"] for _, f in fits], float)
    LO = np.array([f["lon"] for _, f in fits], float)
    ER = np.array([f["med_err"] for _, f in fits], float)
    NO = np.array([f["norad"] for _, f in fits], np.int64)
    TR = np.array([ti for ti, _ in fits], np.int64)
    # ---- every (fit, neighbour) pair within a cell of each other
    n_lon = math.ceil(360 / dlon)
    cT = np.floor(T / dT).astype(np.int64)
    ca = np.floor(LA / dlat).astype(np.int64)
    co = np.floor(((LO + 180) % 360) / dlon).astype(np.int64)
    cT = cT - cT.min() + 1
    ca = ca - ca.min() + 1
    nA = int(ca.max()) + 2
    key = (cT * nA + ca) * n_lon + co
    order = np.argsort(key, kind="stable")
    skey = key[order]
    I, K = [], []
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nk = ((cT + da) * nA + ca + db) * n_lon + (co + dc) % n_lon
                lo = np.searchsorted(skey, nk, "left")
                cnt = np.searchsorted(skey, nk, "right") - lo
                tot = int(cnt.sum())
                if not tot:
                    continue
                I.append(np.repeat(np.arange(n), cnt))
                K.append(order[np.repeat(lo - np.cumsum(cnt) + cnt, cnt)
                               + np.arange(tot)])
    I, K = np.concatenate(I), np.concatenate(K)
    ok = (np.abs(T[I] - T[K]) <= dT) & (np.abs(LA[I] - LA[K]) <= dlat) & \
        (np.abs((LO[I] - LO[K] + 180) % 360 - 180) <= dlon)
    I, K = I[ok], K[ok]
    # ---- each fit's members: the best fit per track near it (the first of
    # equals, in fit order)
    o = np.lexsort((K, ER[K], TR[K], I))
    I, K = I[o], K[o]
    first = np.r_[True, (I[1:] != I[:-1]) | (TR[K[1:]] != TR[K[:-1]])]
    MI, MK = I[first], K[first]
    ntr = np.bincount(MI, minlength=n)
    # their median error
    o = np.lexsort((ER[MK], MI))
    v = ER[MK][o]
    start = np.searchsorted(MI[o], np.arange(n))
    h = ntr // 2
    odd = ntr % 2 == 1
    med = np.where(odd, v[np.minimum(start + h, len(v) - 1)],
                   (v[np.maximum(start + h - 1, 0)] +
                    v[np.minimum(start + h, len(v) - 1)]) / 2)
    # how many independent objects: once per distinct set of members
    o = np.lexsort((NO[MK], MI))
    gi, gn = MI[o], NO[MK][o]
    keep_ = np.r_[True, (gi[1:] != gi[:-1]) | (gn[1:] != gn[:-1])]
    gi, gn = gi[keep_], gn[keep_]
    bounds = np.searchsorted(gi, np.arange(n + 1))
    distinct = _Distinct(orbits)
    memo: dict = {}
    nobj = np.empty(n, np.int64)
    for i in range(n):
        kb = gn[bounds[i]:bounds[i + 1]].tobytes()
        c = memo.get(kb)
        if c is None:
            c = memo[kb] = distinct(gn[bounds[i]:bounds[i + 1]].tolist())
        nobj[i] = c
    # ---- ranked; the best not near a better one
    rank = np.lexsort((np.arange(n), med, -ntr, -nobj))
    near_at = np.searchsorted(I, np.arange(n + 1))      # I sorted: pairs
    by_fit = np.searchsorted(MI, np.arange(n + 1))

    def near(a, b, sT, sa, so):
        dl = (LO[a] - LO[b] + 180) % 360 - 180
        return abs(T[a] - T[b]) <= sT and abs(LA[a] - LA[b]) <= sa and \
            abs(dl) <= so

    seen, out = [], []
    for i in rank.tolist():
        if any(near(i, s_, 3 * dT, 3 * dlat, 3 * dlon) for s_ in seen):
            continue
        seen.append(i)
        best = {int(TR[k]): int(k) for k in MK[by_fit[i]:by_fit[i + 1]]}
        # tracks in the order their first near fit comes (fit order)
        ks = np.sort(K[near_at[i]:near_at[i + 1]])
        members: dict[int, int] = {}
        for k in ks.tolist():
            t = int(TR[k])
            if t not in members:
                members[t] = best[t]
        ms = [fits[k][1] for k in members.values()]
        out.append(dict(
            n_objects=int(nobj[i]), n_tracks=int(ntr[i]),
            lat=float(np.median([m["lat"] for m in ms])),
            lon=float(np.median([m["lon"] for m in ms])),
            T=float(np.median([m["T"] for m in ms])),
            members=[dict(track=t, **fits[k][1])
                     for t, k in members.items()]))
        if len(out) >= n_max:
            break
    return out


# ====================================================================== joint
def join_orders(track_fits: list[dict]) -> list[list[int]]:
    """Each track's fitted objects, in order, once each: all `joint` needs
    of the track fits (and far less to send to a worker)."""
    return [list(dict.fromkeys(f["j"] for f in r["fits"]))
            for r in track_fits]


def joint(P: Problem, tracks: list[list[Seg]], track_fits: list[dict] | None,
          top: dict, *, orders=None, join_err=0.15, height=True, pool=None,
          dev=None, log=print) -> dict:
    """Fit the top cluster's tracks together, let other tracks join when
    one of their fitted objects agrees, refit; -> solution dict with its
    error bars (`joint_errors`; with a `pool` their refits run side by
    side; with `dev`, the orbit table on the GPU, the join step runs
    there). `orders` (`join_orders`) can stand in for `track_fits`."""
    sol, state = joint_fit(P, tracks, track_fits, top, orders=orders,
                           join_err=join_err, height=height, dev=dev,
                           log=log)
    return joint_errors(P, sol, state, pool=pool)


def joint_fit(P: Problem, tracks: list[list[Seg]],
              track_fits: list[dict] | None, top: dict, *, orders=None,
              join_err=0.15, height=True, dev=None,
              log=print) -> tuple[dict, dict]:
    """`joint` with the formal error bars only: -> (solution, what
    `joint_errors` needs). A candidate site that may lose needs no more.
    In three steps, which the tie-break runs for many sites at once:
    `joint_start`, `join_tracks` (or `gpu.join_tracks`), `joint_finish`."""
    if orders is None:
        orders = join_orders(track_fits)
    start = joint_start(P, tracks, top)
    if dev is not None:
        from . import gpu
        joined = gpu.join_tracks(P, dev, tracks, orders, [start],
                                 join_err)[0]
    else:
        joined = join_tracks(P, tracks, orders, start, join_err)
    return joint_finish(P, tracks, start, joined, height=height, log=log)


def joint_start(P: Problem, tracks: list[list[Seg]], top: dict):
    """A cluster's member tracks fitted together: -> ({track: Assoc}, p)."""
    ms = top["members"]
    p0 = [float(np.median([m["slat"] for m in ms])),
          float(np.median([m["dlst"] for m in ms])),
          float(np.median([m["T"] for m in ms])),
          float(np.median([m["E"] for m in ms]))]

    def assoc(ti, j, p):
        return Assoc(j, tracks[ti], end_order(P, j, tracks[ti], p))

    sets = {m["track"]: assoc(m["track"], m["j"], p0) for m in ms}
    p, _, _ = fit(P, list(sets.values()), p0, f_scale=0.1,
                  x_scale=(0.1, 0.1, 5, 2, 0.5))
    return sets, [float(x) for x in p]


def join_tracks(P: Problem, tracks: list[list[Seg]], orders, start,
                join_err=0.15) -> dict:
    """The tracks outside `start`'s cluster that join it: each one's fitted
    objects tried in order at the start's solution, the first agreeing
    within `join_err` (median end error, deg) taken. -> {track: (object,
    end order, median)}."""
    from .batchfit import first_agreeing
    sets, p = start
    out = {}
    for ti, J in enumerate(orders):
        if ti in sets or not J:
            continue
        hit = first_agreeing(P, tracks[ti], J, np.asarray(p), join_err)
        if hit is not None:
            out[ti] = hit
    return out


def joint_finish(P: Problem, tracks: list[list[Seg]], start, joined: dict,
                 *, height=True, log=print) -> tuple[dict, dict]:
    """`joint_fit` from its start and the tracks that joined: refit, leave
    out ends cut short, refit, the height if the trails fix it."""
    sets, p = start
    sets = dict(sets)
    for ti in sorted(joined):
        j, af, med = joined[ti]
        sets[ti] = Assoc(j, tracks[ti], af)
        if log:
            log(f"  + {P.names[j]:24s} track {ti} frames "
                f"{tracks[ti][0].k}-{tracks[ti][-1].k} agrees "
                f"(median {med:.3f} deg)")
    A = list(sets.values())
    p, sig, errs = fit(P, A, p, f_scale=0.1, x_scale=(0.1, 0.1, 5, 2, 0.5))
    # an end far off the joint fit was cut short (the Earth's shadow, a
    # fade), not by the exposure: drop it and refit
    lim = max(4 * float(np.median(errs)), 0.1)
    n_cut = 0
    for a in A:
        with np.errstate(invalid="ignore"):
            bad = a.end_errors(P, p) > lim
        if bad.any():
            a.cut = bad if a.cut is None else a.cut | bad
            n_cut += int(bad.sum())
    if n_cut:
        if log:
            log(f"  {n_cut} trail ends cut short (not at an exposure "
                f"boundary): left out")
        p, sig, errs = fit(P, A, p, f_scale=0.1,
                           x_scale=(0.1, 0.1, 5, 2, 0.5))
    kmid = int(np.median([s.k for a in A for s in a.segs]))

    def where(q, h):
        return P.site.geodetic(kmid, q[0], q[1], h,
                               float(P.tk(kmid)) + q[2] / 86400)

    fitted_h = False
    if height:
        ph, sigh, errh = fit(P, A, list(p) + [P.h0], height=True,
                             f_scale=0.1, x_scale=(0.1, 0.1, 5, 2, 0.5))
        # keep the height only when the trails really constrain it (a fit
        # resting on a bound or with a loose sigma is noise)
        if np.isfinite(sigh[4]) and sigh[4] < 0.3 and -0.4 < ph[4] < 8.9:
            p, sig, errs, fitted_h = ph, sigh, errh, True
    h = float(p[4]) if fitted_h else P.h0
    lat, lon = where(p, h)
    # longitude: dlst and T both move it (T at the sidereal rate)
    sig_lat = float(sig[0])
    sig_lon = math.hypot(sig[1], sig[2] * 360.98564736629 / 86400)
    anchors = []
    for ti, a in sets.items():
        e = np.linalg.norm(residuals(P, [a], p).reshape(-1, 3), axis=1)
        anchors.append(dict(track=int(ti), name=P.names[a.j],
                            norad=int(P.norad[a.j]),
                            frames=[s.k for s in a.segs],
                            med_err=float(np.median(e)) if len(e) else None))
    sol = dict(lat=lat, lon=lon, height_km=h, height_fitted=fitted_h,
               sig_lat=sig_lat, sig_lon=float(sig_lon),
               sig_height_km=float(sig[4]) if fitted_h else 1.0,
               height_sys_deg=[0.0, 0.0], jackknife_deg=None,
               slat=float(p[0]), dlst=float(p[1]), T=float(p[2]),
               E=float(p[3]), sig_T=float(sig[2]), sig_E=float(sig[3]),
               spf=P.spf, frame0_utc=P.utc(float(p[2])),
               frame0_tt=P.t_ref + p[2] / 86400,
               median_err_deg=float(np.median(errs)),
               n_objects=len({a["norad"] for a in anchors}),
               anchors=anchors)
    return sol, dict(A=A, p=list(p), fitted_h=fitted_h, h=h, kmid=kmid)


def joint_errors(P: Problem, sol: dict, state: dict, *, pool=None) -> dict:
    """`joint_fit`'s solution with its full error bars: the formal errors
    assume every anchor fits the same model perfectly, so leave-one-out
    (jackknife) says how much the answer rests on any one; and with the
    height assumed, how far +-1 km moves the site. The refits run side by
    side on a `pool`. -> the solution, updated."""
    A, p, fitted_h, h, kmid = (state[k] for k in
                               ("A", "p", "fitted_h", "h", "kmid"))
    lon = sol["lon"]
    sig_lat, sig_lon = sol["sig_lat"], sol["sig_lon"]
    jobs = [(A[:i] + A[i + 1:], p, fitted_h, None, None if fitted_h else h,
             kmid) for i in range(len(A))] if len(A) >= 4 else []
    if not fitted_h:
        jobs += [(A, p[:4], False, h + dh, h + dh, kmid)
                 for dh in (-1.0, 1.0)]
    refits = list(pool.map(_job_refit, jobs)) if pool is not None else \
        [_refit(P, *job) for job in jobs]
    jack = None
    if len(A) >= 4:
        pts = np.array(refits[:len(A)])
        pts[:, 1] = lon + (pts[:, 1] - lon + 180) % 360 - 180
        m = len(pts)
        jack = np.sqrt((m - 1) / m * ((pts - pts.mean(0)) ** 2).sum(0))
        sig_lat, sig_lon = max(sig_lat, jack[0]), max(sig_lon, jack[1])
    h_sys = (0.0, 0.0)
    if not fitted_h:
        lo_hi = refits[-2:]
        h_sys = (abs(lo_hi[1][0] - lo_hi[0][0]) / 2,
                 abs((lo_hi[1][1] - lo_hi[0][1] + 180) % 360 - 180) / 2)
        sig_lat = math.hypot(sig_lat, h_sys[0])
        sig_lon = math.hypot(sig_lon, h_sys[1])
    sol.update(sig_lat=float(sig_lat), sig_lon=float(sig_lon),
               height_sys_deg=list(h_sys),
               jackknife_deg=None if jack is None else jack.tolist())
    return sol


def _refit(P: Problem, A, p, height, h0, h_where, kmid):
    """One of `joint`'s error refits: `A` from `p` (the height fitted, or
    held at `h0`), -> where it puts the site at frame `kmid`, with height
    `h_where` (None: the fitted one)."""
    keep = P.h0
    try:
        if h0 is not None:
            P.h0 = h0
        q, _, _ = fit(P, A, p, height=height, f_scale=0.1,
                      x_scale=(0.1, 0.1, 5, 2, 0.5))
    finally:
        P.h0 = keep
    hw = float(q[4]) if h_where is None else h_where
    return P.site.geodetic(kmid, q[0], q[1], hw,
                           float(P.tk(kmid)) + q[2] / 86400)


def _job_refit(args):
    return _refit(_problem(), *args)


# ====================================================================== identify
# `identify`'s times (s from the clock's mid-exposure): coarse ones for the
# objects near a segment, fine ones for how close each passes its ends
_ID_COARSE = np.arange(-25, 26, 5.0)
_ID_FINE = np.arange(-25, 25.25, 0.5)


def identify(P: Problem, segs: list[Seg], sol: dict, *, T=None, tol=0.25,
             centre_s=8.0, prefilter_deg=8.0, pool=None, dev=None,
             first: int = 0, chunk: int = 12) -> list[dict]:
    """The best-fitting object for each segment at the solution: it must
    pass both ends within `tol` deg, inside this exposure (centred within
    `centre_s`, no longer than it). Ends cut short relax the centring.
    With a `pool`, segments go out in chunks of `chunk`; with `dev` (the
    orbit table on the GPU) it's done there."""
    if pool is not None or dev is not None:
        return identify_all(P, segs, [sol if T is None else dict(sol, T=T)],
                            tol=tol, centre_s=centre_s,
                            prefilter_deg=prefilter_deg, pool=pool, dev=dev,
                            chunk=chunk)[0]
    lat, dlst, E = sol["slat"], sol["dlst"], sol["E"]
    h = sol.get("height_km", P.h0)
    T = sol["T"] if T is None else T
    ss = _ID_FINE
    cos_pre = math.cos(math.radians(prefilter_deg))
    out = []
    for i, s in enumerate(segs):
        tk = float(P.tk(s.k))
        rs = P.site.gcrs(s.k, lat, dlst, h)
        mid = unit(s.a + s.b)
        cand = set()
        for dt in _ID_COARSE:
            d = P.orbits.at(tk + (T + dt) / 86400) - rs
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            cand |= set(np.flatnonzero(d @ mid > cos_pre).tolist())
        best = _best_pass(P, s, np.array(sorted(cand), np.int64), ss, T,
                          lat, dlst, h, E, tol, centre_s)
        if best:
            j, err, tm, span = best
            out.append(dict(seg=first + i, k=s.k, name=P.names[j],
                            norad=int(P.norad[j]), err_deg=round(err, 4),
                            dt=float(tm), span=float(span),
                            utc=P.utc(T, s.k)[11:],
                            clipped=bool(s.clip_a or s.clip_b)))
    return out


def _best_pass(P: Problem, s: Seg, J: np.ndarray, ss: np.ndarray, T, lat,
               dlst, h, E, tol, centre_s):
    """Of objects `J`, the one passing segment `s` best at clock T: both
    ends within `tol` deg, at times inside this exposure (centred within
    `centre_s`, no longer than it; ends cut short relax the centring).
    All of them at once, over the times `ss` (s from mid-exposure).
    -> (j, err, mid time, span) or None."""
    if not J.size:
        return None
    n = len(ss)
    d = P.dirs(J[:, None], np.full((len(J), n), float(s.k)),
               np.broadcast_to(ss, (len(J), n)), T, lat, dlst, h)
    finite = np.isfinite(d).all((1, 2))
    ca = np.nan_to_num(d @ s.a, nan=-2.0)
    cb = np.nan_to_num(d @ s.b, nan=-2.0)
    ia, ib = ca.argmax(1), cb.argmax(1)
    rows = np.arange(len(J))
    err = np.maximum(_deg(ca[rows, ia]), _deg(cb[rows, ib]))
    tm, span = (ss[ia] + ss[ib]) / 2, np.abs(ss[ib] - ss[ia])
    if s.clip_a or s.clip_b:
        inside = np.maximum(np.abs(ss[ia]), np.abs(ss[ib])) <= E / 2 + 3
    else:
        inside = np.abs(tm) <= centre_s
    ok = finite & (ia > 0) & (ia < n - 1) & (ib > 0) & (ib < n - 1) & \
        (err < tol) & inside & (span <= E + 3)
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[np.argmin(err[ok])])
    return int(J[k]), float(err[k]), float(tm[k]), float(span[k])


def objects(ids: list[dict]) -> list[dict]:
    """Group segment IDs by object with a confidence tier: confirmed when
    seen in several frames or fitted tightly (< 0.12 deg, centred < 3 s)."""
    by: dict = {}
    for d in ids:
        by.setdefault((d["name"], d["norad"]), []).append(d)
    rows = []
    for (name, no), ds in by.items():
        ds.sort(key=lambda d: d["k"])
        multi = len({d["k"] for d in ds}) >= 2
        tight = any(d["err_deg"] < 0.12 and abs(d["dt"]) < 3 for d in ds)
        rows.append(dict(name=name, norad=no, frames=[d["k"] for d in ds],
                         segs=[d["seg"] for d in ds], utc=ds[0]["utc"],
                         err=float(np.median([d["err_deg"] for d in ds])),
                         conf="confirmed" if multi or tight else "probable"))
    rows.sort(key=lambda r: (r["frames"][0], r["name"]))
    for i, r in enumerate(rows, 1):
        r["n"] = i
    return rows


def identify_all(P: Problem, segs: list[Seg], sols: list[dict], *,
                 tol=0.25, centre_s=8.0, prefilter_deg=8.0, pool=None,
                 dev=None, chunk: int = 12) -> list[list[dict]]:
    """`identify` at each of several solutions (sites, clocks) at once: on
    the GPU with `dev`, else over the `pool` in one go (every solution's
    segments mixed, so no core waits for one solution's last chunk).
    -> one list of IDs per solution."""
    kw = dict(tol=tol, centre_s=centre_s, prefilter_deg=prefilter_deg)
    if dev is not None:
        from . import gpu
        return gpu.identify(P, dev, segs, sols, **kw)
    if pool is None:
        return [identify(P, segs, s, **kw) for s in sols]
    jobs = [(q, a, segs[a:a + chunk], s, kw) for q, s in enumerate(sols)
            for a in range(0, len(segs), chunk)]
    out: list[list[dict]] = [[] for _ in sols]
    for q, part in pool.map(_job_identify, jobs):
        out[q] += part
    return out


def _job_identify(args):
    q, first, segs, sol, kw = args
    return q, identify(_problem(), segs, sol, first=first, **kw)


def _job_joint_start(args):
    tracks, top = args
    try:
        return joint_start(_problem(), tracks, top)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def _job_join(args):
    tracks, orders, start = args
    return join_tracks(_problem(), tracks, orders, start)


def _job_joint_finish(args):
    tracks, start, joined = args
    try:
        return joint_finish(_problem(), tracks, start, joined, log=None)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def null_rate(P: Problem, segs: list[Seg], sol: dict,
              offsets=(-600, -300, 300, 600), pool=None, dev=None) -> dict:
    """Matches with the clock deliberately wrong: the chance baseline."""
    out = {}
    wrong = identify_all(P, segs, [dict(sol, T=sol["T"] + off)
                                   for off in offsets], pool=pool, dev=dev)
    for off, ids in zip(offsets, wrong):
        out[str(off)] = dict(
            segments=len(ids),
            tight=sum(d["err_deg"] < 0.12 and abs(d["dt"]) < 3 for d in ids))
    return out
