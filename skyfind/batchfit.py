"""Batched track fits: every candidate object for one track fitted at once.

`solve.fit_candidate` fits one candidate with SciPy's `least_squares`: a
fraction of a millisecond of arithmetic under tens of milliseconds of
Python overhead, and a solve fits ~10^5-10^6 candidates. Every candidate of
a track has the same unknowns (lat, dlst, T, E) and the same observations
(the track's segment ends), so here they share one set of arrays
(candidates x ends x 3) and one Levenberg-Marquardt loop: the same robust
loss (soft L1), variable scales, bounds and finite-difference steps as
`solve.fit`, each candidate with its own damping, stopping on its own.
"""

from __future__ import annotations

import numpy as np

from .solve import GOOD_FIT, STEPS, Problem, Seg

X_SCALE = np.array([0.5, 0.5, 20.0, 5.0])      # as `solve.fit`
F_SCALE = 0.15                                 # deg, soft-L1 knee
MAX_ITER = 60
FTOL = 1e-10


def _observed(segs: list[Seg], af: np.ndarray):
    """Per candidate (its end order `af`, (C, n) bool): observed ends
    (C, 2n, 3) in the order the object passes them, and which are free
    (not cut short by the frame edge, the skyline or the Earth's shadow)."""
    a = np.array([s.a for s in segs])
    b = np.array([s.b for s in segs])
    fa = np.array([not s.clip_a for s in segs])
    fb = np.array([not s.clip_b for s in segs])
    first = np.where(af[..., None], a, b)
    second = np.where(af[..., None], b, a)
    obs = np.stack([first, second], 2).reshape(len(af), -1, 3)
    free = np.stack([np.where(af, fa, fb), np.where(af, fb, fa)], 2)
    return obs, free.reshape(len(af), -1)


class Batch:
    """One track's candidates: objects `j` (C,), end orders (C, n). (The
    GPU's is `gpu.FitBatch`: all tracks' candidates at once; `fit` takes
    either.)"""

    xp = np

    def __init__(self, P: Problem, segs: list[Seg], j, af):
        self.P, self.segs = P, segs
        self.j = np.asarray(j, np.int64)
        self.obs, self.free = _observed(segs, np.asarray(af, bool))
        self.ks = np.repeat([float(s.k) for s in segs], 2)

    def residuals(self, p: np.ndarray, idx=None) -> np.ndarray:
        """Degrees, (C, 2n, 3): predicted minus observed direction of each
        end; zero at ends that aren't free (as `solve.residuals` leaves
        them out)."""
        sl = slice(None) if idx is None else idx
        j, obs, free = self.j[sl], self.obs[sl], self.free[sl]
        C = len(j)
        ks = np.broadcast_to(self.ks, (C, len(self.ks)))
        offs = np.empty(ks.shape)
        offs[:, 0::2] = -p[:, 3:4] / 2
        offs[:, 1::2] = p[:, 3:4] / 2
        d = self.P.dirs(j[:, None], ks, offs, p[:, 2:3], p[:, 0:1],
                        p[:, 1:2])
        r = np.degrees(d - obs)
        r[~free] = 0.0
        return r


def _cost(r, xp=np):
    """Soft-L1 cost per candidate (x 2 / f_scale^2, as least_squares')."""
    z = (r.reshape(len(r), -1) / F_SCALE) ** 2
    return (xp.sqrt(1 + z) - 1).sum(1)


def fit(batch, p0, lo, hi):
    """Levenberg-Marquardt on every candidate at once (iteratively
    reweighted for the soft-L1 loss; Marquardt's scaling, floored by the
    variables' scales; steps clipped to the bounds). -> (p, residuals,
    ok): ok is False where the start wasn't even finite. The arrays are
    numpy or CuPy, as the batch's (`batch.xp`); every candidate has its
    own damping and stops on its own, so how they're batched never changes
    a result."""
    xp = batch.xp
    p = xp.clip(p0, lo + 1e-6, hi - 1e-6)
    r = batch.residuals(p)
    ok = xp.isfinite(r).all((1, 2))
    cost = xp.where(ok, _cost(xp.nan_to_num(r), xp), xp.inf)
    lam = xp.full(len(p), 1e-3)
    live = ok.copy()
    jac = None
    need = live.copy()
    xs = xp.asarray(X_SCALE)
    floor = 1e-9 / xs ** 2
    steps = [float(v) for v in STEPS[:4]]
    eye = xp.eye(4)
    for _ in range(MAX_ITER):
        if not bool(live.any()):
            break
        if jac is None:
            jac = xp.zeros(r.shape[:1] + (r[0].size, 4))
        idx = xp.flatnonzero(need & live)
        if idx.size:
            f0 = r[idx].reshape(len(idx), -1)
            for i in range(4):
                q = p[idx].copy()
                q[:, i] += steps[i]
                jac[idx, :, i] = (batch.residuals(q, idx).reshape(len(idx), -1)
                                  - f0) / steps[i]
            need[idx] = False
        idx = xp.flatnonzero(live)
        f = r[idx].reshape(len(idx), -1)
        J = jac[idx]
        w = 1 / xp.sqrt(1 + (f / F_SCALE) ** 2)           # soft-L1 weights
        Jw = J * w[..., None]
        A = Jw.transpose(0, 2, 1) @ J
        g = (Jw * f[..., None]).sum(1)
        dA = xp.maximum(xp.einsum("cii->ci", A), floor)
        M = A + (lam[idx, None] * dA)[:, :, None] * eye
        try:
            step = -xp.linalg.solve(M, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = -xp.einsum("cij,cj->ci", xp.linalg.pinv(M), g)
        step = xp.nan_to_num(step)
        pn = xp.clip(p[idx] + step, lo[idx], hi[idx])
        rn = batch.residuals(pn, idx)
        cn = xp.where(xp.isfinite(rn).all((1, 2)),
                      _cost(xp.nan_to_num(rn), xp), xp.inf)
        better = cn < cost[idx]
        a = idx[better]
        gain = cost[a] - cn[better]
        p[a], r[a], cost[a] = pn[better], rn[better], cn[better]
        need[a] = True
        lam[a] = xp.maximum(lam[a] / 3, 1e-12)
        b = idx[~better]
        lam[b] *= 8
        # stop: a step that barely helps, or one too small to matter
        moved = xp.abs(step[better] / xs).max(1) if a.size else \
            xp.zeros(0)
        done_a = (gain <= FTOL * (1 + cost[a])) | (moved < 1e-9)
        live[a[done_a]] = False
        live[b[lam[b] > 1e10]] = False
    return p, r, ok


def fit_candidates(P: Problem, segs: list[Seg], cands) -> list[dict | None]:
    """`solve.fit_candidate` for many candidates of one track at once."""
    from skyfield.api import wgs84
    from skyfield.positionlib import Geocentric

    from . import data
    from .site import AU_KM
    cands = list(cands)
    out: list[dict | None] = [None] * len(cands)
    if not cands:
        return out
    j = np.array([c[0] for c in cands])
    af = np.array([c[3] for c in cands], bool)
    batch = Batch(P, segs, j, af)
    use = batch.free.any(1)
    if not use.any():
        return out
    keep = np.flatnonzero(use)
    batch = Batch(P, segs, j[keep], af[keep])
    p0 = np.array([[c[4], c[5], c[1], min(max(c[2], 0.5), P.spf)]
                   for c in (cands[i] for i in keep)], float)
    lo = np.column_stack([np.full(len(keep), -89.9), p0[:, 1] - 12,
                          np.full(len(keep), -P.twin - 120),
                          np.full(len(keep), 0.2)])
    hi = np.column_stack([np.full(len(keep), 89.9), p0[:, 1] + 12,
                          np.full(len(keep), P.twin + 120),
                          np.full(len(keep), P.spf)])
    p, r, ok = fit(batch, p0, lo, hi)
    errs = np.linalg.norm(r, axis=2)
    errs[~batch.free] = np.nan
    med = np.nanmedian(errs, axis=1)
    mx = np.nanmax(errs, axis=1)
    # the site at the track's first frame, on the ellipsoid, when it is
    k1 = segs[0].k
    rs = P.site.gcrs(k1, p[:, 0], p[:, 1], P.h0)
    t = data.times(float(P.tk(k1)) + p[:, 2] / 86400)
    g = wgs84.geographic_position_of(Geocentric(rs.T / AU_KM, t=t))
    glat = np.atleast_1d(g.latitude.degrees)
    glon = np.atleast_1d(g.longitude.degrees)
    for n, i in enumerate(keep):
        if not ok[n] or not np.isfinite(med[n]):
            continue
        jj = int(j[i])
        out[i] = dict(obj=P.names[jj], norad=int(P.norad[jj]), j=jj,
                      lat=float(glat[n]), lon=float(glon[n]),
                      slat=float(p[n, 0]), dlst=float(p[n, 1]),
                      T=float(p[n, 2]), E=float(p[n, 3]),
                      med_err=float(med[n]), max_err=float(mx[n]))
    return out


def end_orders(P: Problem, J, segs: list[Seg], p, span=60.0) -> np.ndarray:
    """`solve.end_order` for objects `J` at solution `p`: (len(J), n) bool,
    does object J pass segment i's end a before end b."""
    ss = np.arange(-span, span + 0.25, 0.5)
    J = np.asarray(J, np.int64)
    h = p[4] if len(p) > 4 else None
    out = np.zeros((len(J), len(segs)), bool)
    for i, s in enumerate(segs):
        d = P.dirs(J[:, None], np.full((len(J), len(ss)), float(s.k)),
                   np.broadcast_to(ss, (len(J), len(ss))), p[2], p[0], p[1],
                   h)
        out[:, i] = np.argmax(d @ s.a, 1) < np.argmax(d @ s.b, 1)
    return out


def first_agreeing(P: Problem, segs: list[Seg], J, p, join_err: float):
    """The first of objects `J` (in order) whose passes fit the track at
    solution `p` with a median end error under `join_err` deg, as
    `solve.joint` tries them one by one: -> (j, end order, median) or
    None."""
    J = np.asarray(J, np.int64)
    af = end_orders(P, J, segs, p)
    batch = Batch(P, segs, J, af)
    q = np.broadcast_to(np.asarray(p[:4], float), (len(J), 4))
    with np.errstate(invalid="ignore"):
        e = np.linalg.norm(batch.residuals(q), axis=2)
        e[~batch.free] = np.nan
        ok = batch.free.any(1)
        med = np.full(len(J), np.inf)
        med[ok] = np.nanmedian(e[ok], axis=1)
    good = np.flatnonzero(med < join_err)
    if not good.size:
        return None
    k = int(good[0])
    return int(J[k]), [bool(x) for x in af[k]], float(med[k])


__all__ = ["Batch", "fit", "fit_candidates", "first_agreeing", "end_orders",
           "GOOD_FIT"]
