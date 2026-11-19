"""The free-site pass search as kernels over the whole clock window (what
`solve.free_passes` does a row at a time): the tracks' setup and the
candidates from the kernels' survivors, shared by the two backends — the
GPU's CUDA kernels (`gpu`) and the same kernels in C++ threads (`native`,
the phone). A backend has `xp` (its array module), `get(a)` (to numpy) and
`stage1`, `exact`, `refine` (see `gpu._STAGE1`, `_EXACT`, `_REFINE`)."""

from __future__ import annotations

import math

import numpy as np

from .site import A_WGS84, F_WGS84, WE

E2 = F_WGS84 * (2 - F_WGS84)


def _rotation(axis, ang):
    """Matrix M with rotate(v, axis, ang) == v @ M.T (Rodrigues)."""
    a = np.asarray(axis, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * math.cos(ang) + math.sin(ang) * K + \
        (1 - math.cos(ang)) * np.outer(a, a)


def run(P, tracks, backend, *, n_t: int, step: float, t0: float,
        rate_factor=1.25, dir_deg=12.0, tol=0.3, refine_s=2.0, fine_s=0.25,
        merge_s=60.0, sun_max_alt=-6.0) -> list[list]:
    """`solve.free_passes` for every track at once: the first tests over the
    whole clock window (each object's position read once for all the
    tracks) in single precision with margins, the survivors checked in
    double precision, then refined, as on the CPU. -> each track's
    candidates."""
    from .camera import unit
    from .date import sun_vectors
    from .solve import _deg, _mid
    xp = backend.xp
    dark = math.sin(math.radians(sun_max_alt))
    cosd = math.cos(math.radians(dir_deg))
    cost = math.cos(math.radians(tol))
    zra = P.site.zenith_ra
    tr = []
    for segs in tracks:
        n = len(segs)
        m = n // 2
        k0, d0 = segs[m].k, _mid(segs[m])
        tk0 = float(P.tk(k0))
        ga, gb = segs[max(m - 1, 0)], segs[min(m + 1, n - 1)]
        span = float(P.s(gb.k) - P.s(ga.k)) * P.spf
        rate = float(_deg(_mid(ga) @ _mid(gb))) / span
        u0 = _mid(gb) - _mid(ga)
        u0 = unit(u0 - (u0 @ d0) * d0)
        checks = [segs[1], segs[-2]] if n >= 5 else \
            [g for i, g in enumerate(segs) if i != m][:1] + segs[-1:]
        zra0 = float(zra(k0))
        fwd = []
        for i, g in enumerate(segs):
            v = _mid(segs[min(i + 1, n - 1)]) - _mid(segs[max(i - 1, 0)])
            fwd.append(bool((g.b - g.a) @ v > 0))
        tr.append(dict(
            k0=k0, d0=d0, rate=rate, u0=u0, fwd=fwd,
            lens=[float(_deg(g.a @ g.b)) for g in segs],
            lo=max(0, int(np.ceil(((tk0 - P.twin / 86400) - t0) * 86400
                                  / step))),
            hi=min(n_t - 1, int(np.floor(((tk0 + P.twin / 86400) - t0)
                                         * 86400 / step))),
            sky=P.camera.frame_to_gcrs(np.asarray(P.sky, float), k0),
            sun=sun_vectors(np.array([tk0]))[0],
            M=[_rotation(P.site.pole, float(zra(g.k)) - zra0).T
               for g in checks],
            dc=[_mid(g) for g in checks],
            dt=[float(P.s(g.k) - P.s(k0)) * P.spf for g in checks],
            t_row0=(t0 - tk0) * 86400))
    out: list[list] = [[] for _ in tracks]
    if not tr:
        return out
    a = dict(
        t_lo=np.array([t["lo"] for t in tr]),
        t_hi=np.array([t["hi"] for t in tr]),
        d0s=np.array([t["d0"] for t in tr]),
        skys=np.stack([t["sky"] for t in tr]),
        suns=np.array([t["sun"] for t in tr]),
        u0s=np.array([t["u0"] for t in tr]),
        r_lo=np.array([t["rate"] / rate_factor for t in tr]),
        r_hi=np.array([t["rate"] * rate_factor for t in tr]),
        pole=np.asarray(P.site.pole, float), dark=dark, cosd=cosd,
        h0=float(P.h0), A=A_WGS84, e2=E2, WE=WE)
    a["row_lo"] = int(a["t_lo"].min())
    a["nrows"] = int(a["t_hi"].max()) - a["row_lo"] + 1
    # ---- 1: every (row, object) against every track, single precision
    s_row, s_j, s_t = backend.stage1(a)
    if not len(s_row):
        return out
    # the survivors in a set order (the kernels' is a race's), so any two
    # runs, and any two backends, go on from the same list
    o = xp.lexsort(xp.stack([s_t, s_j, s_row]))
    s_row, s_j, s_t = s_row[o], s_j[o], s_t[o]
    # ---- 2: the survivors again, in double precision
    ok, x = backend.exact(a, s_row, s_j, s_t)
    keep = xp.flatnonzero(ok)
    if not len(keep):
        return out
    s_row = xp.ascontiguousarray(s_row[keep])
    s_j = xp.ascontiguousarray(s_j[keep])
    s_t = xp.ascontiguousarray(s_t[keep])
    x = xp.ascontiguousarray(x[keep])
    # ---- 3: each refined across its row, then finely around the best
    coarse = np.arange(-step / 2, step / 2 + 1e-9, refine_s)
    fine = np.arange(-refine_s / 2, refine_s / 2 + 1e-9, fine_s)
    top, top_dt, xr = backend.refine(
        a, s_row, s_j, s_t, x, coarse, fine,
        np.array([t["M"] for t in tr]), np.array([t["dc"] for t in tr]),
        np.array([t["dt"] for t in tr]))
    good = xp.flatnonzero(top > cost)
    rows, J, T_ = (backend.get(v[good]) for v in (s_row, s_j, s_t))
    Cc, dts, X = (backend.get(v[good]) for v in (top, top_dt, xr))
    # ---- per track: the site, and one candidate per pass of each object
    order = np.argsort(T_, kind="stable")
    cut = np.searchsorted(T_[order], np.arange(len(tr) + 1))
    for ti, t in enumerate(tr):
        sel = order[cut[ti]:cut[ti + 1]]
        if not sel.size:
            continue
        Jt = J[sel].astype(np.int64)
        Ct = Cc[sel]
        Tt = t["t_row0"] + rows[sel].astype(np.int64) * step + dts[sel]
        lat, dl = P.site.cell(t["k0"], X[sel])
        rate = t["rate"]
        E = float(np.median(t["lens"])) / rate if rate > 0 else P.spf / 2
        E = min(max(E, 0.5), P.spf)
        fwd = t["fwd"]
        out[ti] = [(int(Jt[i]), float(Tt[i]), E, fwd, float(lat[i]),
                    float(dl[i])) for i in merge(Jt, Ct, Tt, merge_s)]
    return out


def merge(J, C, T, merge_s: float) -> np.ndarray:
    """Passes of one object within `merge_s` collapse to the best: each
    object's passes best first, one kept unless within `merge_s` of one
    already kept (as `solve.free_passes`) — in rounds, every object at
    once: each round keeps each object's best pass left and drops that
    object's passes near it. -> indices, by object then best first."""
    order = np.lexsort((-C, J))
    Jo, To = J[order], T[order]
    alive = np.ones(len(order), bool)
    kept = []
    while alive.any():
        idx = np.flatnonzero(alive)
        jj = Jo[idx]
        first = idx[np.r_[True, jj[1:] != jj[:-1]]]
        kept.append(first)
        alive[first] = False
        idx = np.flatnonzero(alive)
        if not idx.size:
            break
        # the pass each object just kept, for each of its passes left
        k = np.searchsorted(Jo[first], Jo[idx])
        near = np.abs(To[idx] - To[first[k]]) <= merge_s
        alive[idx[near]] = False
    return order[np.sort(np.concatenate(kept))]
