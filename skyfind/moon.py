"""The Moon: where it stands among the stars dates a clip to a few hours,
once every lunar month.

A bright Moon is rarely a disc in a timelapse; it's an overexposed glow
(a 4K moonrise: the glow ~140 px across, the disc ~12 px). The glow's
centre is still where the Moon is to a few tenths of a degree, and it moves
with the stars, unlike the flares and ghosts it throws, which stay put in
the frame. Seen from the unknown site it's up to ~1 deg off its geocentric
place (parallax) and near the horizon lifted by refraction: the match
tolerance allows for both.

Where the Moon is (to `TOL_DEG`) pins the time to a stretch of hours once
every sidereal month (27.3 days); over decades that's hundreds of
stretches, so the Moon alone doesn't date a clip. It narrows a rough span
to a night or two a month, which `pipeline.run_scan` then tests with the
satellites.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .camera import unit, vec_to_radec

TOL_DEG = 2.0            # parallax (<= 0.95 deg) + refraction + the glow
MIN_GLOW = 1e-4          # the glow's bright core: at least this much frame
MIN_PEAK = 200           # a Moon glow is near white


def _deg(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def glow(img: np.ndarray, sky=None) -> tuple[float, float] | None:
    """(x, y) of the centre of the frame's brightest glow, if it's big
    enough to be the Moon and not cut by the frame's edge; `sky`: where
    the sky is (lights on the land don't count)."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    if sky is not None:
        g = np.where(sky, g, 0).astype(g.dtype)
    peak = int(g.max())
    if peak < MIN_PEAK:
        return None
    thr = int(0.9 * peak)
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        (g >= thr).astype(np.uint8))
    y, x = np.unravel_index(int(np.argmax(g)), g.shape)
    i = int(lab[y, x])
    x0, y0, w, h, area = st[i]
    H, W = g.shape
    if area < MIN_GLOW * W * H:
        return None
    if x0 <= 1 or y0 <= 1 or x0 + w >= W - 1 or y0 + h >= H - 1:
        return None                         # cut by the edge: centre off
    ys, xs = np.nonzero(lab == i)
    wt = g[ys, xs].astype(float) - thr + 1
    return float((xs * wt).sum() / wt.sum()), float((ys * wt).sum() / wt.sum())


def find(cam, frames: dict[int, np.ndarray], sky=None, *, log=print
         ) -> dict | None:
    """The Moon in sampled `frames` ({k: BGR image}): {ra, dec, k, frames,
    spread_deg} (apparent GCRS direction at frame k), or None. Kept only
    if its glows agree on one place among the stars (a flare fixed in the
    frame drifts across them at the sidereal rate)."""
    pts = []
    for k, img in sorted(frames.items()):
        c = glow(img, sky)
        if c is not None:
            pts.append((k, unit(cam.unproject(np.array([c]), k)[0])))
    if len(pts) < 3:
        return None
    V = np.array([v for _, v in pts])
    med = unit(np.median(V, 0))
    ok = _deg(V @ med) < 1.5
    if ok.sum() < max(3, len(pts) // 2):
        if log:
            log(f"  a bright glow in {len(pts)} frames, but it doesn't stay "
                "with the stars: not the Moon")
        return None
    ks = [k for (k, _), o in zip(pts, ok) if o]
    V = V[ok]
    med = unit(V.mean(0))
    k = int(np.median(ks))
    # the direction at the middle detection (the Moon drifts ~0.5 deg/h)
    v = V[int(np.argmin([abs(kk - k) for kk in ks]))]
    ra, dec = vec_to_radec(v)
    out = dict(ra=float(ra), dec=float(dec), k=int(ks[int(np.argmin(
        [abs(kk - k) for kk in ks]))]), frames=ks,
        spread_deg=float(_deg(V @ med).max()))
    if log:
        log(f"  the Moon: RA {out['ra']:.2f} Dec {out['dec']:+.2f} (seen in "
            f"{len(ks)} frames, glows within {out['spread_deg']:.2f} deg)")
    return out


def lit(elong_deg: float) -> float:
    """Fraction of the Moon's disc lit, from its angle from the Sun."""
    return (1 - math.cos(math.radians(elong_deg))) / 2


def phase(elong_deg: float) -> str:
    f = lit(elong_deg)
    if f > 0.97:
        return "full"
    if f < 0.03:
        return "new"
    return f"{f * 100:.0f}% lit"


def search(moon: dict, spf: float, *, clock=None, first_tt: float,
           last_tt: float, tol_deg: float = TOL_DEG, log=print
           ) -> list[dict]:
    """When the Moon stood where `moon` saw it, between TT jds `first_tt`
    and `last_tt`: [{tt, utc (frame 0), sep_deg, span_h, elong_deg,
    phase}], in time order. A 2-hour sweep, then 5 minutes around each
    pass (the Moon moves ~0.55 deg an hour among the stars)."""
    from . import date
    ts = date.data.timescale()
    v = unit(np.array([math.cos(math.radians(moon["dec"]))
                       * math.cos(math.radians(moon["ra"])),
                       math.cos(math.radians(moon["dec"]))
                       * math.sin(math.radians(moon["ra"])),
                       math.sin(math.radians(moon["dec"]))]))
    k = moon["k"]
    s_k = float(np.interp(k, np.arange(len(clock)), clock)) \
        if clock is not None else float(k)
    to0 = s_k * spf / 86400                     # frame k -> frame 0
    step = 2 / 24
    grid = np.arange(first_tt + to0, last_tt + to0 + step, step)
    sep = np.concatenate([_deg(date.planet_vectors(g, ("moon",))["moon"] @ v)
                          for g in np.array_split(grid, max(1, len(grid)
                                                            // 50000))])
    near = np.flatnonzero(sep < tol_deg + 1.2)
    runs = [r for r in np.split(near, np.flatnonzero(np.diff(near) > 1) + 1)
            if r.size]
    # every pass's 5-minute grid in one ephemeris call
    fines = [np.arange(grid[r[0]] - step, grid[r[-1]] + step, 5 / 1440)
             for r in runs]
    if not fines:
        fines = [np.zeros(0)]
    allv = date.planet_vectors(np.concatenate(fines), ("moon", "sun")) \
        if runs else {"moon": np.zeros((0, 3)), "sun": np.zeros((0, 3))}
    edges = np.cumsum([0] + [len(f) for f in fines])
    out = []
    for fine, a, b_ in zip(fines, edges[:-1], edges[1:]):
        vm, vs = allv["moon"][a:b_], allv["sun"][a:b_]
        fs = _deg(vm @ v)
        b = int(np.argmin(fs))
        if fs[b] > tol_deg:
            continue
        within = fine[fs <= tol_deg]
        el = float(_deg(vm[b] @ vs[b]))
        t0 = float(fine[b] - to0)
        out.append(dict(tt=t0, utc=ts.tt_jd(t0).utc_strftime("%Y-%m-%d %H:%M"),
                        sep_deg=float(fs[b]),
                        span_h=float((within.max() - within.min()) * 24),
                        elong_deg=el, lit=lit(el), phase=phase(el)))
    if log:
        log(f"  the Moon stood there {len(out)} times between "
            f"{ts.tt_jd(first_tt).utc_strftime('%Y-%m-%d')} and "
            f"{ts.tt_jd(last_tt).utc_strftime('%Y-%m-%d')} (once a lunar "
            "month)")
    return out
