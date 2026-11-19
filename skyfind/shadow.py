"""Earth's shadow: which trails could be sunlit satellites at all.

A satellite shows only while the Sun lights it. The stars give each trail's
direction and the site's sidereal time, the skyline the latitude, and the
night's date the Sun: so along each line of sight the heights where a
satellite would be sunlit are known, and an orbit at any of them crosses
the line of sight at most so fast (a little over circular speed, plus the
site's own motion). A trail moving faster than every sunlit orbit could is
something else — an aircraft, most likely: they carry their own lights
(a dolly clip: the airway streaks low in the east, deep in the shadow by then).

Loose on purpose, like the candidate pruning: every site on the skyline's
grid (`solve.SITE_GRID`), the Sun a day either side, the slowest the trail
could be moving (its length over the whole frame interval: the exposure is
no longer; a track's own drift when it has one), the shadow's umbra (the
penumbra still lights a satellite), and a quarter to spare. It needs no
orbital elements. A free site (no skyline: no latitude, no up) isn't
tested.
"""

from __future__ import annotations

import numpy as np

from . import date
from .camera import unit

MU = 398600.4418                    # km^3/s^2
R_EARTH = 6378.137                  # km
UMBRA = 0.00465                     # rad: the umbra cone's half-angle
HEIGHTS = np.geomspace(150.0, 40000.0, 200)   # km: satellite heights tried
FASTER = 1.1                        # an orbit's speed over circular, at most
SITE_KMS = 0.5                      # km/s: the site's own motion, and some
MARGIN = 1.25                       # a trail's rate to spare
DAYS = 1.0                          # the Sun's date either side


def suns(tt: float, days: float = DAYS) -> np.ndarray:
    """(3, 3) GCRS unit vectors to the Sun at tt - days, tt, tt + days."""
    t = float(tt) + np.array([-days, 0.0, days])
    return date.planet_vectors(t, names=["sun"])["sun"]


def lit(p: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """Points p (..., 3) km GCRS outside the Earth's umbra, for the Sun
    direction `sun` (3,)."""
    along = p @ sun
    perp = np.sqrt(np.maximum((p * p).sum(-1) - along ** 2, 0.0))
    behind = -along
    return (along > 0) | (perp > R_EARTH - behind * np.tan(UMBRA))


def fastest(sites: np.ndarray, d: np.ndarray, sun_dirs: np.ndarray
            ) -> float:
    """Fastest (deg/s) a sunlit satellite could cross the line of sight d
    (3,) from any of `sites` (S, 3) km GCRS, for any of `sun_dirs`
    (U, 3): 0 when nothing along it is ever lit."""
    R = np.linalg.norm(sites, axis=1)[:, None]                  # (S, 1)
    rd = (sites @ d)[:, None]
    rho = -rd + np.sqrt(rd * rd - R * R + (R + HEIGHTS) ** 2)    # (S, H)
    p = sites[:, None, :] + rho[..., None] * d                  # (S, H, 3)
    v = FASTER * np.sqrt(MU / (R_EARTH + HEIGHTS)) + SITE_KMS   # (H,)
    rate = np.degrees(v / rho)                                  # (S, H)
    ok = np.zeros(rate.shape, bool)
    for s in sun_dirs:
        ok |= lit(p, s)
    return float(np.max(np.where(ok, rate, 0.0)))


def rates(P, segs, *, track: bool = False) -> np.ndarray:
    """The slowest each trail could be crossing the sky (deg/s): a
    segment's length over the whole frame interval (the exposure is no
    longer); with `track`, `segs` is one track and its drift over the
    frames gives the rate itself (one value)."""
    from .solve import track_motion
    if track:
        return np.array([track_motion(P, segs)[0]])
    ang = np.array([np.degrees(np.arccos(np.clip(s.a @ s.b, -1, 1)))
                    for s in segs])
    return ang / P.spf


def possible(P, segs, rate: np.ndarray) -> np.ndarray:
    """Per segment: could a sunlit satellite, seen from some site on the
    skyline's grid, cross the sky there that fast (with `MARGIN`)?"""
    from .solve import SITE_GRID
    grid = np.array(SITE_GRID, float) + [P.lat0, 0.0]
    sun_dirs = suns(P.t_ref)
    out = np.ones(len(segs), bool)
    for i, (s, r) in enumerate(zip(segs, rate)):
        sites = P.site.gcrs(s.k, grid[:, 0], grid[:, 1], P.h0)
        out[i] = r <= MARGIN * fastest(sites, unit(s.a + s.b), sun_dirs)
    return out


def tracks_possible(P, track_segs) -> np.ndarray:
    """Per track: its middle segment's line of sight at its own drift."""
    out = np.ones(len(track_segs), bool)
    for i, segs in enumerate(track_segs):
        mid = segs[len(segs) // 2]
        out[i] = possible(P, [mid], rates(P, segs, track=True))[0]
    return out
