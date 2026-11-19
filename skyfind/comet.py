"""Comets date a clip as well as planets do: a comet keeps its place among
the stars through a night, and only one comet was at that place on only
one night.

Finding one: a comet's coma glows green (C2 emission) and is fuzzy, where
stars are white-to-red points. In sampled frames of the cleaned clip the
green excess G - (R+B)/2, with the sky's large-scale glow taken off and
smoothed to coma size, peaks on comets; the peaks become sky positions
through the camera model, and a comet is a peak that stays put among the
stars across frames, away from any catalogued star.

Dating: the Minor Planet Center's orbits of every known comet
(`CometEls.txt`, ~160 kB, fetched once into the data directory), each
followed from a year and a half before to a year and a half after its
perihelion (comets are bright only near the Sun). Where one comes within
`tol_deg` of the observed position is a candidate date. The file's
brightness parameters are left out: for new comets they are often nuclear
magnitudes, far fainter than what a camera sees (Lemmon, magnitude ~4 in
October 2025, comes out ~9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import data
from .camera import Camera, radec_to_vec, unit, vec_to_radec

URL = "https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt"


# ---------------------------------------------------------------- finding
def above_land(sky: np.ndarray) -> np.ndarray:
    """Pixels clearly above the land in their column (a twentieth of the
    frame's height): lit equipment and lamps below the skyline — which a
    mask of the static foreground misses when they move — are left out."""
    land = sky == 0
    h = land.shape[0]
    top = np.where(land.any(0), land.argmax(0), h)
    return np.arange(h)[:, None] < (top[None, :] - h // 20)


def _green(img: np.ndarray, where: np.ndarray | None) -> np.ndarray:
    """Green excess G - (R+B)/2, band-passed to coma size (1.5 px blur minus
    6 px: a star has almost none, a lamp or airglow is too big)."""
    f = img.astype(np.float32)
    g = f[..., 1] - 0.5 * (f[..., 0] + f[..., 2])
    g = cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 6)
    if where is not None:
        g[~where] = 0
    return g


def peaks(img: np.ndarray, sky: np.ndarray | None = None, *, n: int = 12,
          k_sigma: float = 5.0, edge: int = 16) -> np.ndarray:
    """Greenest fuzzy spots in one frame, above the land -> (m, 3) x, y,
    score (sigmas)."""
    where = above_land(sky) if sky is not None else \
        np.ones(img.shape[:2], bool)
    where[:edge], where[-edge:], where[:, :edge], where[:, -edge:] = \
        False, False, False, False
    g = _green(img, where)
    med = np.median(g[where])
    sigma = 1.4826 * np.median(np.abs(g[where] - med)) + 1e-6
    mx = cv2.dilate(g, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero((g == mx) & (g > med + k_sigma * sigma) & where)
    sc = (g[ys, xs] - med) / sigma
    o = np.argsort(sc)[::-1][:n]
    return np.column_stack([xs[o], ys[o], sc[o]]).astype(float)


def candidates(cam: Camera, frames: dict[int, np.ndarray], stars,
               sky: np.ndarray | None = None, *, join_deg: float = 0.3,
               star_deg: float = 0.2, min_frames: int = 2,
               log=print) -> list[dict]:
    """Comet-like objects: green fuzzy peaks that stay at one sky position
    in `min_frames` or more of the sampled `frames` ({k: BGR image}), with
    no catalogued star within `star_deg`. -> [{ra, dec, frames, k_mid,
    score}], strongest first."""
    sv = radec_to_vec(stars.ra[stars.mag <= 6.5], stars.dec[stars.mag <= 6.5])
    pts = []                                       # (k, vec, score)
    for k, img in sorted(frames.items()):
        p = peaks(img, sky)
        if not len(p):
            continue
        v = cam.unproject(p[:, :2], k)
        near_star = (v @ sv.T).max(1) > math.cos(math.radians(star_deg))
        pts += [(k, vv, float(s)) for vv, s, ns in zip(v, p[:, 2], near_star)
                if not ns]
    groups: list[list] = []
    cj = math.cos(math.radians(join_deg))
    for q in pts:
        for g in groups:
            if unit(np.mean([x[1] for x in g], 0)) @ q[1] > cj and \
                    all(x[0] != q[0] for x in g):
                g.append(q)
                break
        else:
            groups.append([q])
    out = []
    for g in groups:
        ks = sorted({x[0] for x in g})
        if len(ks) < min_frames:
            continue
        first = min(g, key=lambda x: x[0])
        ra, dec = vec_to_radec(first[1])
        out.append(dict(ra=float(ra), dec=float(dec), k_mid=int(first[0]),
                        frames=ks, score=float(np.median([x[2] for x in g]))))
    out.sort(key=lambda c: (-len(c["frames"]), -c["score"]))
    if log:
        for c in out[:5]:
            log(f"  comet-like object RA {c['ra']:7.2f} Dec {c['dec']:+6.2f} "
                f"in {len(c['frames'])} frames (green {c['score']:.0f} "
                "sigma)")
    return out


# ---------------------------------------------------------------- orbits
@dataclass
class Orbit:
    name: str
    peri_tt: float                  # TT jd of perihelion
    q: float                        # AU
    e: float
    peri_arg: float                 # deg
    node: float
    incl: float
    g: float | None = None          # magnitude: g + 5 log D + 2.5 k log r
    k: float | None = None


def parse(text: str) -> list[Orbit]:
    """MPC's comet orbits, one-line format (columns as skyfield reads
    them)."""
    ts = data.timescale()
    out = []
    for ln in text.splitlines():
        if len(ln) < 102:
            continue
        try:
            y, m, d = int(ln[14:18]), int(ln[19:21]), float(ln[22:29])
            orb = Orbit(ln[102:158].strip(), ts.tt(y, m, d).tt,
                        float(ln[30:39]), float(ln[41:49]), float(ln[51:59]),
                        float(ln[61:69]), float(ln[71:79]))
        except ValueError:
            continue
        try:
            orb.g, orb.k = float(ln[91:95]), float(ln[96:100])
        except ValueError:
            pass
        out.append(orb)
    return out


def orbits(path: Path | None = None) -> list[Orbit]:
    """Every known comet's orbit (downloaded once)."""
    p = Path(path) if path else data.fetch(URL, "CometEls.txt")
    return parse(p.read_text(encoding="ascii", errors="ignore"))


def _kepler(o: Orbit):
    from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
    from skyfield.data.spice import inertial_frames
    from skyfield.keplerlib import _KeplerOrbit
    p = o.q * 2.0 if o.e == 1.0 else o.q / (1 - o.e) * (1 - o.e * o.e)
    t = data.timescale().tt_jd(o.peri_tt)
    k = _KeplerOrbit._from_periapsis(p, o.e, o.incl, o.node, o.peri_arg, t,
                                     GM_SUN, 10, o.name)
    k._rotation = inertial_frames["ECLIPJ2000"].T
    return k


def earth_sun(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric positions (au, ICRS) of the Earth and the Sun at TT jds:
    computed once for a grid, shared by every comet."""
    eph = data.skyfield_loader()("de421.bsp")
    t = data.timescale().tt_jd(np.asarray(tt, float))
    return (np.asarray(eph["earth"].at(t).position.au).T,
            np.asarray(eph["sun"].at(t).position.au).T)


def positions(o: Orbit, tt: np.ndarray, es=None, *, full: bool = False):
    """Geocentric unit vectors (ICRS) of a comet at TT jds (light time left
    out: under 0.02 deg for a comet near the Earth). `es`: `earth_sun(tt)`
    when already at hand. With `full`: (vectors, r, delta) — distances
    from the Sun and the Earth, au."""
    tt = np.asarray(tt, float)
    earth, sun = es if es is not None else earth_sun(tt)
    helio = np.asarray(_kepler(o).at(data.timescale().tt_jd(tt))
                       .position.au).T
    geo = sun + helio - earth
    if not full:
        return unit(geo)
    return (unit(geo), np.linalg.norm(helio, axis=-1),
            np.linalg.norm(geo, axis=-1))


def magnitude(o: Orbit, r, delta):
    """Total magnitude by the file's parameters (None without them); for
    new comets often far too faint — use as a loose cut only."""
    if o.g is None or o.k is None:
        return None
    return o.g + 5 * np.log10(delta) + 2.5 * o.k * np.log10(r)


# ---------------------------------------------------------------- dating
DE421 = (2414864.5, 2471184.5)           # 1900-2050, TT jd


def search(objects: list[dict], spf: float, *, orbit_list=None,
           years=(1990, 2049), tol_deg: float = 0.5, around_peri_d=550,
           max_q: float = 4.0, max_mag: float = 12.0, agree_d: float = 0.15,
           clock=None, log=print) -> list[dict]:
    """Candidate times of frame 0 (TT jd) from comet-like `objects`
    (`candidates`), best first: [{tt, utc, matched, comets, sep_deg}].

    Each comet within `max_q` AU of the Sun at perihelion is followed daily
    for `around_peri_d` days either side of it, then every 10 minutes where
    it comes within a few degrees of an object; a pass within `tol_deg`,
    no fainter than `max_mag` by the file's (pessimistic) magnitudes, is a
    match. Hundreds of faint comets wander the sky, so a single match can
    be chance: matches of different objects within `agree_d` days of each
    other make one candidate, ranked by how many objects agree."""
    if not objects:
        return []
    obs = orbit_list if orbit_list is not None else orbits()
    ts = data.timescale()
    t_lo = max(ts.J(years[0]).tt, DE421[0] + 1)
    t_hi = min(ts.J(years[1] + 1).tt, DE421[1] - 1)

    def cap(k):
        if clock is None:
            return float(k)
        return float(np.interp(k, np.arange(len(clock)), clock))

    ov = radec_to_vec(np.array([o["ra"] for o in objects]),
                      np.array([o["dec"] for o in objects]))
    obs = [o for o in obs if o.q <= max_q and
           t_lo < o.peri_tt + around_peri_d and o.peri_tt - around_peri_d < t_hi]
    if log:
        log(f"  {len(obs)} comets near perihelion between {years[0]} and "
            f"{years[1]}...")
    grid = np.arange(t_lo, t_hi, 1.0)
    es = earth_sun(grid)
    hits = []
    for o in obs:
        idx = np.flatnonzero(np.abs(grid - o.peri_tt) < around_peri_d)
        if not idx.size:
            continue
        try:
            v = positions(o, grid[idx], (es[0][idx], es[1][idx]))
        except Exception:                       # a degenerate orbit
            continue
        for i, obj in enumerate(objects):
            sep = np.degrees(np.arccos(np.clip(v @ ov[i], -1, 1)))
            near = grid[idx][sep < 6.0]         # comets move < ~6 deg/day
            if not near.size:
                continue
            brk = np.diff(near) > 1.5           # one fine look per run
            for lo, hi in zip(near[np.r_[True, brk]], near[np.r_[brk, True]]):
                fine = np.arange(lo - 1, hi + 1, 10 / 1440)
                fv, r, dist = positions(o, fine, full=True)
                fs = np.degrees(np.arccos(np.clip(fv @ ov[i], -1, 1)))
                j = int(np.argmin(fs))
                if fs[j] > tol_deg:
                    continue
                mag = magnitude(o, r[j], dist[j])
                if mag is not None and mag > max_mag:
                    continue
                t0 = fine[j] - cap(obj["k_mid"]) * spf / 86400
                hits.append(dict(tt=float(t0), sep_deg=float(fs[j]),
                                 comet=o.name, object=i,
                                 mag=None if mag is None else float(mag)))
    # different objects agreeing on one time: one candidate
    hits.sort(key=lambda c: c["sep_deg"])
    groups: list[list[dict]] = []
    for h in hits:
        for g in groups:
            if abs(h["tt"] - g[0]["tt"]) < agree_d and \
                    all(h["object"] != x["object"] for x in g) and \
                    all(h["comet"] != x["comet"] for x in g):
                g.append(h)
                break
        else:
            if all(abs(h["tt"] - g[0]["tt"]) > 2 or h["comet"] !=
                   g[0]["comet"] for g in groups):
                groups.append([h])
    out = []
    for g in groups:
        tt = float(np.mean([x["tt"] for x in g]))
        out.append(dict(tt=tt, matched=len(g),
                        comets=[x["comet"] for x in g],
                        objects=[x["object"] for x in g],
                        sep_deg=float(np.mean([x["sep_deg"] for x in g])),
                        span_h=float(np.ptp([x["tt"] for x in g]) * 24),
                        utc=ts.tt_jd(tt).utc_strftime("%Y-%m-%d %H:%M")))
    out.sort(key=lambda c: (-c["matched"], c["sep_deg"]))
    if log:
        for c in out[:5]:
            log(f"  frame 0 ~ {c['utc']} UTC: {', '.join(c['comets'])} "
                f"({c['matched']} object(s), {c['sep_deg']:.2f} deg off)")
    return out
