"""When: date the shoot from the planets (and the Sun, and precession).

The site/time solve needs a nominal time to search around (+-12 h covers
every longitude, but not every day). Blind, the footage has to supply it:

- **Planets** are bright points missing from the star catalogue that stay put
  among the stars all night. Where each sits on the sky, compared with the
  ephemeris (DE421, 1900-2050), dates the shoot: Mars moves ~0.5 deg a day,
  so a Mars + Jupiter pair pins the day and even the hour roughly.
- **The Sun** must be below the horizon at every usable frame. Its altitude
  needs only the latitude and each frame's sidereal time (both from stars +
  skyline, no longitude), so this rules out the wrong seasons.
- **Precession**: the fitted rotation axis is the pole *of date*, which
  drifts ~20"/yr from the J2000 pole — the year to within a few.

Planet positions are geocentric; parallax is below 0.01 deg for all but the
Moon, which is not used.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import data
from .camera import Camera, radec_to_vec, unit, vec_to_radec
from .catalog import Stars

PLANETS = ("mercury", "venus", "mars", "jupiter barycenter",
           "saturn barycenter", "uranus barycenter")
DE421_SPAN = (1900.1, 2049.9)


def _sep(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1), -1, 1)))


# ---------------------------------------------------------------- measure
# Bright nebulae, clusters and galaxies within the planets' reach (|Dec| up
# to ~35 deg), J2000 RA, Dec and a radius (deg): a glow no catalogue star
# explains that sits on one is that, not a planet (one real clip's two "planets"
# were the Lagoon Nebula; dated by them, 1993).
DEEP_SKY = [
    ("M1", 83.63, 22.01, 0.3), ("M2", 323.36, -0.82, 0.3),
    ("M3", 205.55, 28.38, 0.3), ("M4", 245.90, -26.53, 0.4),
    ("M5", 229.64, 2.08, 0.3), ("M6", 265.03, -32.22, 0.5),
    ("M7", 268.47, -34.82, 0.8), ("M8 Lagoon Nebula", 270.90, -24.38, 0.9),
    ("M9", 259.80, -18.52, 0.2), ("M10", 254.29, -4.10, 0.3),
    ("M11", 282.77, -6.27, 0.3), ("M12", 251.81, -1.95, 0.3),
    ("M14", 264.40, -3.25, 0.2), ("M15", 322.49, 12.17, 0.3),
    ("M16 Eagle Nebula", 274.70, -13.78, 0.5),
    ("M17 Omega Nebula", 275.20, -16.18, 0.4), ("M18", 274.98, -17.13, 0.2),
    ("M19", 255.66, -26.27, 0.2), ("M20 Trifid Nebula", 270.65, -23.03, 0.5),
    ("M21", 271.15, -22.50, 0.3), ("M22", 279.10, -23.90, 0.4),
    ("M23", 269.20, -19.02, 0.4), ("M24 Sagittarius Star Cloud", 274.23,
                                   -18.48, 1.2),
    ("M25", 277.90, -19.25, 0.4), ("M26", 281.30, -9.40, 0.2),
    ("M27 Dumbbell Nebula", 299.90, 22.72, 0.3), ("M28", 276.14, -24.87, 0.2),
    ("M30", 325.09, -23.18, 0.2), ("M33", 23.46, 30.66, 0.6),
    ("M35", 92.23, 24.33, 0.5), ("M36", 84.03, 34.13, 0.3),
    ("M37", 88.10, 32.55, 0.4), ("M38", 82.18, 35.83, 0.4),
    ("M41", 101.50, -20.73, 0.5), ("M42 Orion Nebula", 83.82, -5.39, 0.8),
    ("M44 Beehive Cluster", 130.03, 19.98, 1.0),
    ("M45 Pleiades", 56.75, 24.12, 1.2), ("M46", 115.45, -14.82, 0.4),
    ("M47", 114.15, -14.50, 0.4), ("M48", 123.45, -5.80, 0.5),
    ("M50", 105.80, -8.33, 0.3), ("M54", 283.76, -30.48, 0.2),
    ("M55", 294.99, -30.96, 0.3), ("M62", 255.30, -30.11, 0.2),
    ("M67", 132.83, 11.82, 0.4), ("M69", 277.85, -32.35, 0.2),
    ("M70", 280.80, -32.29, 0.2), ("M72", 313.37, -12.54, 0.2),
    ("M75", 301.52, -21.92, 0.2), ("M77", 40.67, -0.01, 0.2),
    ("M78", 86.68, 0.05, 0.3), ("M80", 244.26, -22.98, 0.2),
    ("M83", 204.25, -29.87, 0.3), ("M93", 116.15, -23.87, 0.3),
    ("M104 Sombrero Galaxy", 190.00, -11.62, 0.2),
    ("NGC 253 Sculptor Galaxy", 11.89, -25.29, 0.4),
]


def deep_sky(ra: float, dec: float) -> str | None:
    """The bright deep-sky object at (ra, dec), if one is (`DEEP_SKY`)."""
    v = radec_to_vec(np.array([ra]), np.array([dec]))[0]
    for name, r, d, rad in DEEP_SKY:
        if _sep(radec_to_vec(np.array([r]), np.array([d]))[0], v) < rad:
            return name
    return None


def planet_candidates(cam: Camera, dets: dict[int, np.ndarray], stars: Stars,
                      *, per_frame: int = 25, clear_deg: float = 0.3,
                      join_deg: float = 0.25, min_frames: int = 3,
                      log=print) -> list[dict]:
    """Bright detections with no catalogue star within `clear_deg`, grouped
    by sky position across sampled frames; kept when seen in `min_frames`
    or more (aircraft and satellites don't stay put; hot pixels ride with
    the camera, not the sky)."""
    cv = radec_to_vec(stars.ra[stars.mag <= 6.5], stars.dec[stars.mag <= 6.5])
    tree = cKDTree(cv)
    chord = 2 * np.sin(np.radians(clear_deg) / 2)
    pts = []                                     # (k, vec, flux)
    for k, d in sorted(dets.items()):
        d = d[:per_frame]
        if not len(d):
            continue
        v = cam.unproject(d[:, :2], k)
        dist, _ = tree.query(v)
        for vv, fl, dd in zip(v, d[:, 2], dist):
            if dd > chord:
                pts.append((k, vv, float(fl)))
    groups: list[list] = []
    for p in pts:
        for g in groups:
            if _sep(p[1], unit(np.mean([q[1] for q in g], 0))) < join_deg:
                g.append(p)
                break
        else:
            groups.append([p])
    out = []
    for g in groups:
        ks = sorted({q[0] for q in g})
        if len(ks) < min_frames:
            continue
        v = unit(np.mean([q[1] for q in g], 0))
        ra, dec = vec_to_radec(v)
        spread = float(np.max([_sep(q[1], v) for q in g]))
        out.append(dict(ra=float(ra), dec=float(dec), frames=ks,
                        k_mid=int(np.median(ks)),
                        flux=float(np.median([q[2] for q in g])),
                        spread_deg=spread))
    out.sort(key=lambda o: -o["flux"])
    kept = []
    for o in out:
        name = deep_sky(o["ra"], o["dec"])
        if name is None:
            kept.append(o)
        elif log:
            log(f"  glow at RA {o['ra']:7.2f} Dec {o['dec']:+6.2f}: {name}, "
                "not a planet")
    out = kept
    if log:
        for o in out:
            log(f"  non-catalogue object RA {o['ra']:7.2f} Dec "
                f"{o['dec']:+6.2f} in {len(o['frames'])} frames "
                f"(flux {o['flux']:.0f})")
    return out


# ---------------------------------------------------------------- ephemeris
def planet_vectors(tt: np.ndarray, names=PLANETS) -> dict[str, np.ndarray]:
    """Astrometric geocentric unit vectors (ICRS) at TT jds `tt`."""
    load = data.skyfield_loader()
    eph = load("de421.bsp")
    t = data.timescale().tt_jd(np.asarray(tt, float))
    earth = eph["earth"]
    out = {}
    for n in names:
        p = earth.at(t).observe(eph[n]).position.au
        out[n] = unit(np.asarray(p).T)
    return out


def sun_vectors(tt: np.ndarray) -> np.ndarray:
    return planet_vectors(tt, ("sun",))["sun"]


def pole_year(axis: np.ndarray, years=np.arange(1900, 2050.1, 0.5)
              ) -> tuple[float, np.ndarray]:
    """Year whose pole of date is closest to `axis` -> (year, separations).
    For the record only, never to narrow a date search: the pole of date
    moves 20" a year and a fitted sky axis is good to a tenth of a degree or
    a few — decades (two real clips read 13 and 60 years early)."""
    ts = data.timescale()
    t = ts.J(years)
    poles = np.asarray(t.M)[2].T                      # (n, 3) pole in GCRS
    sep = _sep(poles, axis[None])
    return float(years[int(np.argmin(sep))]), sep


# ---------------------------------------------------------------- search
# the planets keep near the ecliptic: declination within
# 23.4 deg (obliquity) + ~8.7 deg (Venus at inferior conjunction)
PLANET_MAX_DEC = 33.0


def search(objects: list[dict], spf: float, zenith: np.ndarray | None = None,
           *, years=(1990, 2049), coarse_h: float = 6.0, tol_deg: float = 1.0,
           fine_tol_deg: float = 0.3, sun_max_alt: float = -6.0,
           usable: tuple[int, int] | None = None, max_runs: int = 60,
           clock=None, log=print) -> list[dict]:
    """Candidate times of frame 0 (TT jd), best first.

    Each object must sit on a different planet at its frames' times. With
    `zenith` ((n_frames, 3) camera zenith in GCRS per frame), the Sun must be
    below `sun_max_alt` at the usable frames."""
    far = [o for o in objects if abs(o["dec"]) > PLANET_MAX_DEC]
    objects = [o for o in objects if abs(o["dec"]) <= PLANET_MAX_DEC]
    if far and log:
        log(f"  {len(far)} of {len(far) + len(objects)} object(s) too far "
            f"from the ecliptic to be planets (|Dec| > {PLANET_MAX_DEC:g})")
    if not objects:
        if log:
            log("  nothing in view can be a planet")
        return []
    def cap(k):                          # capture index of frame k
        if clock is None:
            return float(k)
        return float(np.interp(k, np.arange(len(clock)), clock))
    y0, y1 = max(years[0], DE421_SPAN[0]), min(years[1], DE421_SPAN[1])
    ts = data.timescale()
    t_lo, t_hi = ts.J(y0).tt, ts.J(y1).tt
    grid = np.arange(t_lo, t_hi, coarse_h / 24)
    kmid = [o["k_mid"] for o in objects]
    ov = radec_to_vec(np.array([o["ra"] for o in objects]),
                      np.array([o["dec"] for o in objects]))
    if log:
        log(f"  searching {y0:.0f}-{y1:.0f} in {coarse_h:g}-hour steps for "
            f"{len(objects)} object(s)...")
    # planets barely move within a night (Mercury at most ~2 deg a day, well
    # inside tol_deg): one ephemeris at the clip's middle serves every object
    names = list(PLANETS)
    pv = planet_vectors(grid + cap(float(np.median(kmid))) * spf / 86400)
    seps = np.stack([np.stack([_sep(pv[n], ov[i][None]) for n in names])
                     for i in range(len(objects))])   # (n_obj, n_planet, n_t)
    matched, coarse_cost, _ = _assign(seps, tol_deg)
    ok = matched > 0
    if zenith is not None:
        ok &= _dark(grid, spf, zenith, usable, sun_max_alt, cap)
    if ok.any():                        # as many objects on planets as can be
        ok &= matched == matched[ok].max()
    idx = np.flatnonzero(ok)
    if log:
        log(f"  date search {y0:.0f}-{y1:.0f}: {len(idx)} of {len(grid)} "
            f"{coarse_h:g}-hour steps put {matched[ok].max() if ok.any() else 0}"
            f" of {len(objects)} "
            f"object(s) on planets")
    # refine the best `max_runs` runs of coarse hits at 15-minute steps
    # around each run's best step (a slow planet can sit by an object for
    # months: its whole run needn't be combed); one ephemeris serves every
    # object, as above
    runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            if r.size]
    runs.sort(key=lambda r: float(coarse_cost[r].min()))
    out = []
    for run in runs[:max_runs]:
        c0 = grid[run[int(np.argmin(coarse_cost[run]))]]
        fine = np.arange(c0 - coarse_h / 24, c0 + coarse_h / 24, 0.25 / 24)
        pv = planet_vectors(fine + cap(float(np.median(kmid))) * spf / 86400)
        fs = np.stack([np.stack([_sep(pv[n], ov[i][None]) for n in names])
                       for i in range(len(objects))])
        fm, cost, assign = _assign(fs, fine_tol_deg)
        fok = fm > 0
        if zenith is not None:
            fok &= _dark(fine, spf, zenith, usable, sun_max_alt, cap)
        if not fok.any():
            continue
        fok &= fm == fm[fok].max()
        cost = np.where(fok, cost, np.inf)
        b = int(np.argmin(cost))
        within = fine[np.sqrt(cost) < np.sqrt(cost[b]) + 0.05]
        out.append(dict(tt=float(fine[b]), matched=int(fm[b]),
                        rms_deg=float(np.sqrt(cost[b])),
                        planets={i: names[assign[i, b]]
                                 for i in range(len(objects))
                                 if assign[i, b] >= 0},
                        span_h=float((within.max() - within.min()) * 24),
                        utc=ts.tt_jd(fine[b]).utc_strftime("%Y-%m-%d %H:%M")))
    out.sort(key=lambda c: (-c["matched"], c["rms_deg"]))
    if log:
        for c in out[:5]:
            log(f"  frame 0 ~ {c['utc']} UTC (+-{c['span_h'] / 2:.1f} h): "
                + ", ".join(f"object {i} = {n.split()[0]}"
                            for i, n in c["planets"].items())
                + f", rms {c['rms_deg']:.2f} deg")
    return out


def _assign(seps: np.ndarray, tol: float):
    """Best assignment of objects to distinct planets within `tol` (objects
    may stay unmatched: a star missing from the catalogue, a flare).
    -> (matched count, mean squared separation of the matched, assignment
    with -1 = unmatched), each per time.

    Only objects within `tol` of some planet at a time can matter then, and
    those are few (usually none, rarely three): times are grouped by which
    objects are, and each group searched exhaustively over just those
    (all of them at once would be (planets + objects)! / planets! orders —
    121 million for 8 objects)."""
    n_obj, n_pl, n_t = seps.shape
    best_m = np.zeros(n_t, int)
    best_c = np.full(n_t, np.inf)
    assign = np.full((n_obj, n_t), -1)
    active = (seps < tol).any(1)                         # (object, time)
    if not active.any():
        return best_m, best_c, assign
    pats, inv = np.unique(active.T, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for g, pat in enumerate(pats):
        if not pat.any():
            continue
        ts_ = np.flatnonzero(inv == g)
        objs = np.flatnonzero(pat)
        m, c, a = _assign_all(seps[objs][:, :, ts_], tol)
        best_m[ts_], best_c[ts_] = m, c
        assign[np.ix_(objs, ts_)] = a
    return best_m, best_c, assign


def _assign_all(seps: np.ndarray, tol: float):
    """`_assign` by trying every assignment (few objects only)."""
    import itertools
    n_obj, n_pl, n_t = seps.shape
    best_m = np.zeros(n_t, int)
    best_c = np.full(n_t, np.inf)
    assign = np.full((n_obj, n_t), -1)
    slots = list(range(n_pl)) + [-1 - i for i in range(n_obj)]
    zero = np.zeros(n_t)
    for perm in itertools.permutations(slots, n_obj):
        hit = [(seps[i, p] < tol) if p >= 0 else np.zeros(n_t, bool)
               for i, p in enumerate(perm)]
        m = np.sum(hit, axis=0)
        sq = np.sum([np.where(h, seps[i, p] ** 2, 0.0) if p >= 0 else zero
                     for i, (p, h) in enumerate(zip(perm, hit))], axis=0)
        c = np.where(m > 0, sq / np.maximum(m, 1), np.inf)
        better = (m > best_m) | ((m == best_m) & (c < best_c))
        best_m[better] = m[better]
        best_c[better] = c[better]
        a = np.array([p if p >= 0 else -1 for p in perm])[:, None]
        hitm = np.array(hit)
        assign[:, better] = np.where(hitm[:, better], a, -1)
    return best_m, best_c, assign


def _dark(t0: np.ndarray, spf: float, zenith: np.ndarray,
          usable, max_alt: float, cap=float) -> np.ndarray:
    """Sun below `max_alt` at the first, middle and last usable frame."""
    a, b = usable if usable else (0, len(zenith) - 1)
    ok = np.ones(len(t0), bool)
    for k in (a, (a + b) // 2, b):
        s = sun_vectors(t0 + cap(k) * spf / 86400)
        alt = np.degrees(np.arcsin(np.clip(s @ zenith[k], -1, 1)))
        ok &= alt < max_alt
    return ok
