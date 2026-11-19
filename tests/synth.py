"""A synthetic sky: real physics, made-up everything else.

The solver has to recover what is chosen here and hidden from it: the site
(latitude, longitude, height), the UTC of frame 0, the frame interval and
exposure, and the camera's pointing and lens. Stars come from a random
catalogue, satellites from made-up but valid orbital elements (written as
TLE text) propagated with SGP4. Earth rotation, precession and nutation come
from skyfield, so the rendered sky turns about the true pole of date.
Refraction is optional (`refract`, the atmosphere's density: it bends stars
and satellites alike); aberration is not modelled. The camera may sit on a
motion-control rig (`pan`, `tilt`): it then turns frame by frame, the land
and its skyline move through the picture with it, and the land's texture is
fixed to the ground (by azimuth and altitude), not to the frame.

Nothing here reads the solver's code paths for the physics: observer and
camera go through skyfield's own frames, so a sign or frame slip in skyfind
shows up as a failed recovery rather than cancelling out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cached_property

import cv2
import numpy as np
from sgp4.api import SatrecArray
from skyfield.api import load, wgs84
from skyfield.sgp4lib import TEME

from skyfind import catalog, tles
from skyfind.camera import (SIDEREAL_DEG_PER_DAY, Camera, lift, refraction,
                            unit)

TS = load.timescale(builtin=True)
MU = 398600.4418                                   # km^3/s^2
RE = 6378.137


# ------------------------------------------------------------------ elements
def _checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else (c == "-") for c in line[:68]) % 10


def _exp_field(x: float) -> str:
    """TLE assumed-decimal exponent field, e.g. 1.2e-4 -> ' 12000-3'."""
    if x == 0:
        return " 00000-0"
    sign = "-" if x < 0 else " "
    e = math.floor(math.log10(abs(x))) + 1
    m = abs(x) / 10 ** e
    mant = int(round(m * 1e5))
    if mant >= 100000:
        mant, e = mant // 10, e + 1
    return f"{sign}{mant:05d}{'-' if e < 0 else '+'}{abs(e):d}"


def make_tle(satnum: int, name: str, epoch: datetime, incl: float,
             raan: float, ecc: float, argp: float, mean_anom: float,
             rev_per_day: float, bstar: float = 1e-4) -> tles.Element:
    doy = (epoch - datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
           ).total_seconds() / 86400 + 1
    l1 = (f"1 {satnum:05d}U 24001A   {epoch.year % 100:02d}{doy:012.8f} "
          f" .00000000  00000-0 {_exp_field(bstar)} 0  999")
    l2 = (f"2 {satnum:05d} {incl:8.4f} {raan % 360:8.4f} "
          f"{int(round(ecc * 1e7)):07d} {argp % 360:8.4f} "
          f"{mean_anom % 360:8.4f} {rev_per_day:11.8f}    1")
    l1 = l1 + str(_checksum(l1))
    l2 = l2 + str(_checksum(l2))
    assert len(l1) == 69 and len(l2) == 69, (l1, l2)
    return tles.Element(name, l1, l2)


def _rev_per_day(a_km: float) -> float:
    return math.sqrt(MU / a_km ** 3) * 86400 / (2 * math.pi)


def constellation(epoch: datetime, seed: int = 1) -> tuple[list, dict]:
    """~2800 made-up objects: Starlink-like shells and a train, LEO rocket
    bodies, GTO rocket bodies (slow near apogee: the multi-frame tracks),
    MEO/GEO payloads and faint debris. Returns (elements, brightness) where
    brightness[norad] is an absolute magnitude at 1000 km."""
    rng = np.random.default_rng(seed)
    els, mag = [], {}
    num = [10000]

    def add(name, incl, raan, ecc, argp, ma, a_km, m1000):
        num[0] += 1
        e = make_tle(num[0], name, epoch, incl, raan, ecc, argp, ma,
                     _rev_per_day(a_km), bstar=float(rng.uniform(1e-5, 3e-4)))
        els.append(e)
        mag[num[0]] = m1000

    for incl, alt, planes, per in [(53.0, 550, 22, 22), (53.2, 540, 22, 22),
                                   (70.0, 570, 10, 20), (97.6, 560, 8, 20)]:
        for p in range(planes):
            raan = 360 * p / planes + rng.uniform(-2, 2)
            for s in range(per):
                add(f"SYN-STARLINK-{num[0] + 1}", incl, raan, 1e-4, 90,
                    360 * s / per + 7.3 * p + rng.uniform(-1, 1), RE + alt,
                    5.0 + rng.uniform(-0.5, 0.5))
    raan, ma = rng.uniform(0, 360), rng.uniform(0, 360)
    for s in range(20):                              # a fresh launch train
        add(f"SYN-STARLINK-T{s:02d}", 43.0, raan, 2e-4, 90, ma + 0.4 * s,
            RE + 350, 4.0)
    for _ in range(300):                             # LEO rocket bodies
        hp = rng.uniform(300, 900)
        ha = hp + rng.uniform(0, 1500)
        a = RE + (hp + ha) / 2
        add(f"SYN R/B {num[0] + 1}", rng.uniform(28, 100), rng.uniform(0, 360),
            (ha - hp) / (2 * a), rng.uniform(0, 360), rng.uniform(0, 360),
            a, 3.0 + rng.uniform(-0.5, 1.0))
    for _ in range(150):                             # GTO rocket bodies
        hp, ha = rng.uniform(200, 700), rng.uniform(25000, 36000)
        a = RE + (hp + ha) / 2
        add(f"SYN GTO R/B {num[0] + 1}", rng.uniform(5, 30),
            rng.uniform(0, 360), (ha - hp) / (2 * a), rng.uniform(0, 360),
            rng.uniform(0, 360), a, 2.5 + rng.uniform(-0.5, 1.0))
    for _ in range(60):                              # MEO
        add(f"SYN MEO {num[0] + 1}", 55 + rng.uniform(-2, 2),
            rng.uniform(0, 360), 0.002, 0, rng.uniform(0, 360), RE + 20200,
            3.0)
    for _ in range(60):                              # GEO
        add(f"SYN GEO {num[0] + 1}", rng.uniform(0, 5), rng.uniform(0, 360),
            3e-4, 0, rng.uniform(0, 360), 42164.0, 2.0)
    for _ in range(800):                             # small debris
        hp = rng.uniform(400, 1400)
        ha = hp + rng.uniform(0, 800)
        a = RE + (hp + ha) / 2
        add(f"SYN DEB {num[0] + 1}", rng.uniform(40, 100), rng.uniform(0, 360),
            (ha - hp) / (2 * a), rng.uniform(0, 360), rng.uniform(0, 360), a,
            7.5 + rng.uniform(0, 2))
    return els, mag


def star_catalogue(seed: int = 2, maglim: float = 7.0) -> catalog.Stars:
    """Random all-sky stars with a realistic count-magnitude law:
    N(<m) ~ 5000 * 10^(0.5 (m - 6)) (about 1600 brighter than 5)."""
    rng = np.random.default_rng(seed)
    n = int(5000 * 10 ** (0.5 * (maglim - 6)))
    u = rng.uniform(0, 1, n)
    mag = maglim + 2 * np.log10(u)                    # inverse CDF
    mag = mag[mag > -1.0]
    n = len(mag)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    o = np.argsort(mag)
    return catalog.Stars(ra=ra[o], dec=dec[o], mag=mag[o].astype(np.float32),
                         ident=np.arange(n)[o], epoch=2000.0)


def sun_dir_gcrs(jd_tt) -> np.ndarray:
    """Low-precision Sun direction (Astronomical Almanac, ~0.01 deg)."""
    n = np.asarray(jd_tt) - 2451545.0
    L = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam),
                     np.sin(eps) * np.sin(lam)], -1)


# ------------------------------------------------------------------ world
@dataclass
class World:
    lat: float = 46.55                     # deg, geodetic
    lon: float = 7.98                      # deg east
    height_km: float = 1.9
    t0: datetime = datetime(2025, 2, 14, 0, 30, 0, tzinfo=timezone.utc)
    spf: float = 30.0                      # s between frames
    exposure: float = 25.0                 # s
    n_frames: int = 600                    # ends near nautical twilight
    az: float = 250.0                      # boresight azimuth, deg (N=0, E=90)
    alt: float = 25.0                      # boresight altitude
    roll: float = 2.0                      # deg
    f: float = 1400.0                      # px/rad
    k1: float = 0.02
    k2: float = 0.0
    width: int = 3840
    height: int = 2160
    ridge_deg: float = 1.5                 # mean skyline altitude
    refract: float = 0.0                   # atmosphere density (0: none)
    pan: float = 0.0                       # deg per frame, about the vertical
    tilt: float = 0.0                      # deg: a slow tilt wobble
    tracked: bool = False                  # on a mount following the stars
    seed: int = 0
    extra: dict = field(default_factory=dict)

    # -------------------------------------------------------- time, frames
    def t(self, k, offs=0.0):
        """skyfield Time of frame k's mid-exposure plus `offs` seconds."""
        s = np.asarray(k, float) * self.spf + np.asarray(offs, float)
        return TS.tt_jd(TS.from_datetime(self.t0).tt + s / 86400)

    @property
    def observer(self):
        return wgs84.latlon(self.lat, self.lon, elevation_m=self.height_km * 1e3)

    @cached_property
    def C(self) -> np.ndarray:
        """Rows: camera right, down, forward in local ENU."""
        az, alt = math.radians(self.az), math.radians(self.alt)
        fwd = np.array([math.cos(alt) * math.sin(az),
                        math.cos(alt) * math.cos(az), math.sin(alt)])
        right = unit(np.cross(fwd, [0, 0, 1.0]))
        down = np.cross(fwd, right)
        r = math.radians(self.roll)
        right, down = (math.cos(r) * right + math.sin(r) * down,
                       -math.sin(r) * right + math.cos(r) * down)
        return np.stack([right, down, fwd])

    @property
    def moving(self) -> bool:
        return bool(self.pan or self.tilt or self.tracked)

    def C_at(self, kf) -> np.ndarray:
        """(N, 3, 3) the camera's axes in ENU at (fractional) frames kf: the
        rig pans about the vertical and tilts about the camera's own
        sideways axis, from the pose `C` at mid-clip."""
        kf = np.atleast_1d(np.asarray(kf, float))
        if not self.moving:
            return np.broadcast_to(self.C, (len(kf), 3, 3))
        if self.tracked:
            # fixed among the stars: the pose `C` at mid-clip, carried
            # back into each frame's own east-north-up
            G0 = self.C @ self.enu(self.t(self.n_frames // 2))
            return np.einsum("ij,kjn->nik", G0, self.enu(self.t(kf)))
        u = kf - self.n_frames // 2
        th = np.radians(self.pan) * u
        ta = np.radians(self.tilt) * np.sin(u / 90.0)
        c, s = np.cos(th), np.sin(th)
        Rz = np.zeros((len(kf), 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = c, -s, s, c
        Rz[:, 2, 2] = 1
        c, s = np.cos(ta), np.sin(ta)
        Rx = np.zeros((len(kf), 3, 3))
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = c, -s, s, c
        return np.einsum("nij,jk,nlk->nil", Rx, self.C, Rz)

    def gcrs_to_cam_at(self, kf, t) -> np.ndarray:
        """GCRS -> camera at (fractional) frames kf, times t (skyfield)."""
        if not self.moving:
            return self.gcrs_to_cam(t)
        R = self.enu(t)
        Ck = self.C_at(kf)
        if R.ndim == 3:
            return np.einsum("nij,jkn->nik", Ck, R)
        return Ck[0] @ R

    def enu(self, t) -> np.ndarray:
        """GCRS -> local ENU rotation(s) at skyfield Time t."""
        R = self.observer.rotation_at(t)               # rows: N, E, U
        return R[[1, 0, 2]]

    def gcrs_to_cam(self, t) -> np.ndarray:
        R = self.enu(t)
        if R.ndim == 3:
            return np.einsum("ij,jkn->nik", self.C, R)
        return self.C @ R

    @property
    def up_cam(self) -> np.ndarray:
        return self.C[:, 2].copy()

    def true_camera(self, k0: int | None = None) -> Camera:
        """The solver's camera model with this world's exact values."""
        k0 = self.n_frames // 2 if k0 is None else k0
        t = self.t(k0)
        M = self.gcrs_to_cam(t)
        axis = t.M[2]                                    # pole of date, GCRS
        from scipy.spatial.transform import Rotation
        p = np.r_[Rotation.from_matrix(M).as_rotvec(),
                  SIDEREAL_DEG_PER_DAY * self.spf / 86400, self.f,
                  self.width / 2, self.height / 2, self.k1, self.k2,
                  axis[0] / axis[2], axis[1] / axis[2]]
        turn = None
        if self.moving:
            from scipy.spatial.transform import Rotation
            Ck = self.C_at(np.arange(self.n_frames))
            C0 = self.C_at(k0)[0]
            turn = Rotation.from_matrix(
                np.einsum("nij,kj->nik", Ck, C0)).as_rotvec()
            # the rig's pose at k0 is the model's Earth-fixed one
            p[:3] = Rotation.from_matrix(C0 @ self.enu(t)).as_rotvec()
        cam = Camera(p, k0, self.width, self.height, turn=turn)
        return cam.with_up(self.up_cam, self.refract) if self.refract \
            else cam

    def refracted(self, r: np.ndarray) -> np.ndarray:
        """Camera-frame true rays -> as the atmosphere bends them."""
        if not self.refract:
            return r
        alt = np.degrees(np.arcsin(np.clip(r @ self.up_cam, -1, 1)))
        return lift(r, self.up_cam, refraction(alt, self.refract))

    def project_cam(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = r[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = r[..., 0] / z, r[..., 1] / z
        rr = x * x + y * y
        d = 1 + self.k1 * rr + self.k2 * rr * rr
        # far outside the field a distortion polynomial can fold back into
        # the frame; a real lens can't see there
        rlim = 1.3 * np.hypot(self.width, self.height) / 2 / self.f
        z = np.where(rr > rlim ** 2, -1.0, z)
        return np.stack([self.width / 2 + self.f * x * d,
                         self.height / 2 + self.f * y * d], -1), z

    # -------------------------------------------------------- landscape
    def ridge_alt(self, az_deg) -> np.ndarray:
        """Skyline altitude (deg) by azimuth: hills around `ridge_deg`."""
        a = np.radians(az_deg)
        return self.ridge_deg * (1 + 0.6 * np.sin(3 * a + 1.0)
                                 + 0.3 * np.sin(7 * a + 0.3))

    def _altaz_of_pixels(self, xy: np.ndarray, C=None):
        # invert the lens numerically (camera-fixed: the same every frame)
        xd = (xy[:, 0] - self.width / 2) / self.f
        yd = (xy[:, 1] - self.height / 2) / self.f
        x, y = xd.copy(), yd.copy()
        for _ in range(30):
            rr = x * x + y * y
            d = 1 + self.k1 * rr + self.k2 * rr * rr
            x, y = xd / d, yd / d
        C = self.C if C is None else C
        r = unit(np.stack([x, y, np.ones_like(x)], -1)) @ C        # ENU
        alt = np.degrees(np.arcsin(np.clip(r[:, 2], -1, 1)))
        az = np.degrees(np.arctan2(r[:, 0], r[:, 1])) % 360
        return alt, az

    @cached_property
    def ground(self) -> np.ndarray:
        """uint8 mask, 1 = land (below the skyline)."""
        s = 4
        ys, xs = np.mgrid[0:self.height:s, 0:self.width:s]
        alt, az = self._altaz_of_pixels(
            np.column_stack([xs.ravel(), ys.ravel()]).astype(float))
        g = (alt < self.ridge_alt(az)).reshape(xs.shape).astype(np.uint8)
        return cv2.resize(g, (self.width, self.height),
                          interpolation=cv2.INTER_NEAREST)

    def ground_at(self, k) -> np.ndarray:
        """The land mask at frame k (the rig moves it)."""
        if not self.moving:
            return self.ground
        cache = self.__dict__.setdefault("_ground_cache", {})
        k = int(round(k))
        if k not in cache:
            if len(cache) > 8:
                cache.clear()
            s = 4
            ys, xs = np.mgrid[0:self.height:s, 0:self.width:s]
            alt, az = self._altaz_of_pixels(
                np.column_stack([xs.ravel(), ys.ravel()]).astype(float),
                self.C_at(k)[0])
            g = (alt < self.ridge_alt(az)).reshape(xs.shape).astype(np.uint8)
            cache[k] = cv2.resize(g, (self.width, self.height),
                                  interpolation=cv2.INTER_NEAREST)
        return cache[k]

    @cached_property
    def _land_texture(self) -> np.ndarray:
        """Texture by (azimuth, altitude), 0.05 deg a cell: the land's own."""
        rng = np.random.default_rng(self.seed + 7)
        t = rng.normal(0, 1, (700, 7200)).astype(np.float32)
        t = cv2.GaussianBlur(t, (0, 0), 8.0 * 0.075 / 0.05)
        return np.clip(6 + 3 * t / t.std(), 0, None)

    def terrain_at(self, k) -> np.ndarray:
        """The land's texture as frame k sees it."""
        if not self.moving:
            return self.terrain
        s = 2
        ys, xs = np.mgrid[0:self.height:s, 0:self.width:s]
        alt, az = self._altaz_of_pixels(
            np.column_stack([xs.ravel(), ys.ravel()]).astype(float),
            self.C_at(k)[0])
        mx = (az / 0.05).reshape(xs.shape).astype(np.float32)
        my = ((5.0 - alt) / 0.05).reshape(xs.shape).astype(np.float32)
        tex = cv2.remap(self._land_texture, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
        return cv2.resize(tex, (self.width, self.height),
                          interpolation=cv2.INTER_LINEAR)

    @cached_property
    def terrain(self) -> np.ndarray:
        """Camera-fixed land texture: dark, coarse blotches (shrubs, rock,
        snow; nothing star-like), which is what lets a foreground detector
        tell land from dark sky."""
        rng = np.random.default_rng(self.seed + 7)
        t = rng.normal(0, 1, (self.height, self.width)).astype(np.float32)
        t = cv2.GaussianBlur(t, (0, 0), 8.0)
        return np.clip(6 + 3 * t / t.std(), 0, None)

    # -------------------------------------------------------- stars
    @cached_property
    def stars(self) -> catalog.Stars:
        return star_catalogue(self.seed + 2)

    @cached_property
    def star_vecs(self) -> np.ndarray:
        s = self.stars
        ra, dec = np.radians(s.ra), np.radians(s.dec)
        return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                         np.sin(dec)], -1)

    def star_pixels(self, k: int, maglim: float = 6.5):
        """(xy, mag, index) of catalogue stars visible in frame k."""
        keep = self.stars.mag <= maglim
        r = self.refracted(self.star_vecs[keep] @
                           self.gcrs_to_cam_at(k, self.t(k)).T)
        xy, z = self.project_cam(r)
        ok = (z > 0.1) & (xy[:, 0] >= 0) & (xy[:, 0] < self.width - 1) & \
            (xy[:, 1] >= 0) & (xy[:, 1] < self.height - 1)
        idx = np.flatnonzero(keep)[ok]
        xy = xy[ok]
        ground = self.ground_at(k)
        sky = ground[xy[:, 1].astype(int), xy[:, 0].astype(int)] == 0
        return xy[sky], self.stars.mag[idx[sky]], idx[sky]

    # -------------------------------------------------------- satellites
    @cached_property
    def _catalogue(self):
        epoch = self.t0 - timedelta(hours=10)
        return constellation(epoch, self.seed + 1)

    @property
    def elements(self) -> list[tles.Element]:
        return self._catalogue[0]

    @property
    def tle_text(self) -> str:
        return "".join(f"0 {e.name}\n{e.line1}\n{e.line2}\n"
                       for e in self.elements)

    @cached_property
    def _satrecs(self):
        from sgp4.api import Satrec
        return [Satrec.twoline2rv(e.line1, e.line2) for e in self.elements]

    def sat_gcrs(self, t) -> np.ndarray:
        """(nsat, nt, 3) GCRS km for skyfield Time array t."""
        jd = np.atleast_1d(t.whole)
        fr = np.atleast_1d(t.tai_fraction - t._leap_seconds() / 86400)
        e, r, v = SatrecArray(self._satrecs).sgp4(jd, fr)
        R = TEME.rotation_at(t)                     # gcrs -> teme
        if R.ndim == 2:
            R = R[..., None]
        out = np.einsum("jin,snj->sni", R, r)
        out[e != 0] = np.nan
        return out

    def _pixels_of(self, pos, obs, t, kf=None):
        """Satellite positions (s, n, 3) seen from obs (n, 3) at times t
        (fractional frames kf, for a rig) -> pixels (s, n, 2), z, altitude
        (deg), range (km)."""
        d = pos - obs[None]
        rng_km = np.linalg.norm(d, axis=-1)
        u = d / rng_km[..., None]
        G = self.gcrs_to_cam(t) if kf is None else \
            self.gcrs_to_cam_at(kf, t)              # (n,3,3)
        r = np.einsum("nij,snj->sni", G, u)
        r = self.refracted(r.reshape(-1, 3)).reshape(r.shape)
        E = self.enu(t)                             # (3,3,n)
        alt = np.degrees(np.arcsin(np.clip(np.einsum("jn,snj->sn", E[2], u),
                                           -1, 1)))
        xy, z = self.project_cam(r)
        return xy, z, alt, rng_km

    def segments(self, frames=None, sub: int = 26, vis_limit: float = 0.01,
                 mag_limit: float = 6.0
                 ) -> list[dict]:
        """Satellite trails as rendered: each exposure sampled at `sub`
        instants, kept where the object is sunlit, above the skyline, in
        frame, and bright enough for its angular speed. Each dict has the
        truth (norad, name, frame) and the pixel path `xy` (m, 2); `x1..y2`
        are the path ends, `clip_a/b` whether an end was cut short by the
        frame edge, the skyline or the Earth's shadow rather than by the
        exposure."""
        frames = range(self.n_frames) if frames is None else frames
        mag = self._catalogue[1]
        m1000 = np.array([mag[e.norad] for e in self.elements])
        offs = np.linspace(-self.exposure / 2, self.exposure / 2, sub)
        out = []
        for k in frames:
            t = self.t(k, offs)
            obs = self.observer.at(t).position.km.T       # (sub,3)
            pos = self.sat_gcrs(t)                         # (s,sub,3)
            xy, z, alt, rng_km = self._pixels_of(pos, obs, t,
                                                 k + offs / self.spf)
            sun = sun_dir_gcrs(t.tt)                       # (sub,3)
            sproj = np.einsum("snj,nj->sn", pos, sun)
            perp = np.linalg.norm(pos - sproj[..., None] * sun[None], axis=-1)
            lit = (sproj > 0) | (perp > RE)
            inside = (z > 0.1) & (xy[..., 0] >= 0) & \
                (xy[..., 0] < self.width) & (xy[..., 1] >= 0) & \
                (xy[..., 1] < self.height)
            ok = inside & lit & (alt > 0) & ~np.isnan(xy).any(-1)
            for s in np.flatnonzero(ok.any(1)):
                w = np.flatnonzero(ok[s])
                if len(w) < 2:
                    continue
                p = xy[s, w]
                gx = np.clip(p[:, 0].astype(int), 0, self.width - 1)
                gy = np.clip(p[:, 1].astype(int), 0, self.height - 1)
                onsky = self.ground_at(k)[gy, gx] == 0
                if onsky.sum() < 2:
                    continue
                w, p = w[onsky], p[onsky]
                # a run of consecutive samples (skyline gaps split trails;
                # keep the longest piece)
                runs = np.split(np.arange(len(w)),
                                np.flatnonzero(np.diff(w) > 1) + 1)
                run = max(runs, key=len)
                w, p = w[run], p[run]
                if len(w) < 2:
                    continue
                length = float(np.linalg.norm(p[-1] - p[0]))
                ang_speed = np.degrees(length / self.f) / \
                    max(offs[w[-1]] - offs[w[0]], 1e-3)
                rmid = rng_km[s, w].mean()
                m = m1000[s] + 5 * math.log10(rmid / 1000)
                if m > mag_limit or \
                        10 ** (-0.4 * m) / max(ang_speed, 0.01) < vis_limit:
                    continue
                if length < 30:
                    continue
                e = self.elements[s]
                out.append(dict(
                    k=int(k), norad=e.norad, name=e.name, xy=p,
                    x1=float(p[0, 0]), y1=float(p[0, 1]),
                    x2=float(p[-1, 0]), y2=float(p[-1, 1]),
                    clip_a=bool(w[0] > 0), clip_b=bool(w[-1] < sub - 1),
                    mag=float(m), rate=float(ang_speed)))
        return out

    # -------------------------------------------------------- rendering
    def render(self, k: int, segs=(), *, noise: float = 2.0,
               sky: float = 18.0, rng=None, extra=()) -> np.ndarray:
        """BGR uint8 frame k: sky glow, stars, land, and the given trails.
        `extra`: (GCRS unit vector, magnitude) point sources off the
        catalogue (planets)."""
        rng = np.random.default_rng(self.seed * 100003 + k) if rng is None \
            else rng
        h, w = self.height, self.width
        img = np.full((h, w), sky, np.float32)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        img += 10 * yy ** 3                               # glow near horizon
        xy, mag, _ = self.star_pixels(k)
        if len(extra):
            G = self.gcrs_to_cam_at(k, self.t(k))
            exy, ez = self.project_cam(np.array([v for v, _ in extra]) @ G.T)
            ok = (ez > 0.1) & (exy[:, 0] >= 0) & (exy[:, 0] < w) & \
                (exy[:, 1] >= 0) & (exy[:, 1] < h)
            xy = np.vstack([xy, exy[ok]])
            mag = np.r_[mag, np.array([m for _, m in extra])[ok]]
        amp = np.minimum(240 * 10 ** (-0.4 * (mag - 1.0)), 2000)
        r = 4
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
        for (x, y), a in zip(xy, amp):
            if a < 1.5:
                continue
            ix, iy = int(round(x)), int(round(y))
            if ix - r < 0 or iy - r < 0 or ix + r >= w or iy + r >= h:
                continue
            g = np.exp(-(((ox + ix - x) ** 2 + (oy + iy - y) ** 2)
                         / (2 * 1.2 ** 2)))
            img[iy - r:iy + r + 1, ix - r:ix + r + 1] += a * g
        for s in segs:
            b = float(np.clip(40 * 10 ** (-0.4 * (s["mag"] - 4)), 25, 180))
            _trail(img, s["xy"][0], s["xy"][-1], b)
        img = np.where(self.ground_at(k) > 0, self.terrain_at(k), img)
        img += rng.normal(0, noise, img.shape).astype(np.float32)
        g = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _trail(img: np.ndarray, p0, p1, peak: float, sigma: float = 1.0) -> None:
    """Add the exact image of a straight trail p0 -> p1 through a Gaussian
    PSF: across the trail a Gaussian, along it a step blurred at each end —
    so the true ends sit exactly at half maximum (a drawn stroke would add
    caps past them)."""
    from scipy.special import erf
    h, w = img.shape
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-6:
        return
    d = (p1 - p0) / L
    pad = 5 * sigma
    x0 = int(max(0, np.floor(min(p0[0], p1[0]) - pad)))
    x1 = int(min(w, np.ceil(max(p0[0], p1[0]) + pad) + 1))
    y0 = int(max(0, np.floor(min(p0[1], p1[1]) - pad)))
    y1 = int(min(h, np.ceil(max(p0[1], p1[1]) + pad) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rx, ry = xs - p0[0], ys - p0[1]
    t = rx * d[0] + ry * d[1]
    u = -rx * d[1] + ry * d[0]
    s2 = np.sqrt(2) * sigma
    along = 0.5 * (erf(t / s2) - erf((t - L) / s2))
    img[y0:y1, x0:x1] += peak * np.exp(-u * u / (2 * sigma ** 2)) * along
