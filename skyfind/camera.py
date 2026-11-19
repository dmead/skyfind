"""A tripod camera watching the sky turn (and perhaps turning itself).

For frame k, a sky direction v (GCRS unit vector) is seen along the camera
ray

    ray = M @ R(a, -omega * (k - k0)) @ v

M is the camera orientation at the reference frame k0, R(a, t) a rotation by
t about the sky's rotation axis a, omega the sky rotation per frame. The axis
is the celestial pole *of date*, which precession has moved ~0.14 deg from
the J2000 pole by 2024 — enough to bend a night's fit by a few pixels — so it
is fitted, not assumed. Camera axes: x right, y down, z forward. Pixels come
from a gnomonic projection with two radial distortion terms:

    x = cx + f * xn * (1 + k1 r^2 + k2 r^4),   xn = ray_x / ray_z, ...

**Refraction.** The atmosphere lifts everything outside it (stars and
satellites alike) towards the zenith, by ~0.5 deg at the horizon, 5' at
10 deg. The camera is Earth-fixed, so "up" never moves in the camera frame
and refraction is a fixed warp of the image; with `up` (camera-frame up
vector, from the skyline) set and `refr` (the atmosphere's density relative
to a standard sea-level one, ~0.8 at 2 km) non-zero, `project` refracts true
directions and `unproject` returns true (unrefracted) ones. Without it a
radial lens polynomial half-absorbs the refraction and leaves altitude-
dependent errors of several arcminutes near the horizon — which distant
satellites turn into tens of km of site error.

**A moving camera.** A motion-control rig pans, tilts or slides the camera
through the clip. The model above then describes the camera as it sat at
k0, Earth-fixed; `turn` (rotation vectors, one per frame) says how the real
camera is turned from that pose at each frame:

    ray = G_k @ refract(M @ R(a, -omega * (k - k0)) @ v),   G_k0 = I

Refraction and `up` stay in the Earth-fixed frame, where up doesn't move.
A slide alone doesn't move the stars (they are infinitely far); only the
turn matters, and `calibrate.follow` measures it (the stars give each
frame's orientation, the landscape which part of it is Earth's).

Parameter vector (`Camera.p`): rx ry rz (M as a rotation vector), omega
(deg/frame), f, cx, cy, k1, k2, ax, ay (axis = unit([ax, ay, 1])), refr.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ("rx", "ry", "rz", "omega", "f", "cx", "cy", "k1", "k2", "ax", "ay",
         "refr")
SIDEREAL_DEG_PER_DAY = 360.98564736629


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def radec_to_vec(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                     np.sin(dec)], -1)


def vec_to_radec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = unit(np.asarray(v, float))
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360,
            np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))))


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R a - b| (rows are vectors)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


def rotate(v: np.ndarray, axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rotate rows of v about unit `axis` by `ang` (rad, scalar or per row)."""
    ang = np.asarray(ang, float)[..., None]
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (v @ axis)[..., None] * (1 - c)


def refraction(alt_true_deg, scale: float = 1.0) -> np.ndarray:
    """Astronomical refraction (deg) at true altitude (deg), Saemundsson's
    formula for a standard atmosphere times `scale` (pressure / 1010 hPa x
    283 K / temperature)."""
    h = np.maximum(np.asarray(alt_true_deg, float), -1.5)
    return scale * 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0


def lift(r: np.ndarray, up: np.ndarray, ang_deg: np.ndarray) -> np.ndarray:
    """Tilt unit rays r towards `up` by `ang_deg` (negative: away)."""
    e = up - (r @ up)[:, None] * r
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    a = np.radians(ang_deg)[:, None]
    return r * np.cos(a) + e * np.sin(a)


@dataclass
class Camera:
    p: np.ndarray
    k0: int
    width: int
    height: int
    meta: dict = field(default_factory=dict)
    up: np.ndarray | None = None       # camera-frame up (for refraction)
    # each frame's capture index when a frame-rate conversion dropped
    # frames (remove-satellites' Track.clock); None = frame k is capture k
    clock: np.ndarray | None = None
    # a moving camera: per frame, the rotation vector (rad, camera frame)
    # taking the Earth-fixed pose at k0 to the frame's; None = fixed
    turn: np.ndarray | None = None

    def __post_init__(self):
        p = np.asarray(self.p, float)
        self.p = np.r_[p, np.zeros(len(NAMES) - len(p))] if \
            len(p) < len(NAMES) else p.copy()
        if self.up is not None:
            self.up = unit(np.asarray(self.up, float))
        if self.clock is not None:
            self.clock = np.asarray(self.clock, float)
        if self.turn is not None:
            self.turn = np.asarray(self.turn, float).reshape(-1, 3)
            self._G = Rotation.from_rotvec(self.turn).as_matrix()

    @property
    def moving(self) -> bool:
        return self.turn is not None

    def G(self, k) -> np.ndarray:
        """(N, 3, 3) turn of the camera at frame(s) k from its Earth-fixed
        pose (fractional frames interpolate between the neighbours)."""
        k = np.atleast_1d(np.asarray(k, float))
        n = len(self.turn)
        k = np.clip(k, 0, n - 1)
        i = np.minimum(np.floor(k).astype(int), n - 2) if n > 1 else \
            np.zeros(len(k), int)
        t = (k - i)[:, None]
        if n == 1 or not (t > 0).any():
            return self._G[np.clip(np.rint(k).astype(int), 0, n - 1)]
        rv = self.turn[i] * (1 - t) + self.turn[i + 1] * t
        return Rotation.from_rotvec(rv).as_matrix()

    def _turned(self, r: np.ndarray, k) -> np.ndarray:
        """Earth-fixed-pose rays -> the frame's (rows of r at frames k)."""
        if self.turn is None:
            return r
        k = np.broadcast_to(np.asarray(k, float), (len(r),))
        return np.einsum("nij,nj->ni", self.G(k), r)

    def _unturned(self, r: np.ndarray, k) -> np.ndarray:
        if self.turn is None:
            return r
        k = np.broadcast_to(np.asarray(k, float), (len(r),))
        return np.einsum("nji,nj->ni", self.G(k), r)

    def s(self, k) -> np.ndarray:
        """Capture index of frame(s) k — the unit of sky rotation and time
        (fractional k interpolates)."""
        k = np.asarray(k, float)
        if self.clock is None:
            return k
        return np.interp(k, np.arange(len(self.clock)), self.clock)

    # ---------------------------------------------------------- parameters
    @property
    def M(self) -> np.ndarray:
        return Rotation.from_rotvec(self.p[:3]).as_matrix()

    @property
    def omega(self) -> float:
        return float(self.p[3])

    @property
    def f(self) -> float:
        return float(self.p[4])

    @property
    def axis(self) -> np.ndarray:
        return unit(np.array([self.p[9], self.p[10], 1.0]))

    @property
    def refracts(self) -> bool:
        return self.up is not None and self.p[11] != 0

    @property
    def seconds_per_frame(self) -> float:
        """Real time between captures: the sky turns at the sidereal rate
        (omega is per capture interval)."""
        return self.omega / SIDEREAL_DEG_PER_DAY * 86400.0

    def with_params(self, p) -> "Camera":
        return Camera(p, self.k0, self.width, self.height, dict(self.meta),
                      self.up, self.clock, self.turn)

    def with_up(self, up, refr: float = 0.85) -> "Camera":
        """This camera, refracting about `up` with density `refr`."""
        p = self.p.copy()
        p[11] = refr
        return Camera(p, self.k0, self.width, self.height, dict(self.meta),
                      up, self.clock, self.turn)

    # ---------------------------------------------------------- mapping
    def _angle(self, k) -> np.ndarray:
        return -np.radians(self.p[3]) * (self.s(k) - self.s(self.k0))

    def rays(self, v: np.ndarray, k) -> np.ndarray:
        """Camera-frame unit rays, as seen (refracted), for GCRS unit
        vectors v (N,3) at frame(s) k."""
        v = np.atleast_2d(v)
        k = np.broadcast_to(np.asarray(k, float), (len(v),))
        r = rotate(v, self.axis, self._angle(k)) @ self.M.T
        if self.refracts:
            alt = np.degrees(np.arcsin(np.clip(r @ self.up, -1, 1)))
            r = lift(r, self.up, refraction(alt, self.p[11]))
        return self._turned(r, k)

    def unrefract(self, r: np.ndarray) -> np.ndarray:
        """Apparent Earth-fixed-pose rays -> true (unrefracted) rays."""
        if not self.refracts:
            return r
        app = np.degrees(np.arcsin(np.clip(r @ self.up, -1, 1)))
        true = app - refraction(app, self.p[11])
        for _ in range(4):                       # h + R(h) = apparent
            true = app - refraction(true, self.p[11])
        return lift(r, self.up, true - app)

    def project(self, v: np.ndarray, k) -> tuple[np.ndarray, np.ndarray]:
        """(pixels (N,2), z (N,)); z <= 0 means behind the camera."""
        return self.ray_to_px(self.rays(v, k))

    def ray_to_px(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f, cx, cy, k1, k2 = self.p[4:9]
        z = r[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = r[:, 0] / z, r[:, 1] / z
        rr = x * x + y * y
        d = 1 + k1 * rr + k2 * rr * rr
        return np.stack([cx + f * x * d, cy + f * y * d], -1), z

    def px_to_ray(self, xy: np.ndarray) -> np.ndarray:
        """Pixels -> apparent camera-frame rays (what the lens saw)."""
        f, cx, cy, k1, k2 = self.p[4:9]
        xy = np.atleast_2d(np.asarray(xy, float))
        xd, yd = (xy[:, 0] - cx) / f, (xy[:, 1] - cy) / f
        x, y = xd.copy(), yd.copy()
        for _ in range(20):                           # invert the distortion
            rr = x * x + y * y
            d = 1 + k1 * rr + k2 * rr * rr
            x, y = xd / d, yd / d
        return unit(np.stack([x, y, np.ones_like(x)], -1))

    def unproject(self, xy: np.ndarray, k) -> np.ndarray:
        """Pixels at frame(s) k -> true GCRS unit vectors."""
        r = self.px_to_ray(xy)
        k = np.broadcast_to(np.asarray(k, float), (len(r),))
        r = self.unrefract(self._unturned(r, k)) @ self.M      # M^T r
        return rotate(r, self.axis, -self._angle(k))

    def cam_to_gcrs(self, r: np.ndarray, k) -> np.ndarray:
        """Earth-fixed-pose camera vectors (N,3) at frame(s) k -> GCRS (a
        pure frame change: no refraction). Up (`self.up`, the skyline's) is
        one of these: for a moving camera, the frame's own vectors go
        through `frame_to_gcrs`."""
        r = np.atleast_2d(r) @ self.M
        k = np.broadcast_to(np.asarray(k, float), (len(r),))
        return rotate(r, self.axis, -self._angle(k))

    def gcrs_to_cam(self, v: np.ndarray, k) -> np.ndarray:
        """GCRS vectors (N,3) at frame(s) k -> Earth-fixed-pose camera
        vectors: `cam_to_gcrs` back (no refraction)."""
        v = np.atleast_2d(v)
        k = np.broadcast_to(np.asarray(k, float), (len(v),))
        return rotate(v, self.axis, self._angle(k)) @ self.M.T

    def frame_to_gcrs(self, r: np.ndarray, k) -> np.ndarray:
        """Vectors in frame k's own camera frame -> GCRS."""
        r = np.atleast_2d(r)
        return self.cam_to_gcrs(self._unturned(r, k), k)

    def own_spin(self, spf: float) -> np.ndarray | None:
        """The camera's own turn while it exposes, in GCRS, rad per second,
        averaged over the clip; None for a fixed camera. A point recorded
        `off` s from mid-exposure lands that much further turned from where
        the frame's model puts it, on top of the sky's turn (`axis`,
        `omega`). The average serves every frame: the rig's rate changes
        slowly, and the whole effect is a few hundredths of a degree."""
        if self.turn is None:
            return None
        n = len(self.turn)
        k = np.arange(n, dtype=float)
        G = self._G
        h = Rotation.from_matrix(np.einsum("nji,njk->nik", G[:-1], G[1:])
                                 ).as_rotvec()             # (n-1, 3) per frame
        ds = np.diff(self.s(k)) * spf
        u = rotate(h @ self.M, self.axis, -self._angle(k[:-1] + 0.5))
        return (u / np.maximum(ds, 1e-9)[:, None]).mean(0)

    def tracking(self, spf: float, n_frames: int, pole=None) -> "Camera":
        """This camera, fitted to a clip from a mount that follows the
        stars, as the model has a turning camera: an Earth-fixed pose at k0
        and each frame's turn from it, about the pole at the sidereal rate
        for a frame interval `spf` (the stars can't give it: they stand
        still; `with_interval` moves it). Every pixel keeps its place in
        the sky at every frame: whatever slow drift the fit found (the
        mount's own error) stays in the turn. `pole`: of date, GCRS (the
        fit's axis is noise when nothing turns)."""
        p = self.p.copy()
        p[3] = spf * SIDEREAL_DEG_PER_DAY / 86400.0
        if pole is not None:
            p[9], p[10] = pole[0] / pole[2], pole[1] / pole[2]
        new = Camera(p, self.k0, self.width, self.height, dict(self.meta),
                     None, self.clock)
        k = np.arange(n_frames, dtype=float)
        M = self.M
        Rn = Rotation.from_rotvec(np.outer(new._angle(k), new.axis)
                                  ).as_matrix()
        Ro = Rotation.from_rotvec(np.outer(-self._angle(k), self.axis)
                                  ).as_matrix()
        Gt = np.einsum("ij,njk,nkl,lm->nim", M, Rn, Ro, M.T)
        new.turn = Rotation.from_matrix(np.transpose(Gt, (0, 2, 1))
                                        ).as_rotvec()
        new._G = Rotation.from_rotvec(new.turn).as_matrix()
        return new

    def with_interval(self, spf: float) -> "Camera":
        """A moving camera with another frame interval. The stars fix each
        frame's orientation, so the Earth's turn and the camera's trade:
        what the sky turns less, the camera's turn makes up, about the
        pole (as the Earth-fixed pose sees it)."""
        p = self.p.copy()
        p[3] = np.sign(self.omega or 1.0) * spf * \
            SIDEREAL_DEG_PER_DAY / 86400.0
        cam = Camera(p, self.k0, self.width, self.height, dict(self.meta),
                     self.up, self.clock)
        if self.turn is None:
            return cam
        k = np.arange(len(self.turn), dtype=float)
        d = self._angle(k) - cam._angle(k)
        D = Rotation.from_rotvec(np.outer(d, self.M @ self.axis)).as_matrix()
        cam.turn = Rotation.from_matrix(np.einsum("nij,njk->nik", self._G, D)
                                        ).as_rotvec()
        cam._G = Rotation.from_rotvec(cam.turn).as_matrix()
        return cam

    def in_frame(self, xy: np.ndarray, z: np.ndarray, margin: float = 0.0
                 ) -> np.ndarray:
        return (z > 0.05) & (xy[:, 0] >= margin) & \
            (xy[:, 0] < self.width - margin) & (xy[:, 1] >= margin) & \
            (xy[:, 1] < self.height - margin)

    # ---------------------------------------------------------- io
    def to_dict(self) -> dict:
        d = dict(params=dict(zip(NAMES, map(float, self.p))), k0=self.k0,
                 width=self.width, height=self.height,
                 seconds_per_frame=self.seconds_per_frame, **self.meta)
        if self.up is not None:
            d["up_cam"] = self.up.tolist()
        if self.clock is not None:
            d["clock"] = [round(float(c), 4) for c in self.clock]
        if self.turn is not None:
            d["turn"] = np.round(self.turn, 9).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        p = [d["params"].get(n, 0.0) for n in NAMES]
        meta = {k: v for k, v in d.items()
                if k not in ("params", "k0", "width", "height",
                             "seconds_per_frame", "up_cam", "clock", "turn")}
        return cls(np.array(p), int(d["k0"]), int(d["width"]),
                   int(d["height"]), meta, d.get("up_cam"), d.get("clock"),
                   d.get("turn"))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "Camera":
        return cls.from_dict(json.loads(Path(path).read_text()))
