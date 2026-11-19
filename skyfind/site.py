"""The observer's inertial position at every frame — without knowing where
or when.

The camera fit fixes, frame by frame, how the camera sits against the stars;
the skyline says which way is up in the camera. Together they give the
zenith's direction among the stars at each frame's mid-exposure. A site
hypothesis only adds the geodetic latitude, a small correction `dlst` to the
zenith's sidereal angle (the skyline is good to a degree or two, not better)
and a height. From those the site's GCRS position at each frame follows —
the absolute time is not needed.

That is the key to a blind solve: the unknown clock offset moves only the
satellites; the stars have already pinned the observer. (Moving the observer
by the clock offset as well once produced a phantom offshore solution.) The
longitude is a by-product at the end: where the site's inertial position
sits on the Earth at the solved time.

With no skyline (pure sky) the up vector is only a reference: `dlst` can
then be anything, and the solve finds (lat, dlst) from the satellites
(`solve.free_passes`); `cell` turns a point on the Earth into them.
"""

from __future__ import annotations

import numpy as np

from . import data
from .camera import Camera, unit

WE = 7.2921150e-5                         # Earth's rotation, rad/s
A_WGS84 = 6378.137
F_WGS84 = 1 / 298.257223563
AU_KM = 149597870.7


class Site:
    def __init__(self, camera: Camera, up_cam, t_nominal_tt: float,
                 n_frames: int):
        ts = data.timescale()
        self.P = ts.tt_jd(t_nominal_tt).M            # GCRS -> true of date
        self.pole = self.P[2]                        # pole of date in GCRS
        self.up_cam = unit(np.asarray(up_cam, float))
        ks = np.arange(n_frames, dtype=float)
        z = camera.cam_to_gcrs(np.tile(self.up_cam, (n_frames, 1)), ks)
        zd = z @ self.P.T
        self._zra = np.unwrap(np.arctan2(zd[:, 1], zd[:, 0]))
        self.horizon_lat = float(np.degrees(np.arcsin(np.median(
            np.clip(zd[:, 2], -1, 1)))))

    def zenith_ra(self, k) -> np.ndarray:
        """Right ascension of date (rad) of the camera's zenith at frame k
        (fractional frames interpolate)."""
        return np.interp(np.asarray(k, float), np.arange(len(self._zra)),
                         self._zra)

    def gcrs(self, k, lat, dlst, h=0.0) -> np.ndarray:
        """Site position (km, GCRS) at frame(s) k's mid-exposure for geodetic
        latitude `lat` (deg), sidereal correction `dlst` (deg), height `h`
        (km). Shape (3,) for scalar k, else (..., 3)."""
        e2 = F_WGS84 * (2 - F_WGS84)
        phi = np.radians(lat)
        N = A_WGS84 / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        ra = self.zenith_ra(k) + np.radians(dlst)
        rho = (N + h) * np.cos(phi)
        v = np.stack(np.broadcast_arrays(rho * np.cos(ra), rho * np.sin(ra),
                                         (N * (1 - e2) + h) * np.sin(phi)), -1)
        return v @ self.P                             # P^T v

    def cell(self, k, r) -> tuple[np.ndarray, np.ndarray]:
        """(geodetic latitude, dlst) in deg of surface point(s) r (km, GCRS)
        at frame k's mid-exposure — the inverse of `gcrs` at h = 0."""
        rd = np.asarray(r, float) @ self.P.T
        e2 = F_WGS84 * (2 - F_WGS84)
        psi = np.arctan2(rd[..., 2], np.hypot(rd[..., 0], rd[..., 1]))
        lat = np.degrees(np.arctan(np.tan(psi) / (1 - e2)))
        d = np.arctan2(rd[..., 1], rd[..., 0]) - self.zenith_ra(k)
        return lat, np.degrees((d + np.pi) % (2 * np.pi) - np.pi)

    def turn(self, r: np.ndarray, k_from, k_to) -> np.ndarray:
        """Site position(s) r at frame k_from carried by the Earth's rotation
        to frame k_to."""
        from .camera import rotate
        a = float(self.zenith_ra(k_to) - self.zenith_ra(k_from))
        return rotate(np.atleast_2d(r), self.pole, a).reshape(np.shape(r))

    def velocity(self, r: np.ndarray) -> np.ndarray:
        """Site velocity (km/s, GCRS) at position(s) r."""
        return WE * np.cross(self.pole, r)

    def geodetic(self, k, lat, dlst, h, tt: float) -> tuple[float, float]:
        """(latitude, longitude) of the site at frame k if that frame's
        mid-exposure is TT jd `tt`."""
        from skyfield.api import wgs84
        from skyfield.positionlib import Geocentric
        r = self.gcrs(k, lat, dlst, h)
        t = data.timescale().tt_jd(tt)
        g = wgs84.geographic_position_of(Geocentric(r / AU_KM, t=t))
        return float(g.latitude.degrees), float(g.longitude.degrees)
