"""The Moon dates a clip to a stretch of hours once a lunar month: found
as the glow that stays with the stars, then looked up in the ephemeris."""

import numpy as np

from skyfind import date, moon
from skyfind.camera import vec_to_radec

from .synth import World


def _frames(w, cam, v, ks, flare=None):
    """Frames with an overexposed Moon glow where direction v projects
    (and, with `flare`, a camera-fixed glow as bright)."""
    out = {}
    for k in ks:
        img = np.full((w.height, w.width, 3), 20, np.uint8)
        yy, xx = np.mgrid[:w.height, :w.width]
        for x, y in ([cam.project(v[None], k)[0][0]] +
                     ([flare] if flare is not None else [])):
            g = 255 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 50 ** 2))
            img = np.maximum(img, np.repeat(g[..., None], 3, 2).astype(np.uint8))
        out[k] = img
    return out


def test_the_moon_is_the_glow_that_stays_with_the_stars():
    w = World()
    cam = w.true_camera()
    ks = list(range(100, 500, 50))
    v = cam.unproject(np.array([[w.width * 0.4, w.height * 0.4]]), 300)[0]
    m = moon.find(cam, _frames(w, cam, v, ks), log=None)
    ra, dec = vec_to_radec(v)
    assert m is not None and m["spread_deg"] < 0.05
    assert abs(m["ra"] - ra) < 0.05 and abs(m["dec"] - dec) < 0.05
    # a glow fixed in the frame drifts across the stars: not the Moon
    still = _frames(w, cam, v, [300])[300]
    fixed = {k: still for k in ks}
    assert moon.find(cam, fixed, log=None) is None


def test_the_moon_was_there_once_a_lunar_month():
    ts = date.data.timescale()
    t = ts.utc(2025, 8, 9, 21, 30).tt
    v = date.planet_vectors(np.array([t]), ("moon",))["moon"][0]
    ra, dec = vec_to_radec(v)
    obs = dict(ra=float(ra), dec=float(dec), k=0)
    got = moon.search(obs, 10.0, first_tt=ts.utc(2025, 1, 1).tt,
                      last_tt=ts.utc(2026, 1, 1).tt, log=None)
    assert any(abs(c["tt"] - t) < 0.5 / 24 for c in got)
    gaps = np.diff([c["tt"] for c in got])
    assert np.all(np.abs(gaps - 27.32) < 1.0)                 # sidereal month
    near = min(got, key=lambda c: abs(c["tt"] - t))
    assert near["phase"] in ("full",) or near["elong_deg"] > 150
