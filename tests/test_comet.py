"""Comets: found as green fuzzy spots that keep their place among the stars,
and dated from where they are (orbits in MPC's one-line format; the dating
tests need DE421 in the data directory)."""

import cv2
import numpy as np
import pytest

from skyfind import comet, data
from skyfind.camera import radec_to_vec, vec_to_radec

from .synth import World

# MPC CometEls.txt lines (orbits as of 2026-09-25)
LEMMON = ("    CK25A060  2025 11  8.5252  0.529829  0.995619  132.9562  "
          "108.0980  143.6635  20260925  13.2  4.0  C/2025 A6 (Lemmon)"
          "                                       MPEC 2026-P55")
SWAN = ("    CK25R020  2025 09 12.8095  0.504279  0.994163  308.3763  "
        "335.3135    4.4729  20260925  12.4  4.0  C/2025 R2 (SWAN)"
        "                                         MPEC 2026-N12")
SPAHR = ("0171P         2025 09 24.9224  1.766126  0.502674  347.0302  "
         "101.6913   21.9546  20260925  13.5  4.0  171P/Spahr"
         "                                               MPC194201")

needs_de421 = pytest.mark.skipif(
    not (data.data_dir() / "de421.bsp").exists(),
    reason="DE421 not in the data directory")

def _tt(*a):
    return data.timescale().utc(*a).tt

def test_orbit_lines_parse():
    (o,) = comet.parse(LEMMON)
    assert o.name == "C/2025 A6 (Lemmon)"
    assert abs(o.q - 0.529829) < 1e-9 and abs(o.incl - 143.6635) < 1e-9
    assert (o.g, o.k) == (13.2, 4.0)
    # MPC gives the perihelion in TT
    assert abs(o.peri_tt - data.timescale().tt(2025, 11, 8.5252).tt) < 1e-6

@needs_de421
def test_lemmon_was_in_bootes_that_evening():
    """Where a real clip's first frame shows it: RA 225.5, Dec +23.2."""
    (o,) = comet.parse(LEMMON)
    v = comet.positions(o, np.array([_tt(2025, 10, 23, 19, 48)]))[0]
    ra, dec = vec_to_radec(v)
    assert abs(float(ra) - 225.53) < 0.3 and abs(float(dec) - 23.20) < 0.3

@needs_de421
def test_two_comets_agree_on_the_night_and_a_faint_one_doesnt_count():
    orbits = comet.parse("\n".join((LEMMON, SWAN, SPAHR)))
    t0 = _tt(2025, 10, 23, 19, 48)
    spf = 13.144
    objs = []
    for o, k in ((orbits[0], 0), (orbits[1], 200)):
        v = comet.positions(o, np.array([t0 + k * spf / 86400]))[0]
        ra, dec = vec_to_radec(v)
        objs.append(dict(ra=float(ra) + 0.03, dec=float(dec) - 0.03, k_mid=k))
    c = comet.search(objs, spf, orbit_list=orbits, years=(2024, 2026),
                     log=None)
    assert c and c[0]["matched"] == 2
    assert set(c[0]["comets"]) == {"C/2025 A6 (Lemmon)", "C/2025 R2 (SWAN)"}
    assert abs(c[0]["tt"] - t0) * 24 < 1.0
    assert all("Spahr" not in n for x in c for n in x["comets"])

def _green_blob(img, x, y, sigma=2.5, amp=90.0):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    g = amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    out = img.astype(np.float32)
    out[..., 1] += g                           # green
    out[..., 0] += 0.4 * g                     # a little blue: cyan-green
    return np.clip(out, 0, 255).astype(np.uint8)

def test_a_green_fuzzy_spot_stands_out_from_the_stars():
    rng = np.random.default_rng(1)
    img = np.full((600, 800, 3), 20, np.uint8)
    for x, y in rng.uniform([20, 20], [780, 580], (300, 2)).astype(int):
        cv2.circle(img, (int(x), int(y)), 1, (230, 230, 230), -1)   # stars
    img = _green_blob(img, 420, 250)
    p = comet.peaks(img)
    assert len(p) and np.hypot(p[0, 0] - 420, p[0, 1] - 250) < 3

def test_a_comet_keeps_its_place_among_the_stars():
    """Rendered frames of the synthetic sky with a green fuzzy spot at one
    sky position: found there, in every frame."""
    w = World(width=1280, height=720, f=760.0)
    cam = w.true_camera()
    v = cam.unproject(np.array([[640.0, 200.0]]), 0)
    frames = {}
    for k in (0, 60, 120, 180):
        (x, y), = cam.project(v, k)[0]
        frames[k] = _green_blob(w.render(k), x, y)
    cands = comet.candidates(cam, frames, w.stars, w.ground == 0, log=None)
    assert cands and len(cands[0]["frames"]) == 4
    got = radec_to_vec(cands[0]["ra"], cands[0]["dec"])
    assert np.degrees(np.arccos(np.clip(got @ v[0], -1, 1))) < 0.15
