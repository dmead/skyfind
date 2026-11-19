"""Plate solve, whole-clip camera fit and horizon on rendered frames."""

import numpy as np
import pytest

from skyfind import calibrate, horizon, platesolve
from skyfind.camera import SIDEREAL_DEG_PER_DAY

from .synth import World

FRAMES = list(range(0, 600, 50)) + [599]


@pytest.fixture(scope="module")
def world():
    return World()


@pytest.fixture(scope="module")
def dets(world):
    sky = (world.ground == 0).astype(np.uint8)
    return {k: calibrate.detect(world.render(k), sky) for k in FRAMES}


def _pairs_ok(world, sol):
    xy, _ = world.project_cam(sol.vec @ world.gcrs_to_cam(
        world.t(sol.frame)).T)
    return np.hypot(*(xy - sol.xy).T) < 2.0


def test_blind_plate_solve(world, dets):
    sol = platesolve.solve(dets[300], world.stars, world.width, world.height,
                           frame=300, log=None)
    assert sol is not None and sol.score >= 25
    assert _pairs_ok(world, sol).mean() > 0.95
    # the centre-star scale is the focal length up to lens distortion
    assert abs(sol.scale / world.f - 1) < 0.1


@pytest.fixture(scope="module")
def fitted(world, dets):
    a = platesolve.solve(dets[200], world.stars, world.width, world.height,
                         frame=200, log=None)
    b = platesolve.solve(dets[450], world.stars, world.width, world.height,
                         frame=450, log=None)
    cam0 = calibrate.initial_camera(a, b, world.width, world.height, 300)
    sky = (world.ground == 0).astype(np.uint8)
    cam, stats = calibrate.refine(cam0, dets, world.stars, sky_mask=sky,
                                  near=[200, 450], log=None)
    return cam0, cam, stats


def test_two_plate_solves_give_the_rate(world, fitted):
    cam0, _, _ = fitted
    true_rate = SIDEREAL_DEG_PER_DAY * world.spf / 86400
    assert abs(cam0.omega / true_rate - 1) < 0.02


def test_camera_fit_recovers_lens_rate_and_pole_of_date(world, fitted):
    _, cam, stats = fitted
    true = world.true_camera(cam.k0)
    assert stats["axis_fitted"]
    assert stats["median_resid_px"] < 0.3
    assert abs(cam.f / world.f - 1) < 2e-3
    assert abs(cam.omega / true.omega - 1) < 1e-4
    assert np.degrees(np.arccos(cam.axis @ true.axis)) < 0.02
    assert abs(cam.seconds_per_frame - world.spf) < 0.01
    for k in (0, 300, 599):                   # every star, all night
        xy, _, idx = world.star_pixels(k, maglim=6.0)
        pred, _ = cam.project(world.star_vecs[idx], k)
        assert np.median(np.hypot(*(pred - xy).T)) < 0.3, k


def test_match_fraction_finds_the_usable_frames(world, fitted, dets):
    _, cam, _ = fitted
    sky = (world.ground == 0).astype(np.uint8)
    frac = calibrate.match_fraction(cam, dets, world.stars, sky_mask=sky)
    assert min(frac.values()) > 0.5
    frac[599] = frac[550] = 0.05                    # dawn washes out
    assert calibrate.usable_range(frac) == (0, 500)
    frac[100] = 0.0                                 # one glitched frame
    assert calibrate.usable_range(frac) == (0, 500)


@pytest.mark.parametrize("ridge,tol_deg", [(0.0, 0.3), (1.5, 3.0)])
def test_horizon_gives_up_and_latitude(ridge, tol_deg):
    w = World(ridge_deg=ridge)
    cam = w.true_camera()
    gray = w.render(cam.k0)[..., 0]
    r = horizon.solve(cam, w.ground, gray, log=None)
    up = np.array(r["up_cam"])
    assert np.degrees(np.arccos(up @ w.up_cam)) < tol_deg
    assert abs(r["lat"] - w.lat) < tol_deg


def test_refraction_is_fitted_from_the_stars():
    """Fitted without refraction, a refracting sky leaves altitude-dependent
    residuals; refit about the (true) up vector, the density comes back and
    the residuals go."""
    w = World(refract=0.8)
    sky = (w.ground == 0).astype(np.uint8)
    dets = {k: calibrate.detect(w.render(k), sky) for k in FRAMES}
    a = platesolve.solve(dets[200], w.stars, w.width, w.height, frame=200,
                         log=None)
    b = platesolve.solve(dets[450], w.stars, w.width, w.height, frame=450,
                         log=None)
    cam0 = calibrate.initial_camera(a, b, w.width, w.height, 300)
    cam, stats = calibrate.refine(cam0, dets, w.stars, sky_mask=sky,
                                  near=[200, 450], log=None)
    cam2, stats2 = calibrate.add_refraction(cam, dets, w.stars, w.up_cam,
                                            sky_mask=sky, log=None)
    assert stats2["median_resid_px"] < stats["median_resid_px"]
    assert stats2["median_resid_px"] < 0.3
    assert abs(cam2.p[11] - 0.8) < 0.05
    xy, _, idx = w.star_pixels(300, maglim=6.0)
    low = (np.degrees(np.arcsin(np.clip(
        w.refracted(w.star_vecs[idx] @ w.gcrs_to_cam(w.t(300)).T) @ w.up_cam,
        -1, 1))) < 8)
    pred, _ = cam2.project(w.star_vecs[idx], 300)
    assert np.median(np.hypot(*(pred - xy)[low].T)) < 0.3


def test_plate_solve_refuses_noise_but_finds_stars_among_it(world, dets):
    """Many noisy detections (a compressed video) make chance matches
    common; a solution must beat that clearly. Pure noise never solves;
    real stars still do with the noise mixed in."""
    rng = np.random.default_rng(9)
    W, H = world.width, world.height
    noise = np.column_stack([rng.uniform(0, W, 4000), rng.uniform(0, H, 4000),
                             rng.uniform(1, 50, 4000)]).astype(np.float32)
    assert platesolve.solve(noise, world.stars, W, H, log=None) is None
    d = dets[300]
    mixed = np.vstack([d[:150], noise[:1500], d[150:]])   # noise mid-list
    sol = platesolve.solve(mixed, world.stars, W, H, frame=300, log=None)
    assert sol is not None and _pairs_ok(world, sol).mean() > 0.95


def test_up_from_vertical_structures():
    """No usable skyline, but towers: their edges meet at the zenith."""
    import cv2
    w = World(alt=35.0, roll=3.0)
    cam = w.true_camera()
    img = np.full((w.height, w.width), 20, np.uint8)
    fg = np.zeros((w.height, w.width), np.uint8)
    for az in np.linspace(w.az - 30, w.az + 30, 9):          # nine towers
        for dz in (-0.4, 0.4):                               # two edges each
            alts = np.radians(np.array([2.0, 30.0]))
            a = np.radians(az + dz)
            enu = np.stack([np.cos(alts) * np.sin(a), np.cos(alts) * np.cos(a),
                            np.sin(alts)], -1)
            xy, z = w.project_cam(enu @ w.C.T)
            if (z > 0).all():
                p = [tuple(int(v) for v in q) for q in xy]
                cv2.line(img, p[0], p[1], 200, 3, cv2.LINE_AA)
                cv2.line(fg, p[0], p[1], 1, 25)
    up, n = horizon.verticals(cam, img, fg)
    assert n >= 8
    assert np.degrees(np.arccos(np.clip(up @ w.up_cam, -1, 1))) < 0.5


def test_pure_sky_has_no_up():
    """No skyline and nothing vertical (or a few columns of speckle): no
    up vector, and the sky rays the free-site solve checks against."""
    w = World()
    cam = w.true_camera()
    blank = np.zeros((w.height, w.width), np.uint8)
    assert horizon.solve(cam, blank, blank, log=None)["up_cam"] is None
    speck = blank.copy()
    speck[w.height // 2:, 100:180] = 1                  # ten columns of it
    r = horizon.solve(cam, speck, blank, log=None)
    assert r["up_cam"] is None and r["source"] == "none"
    rays = horizon.sky_rays(cam, w.ground == 0)
    assert 10 < len(rays) < 81
    assert (rays @ w.up_cam).min() > 0                  # all above the horizon
