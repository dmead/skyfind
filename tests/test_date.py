"""Dating a shoot from planets, the Sun and precession (needs DE421 in the
data directory; skipped without it rather than downloading 17 MB)."""

import numpy as np
import pytest

from skyfind import calibrate, data, date
from skyfind.camera import vec_to_radec

from .synth import World

pytestmark = pytest.mark.skipif(
    not (data.data_dir() / "de421.bsp").exists(),
    reason="DE421 not in the data directory")


@pytest.fixture(scope="module")
def world():
    return World()


def _truth(w, names, ks):
    pv = [date.planet_vectors(np.array([w.t(k).tt]), (n,))[n][0]
          for n, k in zip(names, ks)]
    return [dict(zip(("ra", "dec"), map(float, vec_to_radec(v))), k_mid=k)
            for v, k in zip(pv, ks)]


def test_two_planets_date_the_night(world):
    objs = _truth(world, ["mars", "jupiter barycenter"], [100, 400])
    cam = world.true_camera()
    zen = cam.cam_to_gcrs(np.tile(world.up_cam, (world.n_frames, 1)),
                          np.arange(world.n_frames))
    c = date.search(objs, world.spf, zen, usable=(0, world.n_frames - 1),
                    log=None)
    assert c and c[0]["matched"] == 2
    assert set(c[0]["planets"].values()) == {"mars", "jupiter barycenter"}
    assert abs(c[0]["tt"] - world.t(0).tt) * 24 < 6          # hours


def test_an_unknown_object_is_left_unmatched(world):
    objs = _truth(world, ["mars", "jupiter barycenter"], [100, 400])
    objs.append(dict(ra=12.3, dec=-45.6, k_mid=300))      # not a planet
    c = date.search(objs, world.spf, log=None)
    assert c[0]["matched"] == 2 and 2 not in c[0]["planets"]


def test_the_sun_rules_out_daytime_seasons(world):
    """Jupiter alone fits a date every ~12 years and many days around each;
    requiring night at the frames' sidereal times removes most of them."""
    objs = _truth(world, ["jupiter barycenter"], [300])
    cam = world.true_camera()
    zen = cam.cam_to_gcrs(np.tile(world.up_cam, (world.n_frames, 1)),
                          np.arange(world.n_frames))
    loose = date.search(objs, world.spf, log=None)
    dark = date.search(objs, world.spf, zen, usable=(0, world.n_frames - 1),
                       log=None)
    span = lambda cs: sum(c["span_h"] for c in cs)          # noqa: E731
    assert span(dark) < span(loose)
    assert any(abs(c["tt"] - world.t(0).tt) < 2 for c in dark)


def test_precession_gives_the_year(world):
    year, _ = date.pole_year(world.true_camera().axis)
    assert abs(year - 2025.1) <= 0.5


def test_planet_candidates_are_persistent_off_catalogue_points(world):
    cam = world.true_camera()
    mars = date.planet_vectors(np.array([world.t(300).tt]), ("mars",))["mars"]
    # a bright point where no catalogue star is, in view all night
    v = cam.unproject(np.array([[1900.0, 700.0]]), 300)[0]
    sky = (world.ground == 0).astype(np.uint8)
    dets = {k: calibrate.detect(world.render(k, extra=[(v, -0.5)]), sky)
            for k in (100, 200, 300, 400)}
    got = date.planet_candidates(cam, dets, world.stars, log=None)
    assert len(got) >= 1
    ra, dec = vec_to_radec(v)
    best = min(got, key=lambda o: abs(o["ra"] - ra) + abs(o["dec"] - dec))
    assert abs(best["ra"] - ra) < 0.05 and abs(best["dec"] - dec) < 0.05
    assert len(best["frames"]) == 4
    assert mars.shape == (1, 3)


def test_nothing_far_from_the_ecliptic_is_a_planet():
    """Near the pole nothing can be a planet: say so at once instead of
    searching sixty years."""
    import time
    objs = [dict(ra=224.0, dec=74.9, frames=[1, 2, 3], k_mid=2, flux=1.0),
            dict(ra=84.8, dec=37.9, frames=[4, 5, 6], k_mid=5, flux=1.0)]
    logs = []
    t0 = time.time()
    assert date.search(objs, 30.0, log=logs.append) == []
    assert time.time() - t0 < 1.0
    assert any("can be a planet" in m for m in logs)


def test_assignment_by_groups_matches_trying_everything():
    """Objects near a planet only matter where they are; the grouped search
    gives exactly what trying every assignment gives."""
    rng = np.random.default_rng(11)
    seps = rng.uniform(0, 4, (5, 6, 3000))
    seps[rng.random(seps.shape) < 0.97] += 20          # mostly far from all
    fast = date._assign(seps, 1.0)
    slow = date._assign_all(seps, 1.0)
    assert np.array_equal(fast[0], slow[0])
    assert np.allclose(fast[1], slow[1])
    assert np.array_equal(fast[2], slow[2])


def test_eight_objects_are_quick():
    import time
    seps = np.full((8, 6, 40000), 30.0)
    seps[0, 3, 100:200] = 0.2                           # one near Jupiter
    t = time.time()
    m, _, a = date._assign(seps, 1.0)
    assert time.time() - t < 5
    assert m[150] == 1 and a[0, 150] == 3 and m[0] == 0
