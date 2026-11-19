"""Orbit table and site model against the synthetic world (skyfield's own
observer and SGP4 positions)."""

import numpy as np
import pytest

from skyfind.orbits import Orbits
from skyfind.site import Site

from .synth import TS, World


@pytest.fixture(scope="module")
def world():
    return World()


def test_orbit_table_matches_direct_sgp4(world):
    sats = world._satrecs
    t0 = world.t(0).tt
    orb = Orbits.propagate(sats, t0 - 0.01, t0 + 0.1, log=None)
    ks = np.array([13.3, 101.7, 250.05])
    t = world.t(ks)
    truth = world.sat_gcrs(t)                                # (s, 3, 3)
    for i, tt in enumerate(t.tt):
        got = orb.at(tt)
        d = np.linalg.norm(got - truth[:, i], axis=1)
        low = np.linalg.norm(truth[:, i], axis=1) < 8000      # LEO
        ok = ~np.isnan(d)
        assert np.nanmax(d[ok & low]) < 0.1                   # km
        assert np.nanmax(d[ok]) < 1.0
    j = np.array([0, 5, 700])
    many = orb.at_many(np.full(3, t.tt[1]), j)
    assert np.allclose(many, orb.at(t.tt[1])[j], atol=1e-6)


def test_orbit_table_save_load_mmap(world, tmp_path):
    t0 = world.t(0).tt
    orb = Orbits.propagate(world._satrecs[:50], t0, t0 + 0.01, log=None)
    orb.save(tmp_path)
    back = Orbits.load(tmp_path)
    assert np.allclose(back.at(t0 + 0.004), orb.at(t0 + 0.004))


def test_site_from_stars_and_up_vector_is_the_observer(world):
    """True camera + true up vector + true latitude/height must put the site
    where skyfield puts the observer, at every frame, with dlst = 0."""
    cam = world.true_camera()
    site = Site(cam, world.up_cam, world.t(300).tt, world.n_frames)
    assert abs(site.horizon_lat - world.lat) < 0.01
    for k in (0, 300, 599):
        obs = world.observer.at(world.t(k)).position.km
        got = site.gcrs(k, world.lat, 0.0, world.height_km)
        assert np.linalg.norm(got - obs) < 0.05, k             # km
    # vectorised over frames
    ks = np.array([0, 300, 599])
    obs = world.observer.at(world.t(ks)).position.km.T
    assert np.abs(site.gcrs(ks, world.lat, 0.0, world.height_km) - obs
                  ).max() < 0.05
    # velocity: skyfield's observer velocity
    v = world.observer.at(world.t(300)).velocity.km_per_s
    assert np.linalg.norm(site.velocity(site.gcrs(300, world.lat, 0, 1.9))
                          - v) < 1e-3


def test_longitude_comes_back_from_inertial_position_and_time(world):
    cam = world.true_camera()
    site = Site(cam, world.up_cam, world.t(300).tt, world.n_frames)
    lat, lon = site.geodetic(300, world.lat, 0.0, world.height_km,
                             world.t(300).tt)
    assert abs(lat - world.lat) < 1e-4 and abs(lon - world.lon) < 1e-4
    # an hour later on the clock -> 15.04 deg further west
    lat, lon2 = site.geodetic(300, world.lat, 0.0, world.height_km,
                              world.t(300).tt + 1 / 24)
    assert abs((lon - lon2) - 15.041) < 0.01
