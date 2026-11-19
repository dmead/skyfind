import numpy as np

from skyfind.camera import Camera

from .synth import World


def test_model_is_exact_for_a_tripod_on_the_turning_earth():
    """The world renders through skyfield's Earth rotation (precession,
    nutation, true pole of date); the camera model — one orientation, one
    axis, one rate — must reproduce it all night to a small fraction of a
    pixel."""
    w = World()
    cam = w.true_camera()
    for k in (0, 200, 300, 599):
        xy, _, idx = w.star_pixels(k)
        pred, z = cam.project(w.star_vecs[idx], k)
        assert (z > 0).all()
        assert np.abs(pred - xy).max() < 0.05, k


def test_j2000_pole_is_not_good_enough():
    """Rotating about the J2000 pole instead of the pole of date drifts by
    pixels over a night — why the axis is a fitted parameter."""
    w = World()
    cam = w.true_camera()
    j2000 = cam.with_params(np.r_[cam.p[:9], 0.0, 0.0])
    xy, _, idx = w.star_pixels(0)
    pred, _ = j2000.project(w.star_vecs[idx], 0)
    assert np.abs(pred - xy).max() > 1.0


def test_unproject_inverts_project(tmp_path):
    w = World(k1=0.05, k2=-0.01)
    cam = w.true_camera()
    xy, _, idx = w.star_pixels(123)
    v = cam.unproject(xy, 123)
    assert np.degrees(np.arccos(np.clip((v * w.star_vecs[idx]).sum(1), -1, 1))
                      ).max() < 1e-4
    cam.save(tmp_path / "c.json")
    back = Camera.load(tmp_path / "c.json")
    assert np.allclose(back.p, cam.p) and back.k0 == cam.k0


def test_seconds_per_frame_from_the_sidereal_rate():
    w = World(spf=33.87)
    assert abs(w.true_camera().seconds_per_frame - 33.87) < 1e-6


def test_refraction_round_trip_and_exact_against_a_refracting_world():
    w = World(refract=0.8)
    cam = w.true_camera()
    assert cam.refracts
    for k in (0, 300, 599):
        xy, _, idx = w.star_pixels(k)
        pred, _ = cam.project(w.star_vecs[idx], k)
        assert np.abs(pred - xy).max() < 0.05, k
        v = cam.unproject(xy, k)
        err = np.degrees(np.arccos(np.clip((v * w.star_vecs[idx]).sum(1),
                                           -1, 1)))
        assert err.max() < 2e-4
    # near the horizon the atmosphere lifts stars by several arcminutes
    flat = cam.with_params(np.r_[cam.p[:11], 0.0])
    xy, _, idx = w.star_pixels(300)
    shift = np.hypot(*(flat.project(w.star_vecs[idx], 300)[0] - xy).T)
    assert shift.max() > 5.0                                  # px


def test_clock_maps_frames_to_captures():
    """A re-timed clip drops captures: frame k of the video is capture
    clock[k], and the sky turns by captures, not frames."""
    w = World()
    cam = w.true_camera(k0=0)
    clock = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10])        # 1 in 5 dropped
    timed = Camera(cam.p, 0, cam.width, cam.height, clock=clock)
    v = w.star_vecs[:50]
    for k in (3, 4, 8):
        assert np.allclose(timed.project(v, k)[0], cam.project(v, clock[k])[0])
    back = Camera.from_dict(timed.to_dict())
    assert np.array_equal(back.clock, clock)
