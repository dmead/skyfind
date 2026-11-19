"""A camera on a mount that follows the stars: the fit's camera
re-expressed as a turning one, and the synthetic sky shot that way."""

import numpy as np

from skyfind.camera import SIDEREAL_DEG_PER_DAY

from .synth import World


def test_tracking_keeps_every_pixel_where_the_stars_put_it():
    w = World(n_frames=120, f=1100.0, width=1920, height=1080)
    fixed = w.true_camera()
    # what a fit to a tracked clip finds: the sky all but still, about an
    # axis that means nothing
    p = fixed.p.copy()
    p[3], p[9], p[10] = -0.0006, 0.31, -0.2
    still = fixed.with_params(p)
    pole = w.t(60).M[2]
    cam = still.tracking(7.5, w.n_frames, pole)
    assert cam.moving and abs(cam.seconds_per_frame - 7.5) < 1e-9
    xy = np.array([[50.0, 40], [960, 540], [1900, 1000]])
    for k in (0, 37, 119):
        a, b = still.unproject(xy, k), cam.unproject(xy, k)
        assert np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).max() \
            < 1e-4
    # another interval moves only the turn about the pole: the map stays
    other = cam.with_interval(30.0)
    assert abs(other.omega - 30.0 * SIDEREAL_DEG_PER_DAY / 86400) < 1e-12
    a, b = still.unproject(xy, 100), other.unproject(xy, 100)
    assert np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).max() < 1e-4
    # and the Earth-fixed pose now turns with the Earth: the zenith a
    # site's up points at moves at the sidereal rate
    z = other.cam_to_gcrs(np.tile([0.0, -1.0, 0.0], (2, 1)), [0, 119])
    turned = np.degrees(np.arccos(np.clip(z[0] @ z[1], -1, 1)))
    assert 0.05 < turned < 119 * other.omega + 1e-6


def test_the_synthetic_mount_holds_the_stars_still():
    w = World(n_frames=120, f=1100.0, width=1920, height=1080, az=110.0,
              alt=50.0, tracked=True)
    a, _, _ = w.star_pixels(10)
    b, _, _ = w.star_pixels(110)
    n = min(len(a), len(b))
    assert n > 50
    # the same stars at the same pixels 50 minutes apart
    d = np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)).min(1)
    assert np.median(d) < 0.05
    # and the camera `true_camera` builds for it puts them there
    cam = w.true_camera()
    xy, z = cam.project(w.star_vecs[:200], 110)
    ok = cam.in_frame(xy, z)
    assert ok.sum() > 20
    d = np.hypot(*(xy[ok][:, None, :] - b[None, :, :]).transpose(2, 0, 1)
                 ).min(1)
    assert np.median(d) < 0.5
