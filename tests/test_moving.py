"""A camera on a motion-control rig: it pans and tilts as the clip runs.
The synthetic world's tripod camera, turned frame by frame about the
vertical (a pan) with a slow tilt wobble on top."""
import numpy as np
from scipy.spatial.transform import Rotation

from skyfind import calibrate, motion
from skyfind.camera import Camera, radec_to_vec
from skyfind.platesolve import PlateSolution
from skyfind.solve import Problem

from .synth import World

PAN = 0.07          # deg per frame


def _rig(w: World, k1=None) -> Camera:
    if k1 is not None:
        w = World(k1=k1, k2=0.0)
    cam = w.true_camera()
    k = np.arange(w.n_frames, dtype=float) - cam.k0
    pan = np.radians(PAN) * k
    tilt = np.radians(0.8) * np.sin(k / 90.0)
    G = Rotation.from_rotvec(np.outer(pan, w.up_cam)) * \
        Rotation.from_rotvec(np.outer(tilt, [1.0, 0, 0]))
    return Camera(cam.p, cam.k0, cam.width, cam.height, turn=G.as_rotvec())


def _sky(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return radec_to_vec(rng.uniform(0, 360, n),
                        np.degrees(np.arcsin(rng.uniform(-1, 1, n))))


def _Q(cam, k):
    """GCRS -> camera rotation at frame k (the stars' view, no lens)."""
    return np.column_stack([cam.rays(e[None], k)[0] for e in np.eye(3)])


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def test_a_turning_camera_round_trips_and_saves():
    w = World()
    cam = _rig(w)
    v = _sky()
    for k in (0, 123, 299.5, 599):
        xy, z = cam.project(v, k)
        ok = cam.in_frame(xy, z)
        assert ok.sum() > 20
        assert _angle(cam.unproject(xy[ok], k), v[ok]).max() < 1e-5
    back = Camera.from_dict(cam.to_dict())
    assert back.moving and np.allclose(back.turn, cam.turn, atol=1e-9)
    xy, z = cam.project(v, 77)
    ok = cam.in_frame(xy, z)
    assert np.abs(back.project(v[ok], 77)[0] - xy[ok]).max() < 1e-4


def test_another_interval_is_the_same_view_of_the_sky():
    """The stars fix each frame's view; a longer interval only moves the
    turn between the Earth and the camera."""
    w = World()
    cam = _rig(w)
    other = cam.with_interval(w.spf * 1.03)
    assert abs(other.seconds_per_frame - w.spf * 1.03) < 1e-9
    v = _sky()
    for k in (0, 250, 599):
        assert _angle(other.rays(v, k), cam.rays(v, k)).max() < 1e-5


def test_the_stars_tell_a_rig_from_a_tripod():
    w = World()
    tripod, rig = w.true_camera(), _rig(w)
    ks = range(0, w.n_frames, 20)

    def sols(cam):
        return {k: PlateSolution(k, np.zeros((0, 2)), np.zeros((0, 3)),
                                 np.zeros(0), [], w.f, 50, _Q(cam, k))
                for k in ks}
    fixed = np.array(list(calibrate.fixed_misfit(
        sols(tripod), w.width, w.height).values()))
    turning = np.array(list(calibrate.fixed_misfit(
        sols(rig), w.width, w.height).values()))
    # the tripod turns about the pole of date, not J2000's (0.14 deg off)
    assert np.percentile(fixed, 90) < 0.1
    assert np.percentile(turning, 90) > 1.0


def test_the_land_gives_the_frame_interval():
    """How the land turned between frames (here exact) against the stars'
    orientations: only the true interval keeps the land still."""
    w = World()
    rig = _rig(w)
    ks = list(range(0, w.n_frames, 25))
    Q = {k: _Q(rig, k) for k in ks}
    pairs = {(i, j): rig.G(j)[0] @ rig.G(i)[0].T
             for a, i in enumerate(ks) for j in ks[a + 2:a + 5]}
    k0 = rig.k0 if rig.k0 in Q else ks[len(ks) // 2]
    s, sign, err = motion.interval(Q, pairs, k0, lambda k: k)
    # to a quarter of a percent: the search turns the sky about J2000's
    # pole, the world about the pole of date (0.14 deg away)
    assert abs(s - w.spf) < 0.0025 * w.spf, s
    assert sign > 0


def test_the_sky_track_follows_the_stars():
    """remove-satellites' track made from the camera: each step carries a
    star from one frame to the next."""
    w = World(k1=0.0, k2=0.0)
    rig = _rig(w, k1=0.0)
    tr = motion.sky_track(rig, 40)
    v = _sky(2000)
    for k in (0, 17, 38):
        a, za = rig.project(v, k)
        b, zb = rig.project(v, k + 1)
        ok = rig.in_frame(a, za) & rig.in_frame(b, zb)
        h = tr.steps[k]
        p = np.c_[a[ok], np.ones(ok.sum())] @ h.T
        p = p[:, :2] / p[:, 2:]
        assert np.abs(p - b[ok]).max() < 0.05, k


def test_exposure_turn_is_what_the_rig_records():
    """A point recorded `off` s from mid-exposure, where the frame's model
    puts it: the sky's turn and the rig's own. The rig's is averaged over
    the clip (one turn for the GPU's kernels too), which misses its wobble
    and the vertical's swing through the sky by ~0.01 deg (0.1 px) at the
    ends of an exposure; leaving it out misses by several times that."""
    w = World()
    rig = _rig(w)
    P = Problem(None, [], np.zeros(0), None, 0.0, w.spf, 0.0, 0.0,
                camera=rig, n_frames=w.n_frames)
    ax, rate = P.exposure_turn()
    v = _sky()
    from skyfind.camera import rotate
    sky_ax, sky_rate = rig.axis, -np.radians(rig.omega) / w.spf
    for k in (100, 300):
        for off in (-12.0, 12.0):
            xy, z = rig.project(v, k + off / w.spf)
            ok = rig.in_frame(xy, z)
            seen = rig.unproject(xy[ok], k)
            got = rotate(v[ok], ax, rate * off)
            assert _angle(got, seen).max() < 0.012, (k, off)
            sky_only = rotate(v[ok], sky_ax, sky_rate * off)
            assert _angle(sky_only, seen).max() > 0.02, (k, off)
    # a tripod's is the sky's turn, exactly as before
    tri = w.true_camera()
    P = Problem(None, [], np.zeros(0), None, 0.0, w.spf, 0.0, 0.0,
                camera=tri, n_frames=w.n_frames)
    ax, rate = P.exposure_turn()
    assert np.array_equal(ax, tri.axis)
    assert rate == -np.radians(tri.omega) / w.spf


def test_a_rigs_view_is_exactly_its_camera_model():
    """The synthetic world on a rig (pan and tilt, through skyfield's own
    frames): the camera model with the rig's turn puts every star where
    the world renders it, all clip long."""
    w = World(width=1920, height=1080, f=760.0, pan=0.05, tilt=1.0)
    cam = w.true_camera()
    assert cam.moving
    for k in (0, 150, 300, 599):
        xy, _, idx = w.star_pixels(k)
        pred, z = cam.project(w.star_vecs[idx], k)
        assert len(idx) > 50 and (z > 0).all()
        assert np.abs(pred - xy).max() < 0.05, k
    # and the land moves: the skyline isn't where it was
    assert (w.ground_at(0) != w.ground_at(599)).mean() > 0.01
