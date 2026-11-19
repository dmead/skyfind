"""The whole pipeline on a rendered clip: skyfind tracks the sky, masks the
land, takes the trails out and extracts them, plate-solves,
fits the camera, reads the skyline and solves blind (+-12 h) against the
synthetic constellation's TLE text. Slow: a few minutes."""

import collections
from datetime import timedelta

import pytest

from skyfind import pipeline, report

from .synth import World

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    from remove_satellites.core.video import FrameWriter
    w = World(width=1920, height=1080, f=760.0)
    d = tmp_path_factory.mktemp("e2e")
    clip = d / "night.mp4"
    by = collections.defaultdict(list)
    for s in w.segments():
        by[s["k"]].append(s)
    with FrameWriter(clip, w.width, w.height, 24, crf=12) as fw:
        for k in range(w.n_frames):
            fw.write(w.render(k, by.get(k, ())))
    tle = d / "night.3le"
    tle.write_text(w.tle_text, encoding="ascii")
    return w, clip, tle, sum(len(v) for v in by.values())


def test_skyfind_end_to_end(rendered):
    w, clip, tle, n_true = rendered
    wd = pipeline.Workdir(clip)
    pipeline.run_trails(wd, workers=8, log=print)
    around = (w.t0 + timedelta(hours=3.7)).strftime("%Y-%m-%d %H:%M")
    sol = pipeline.run_solve(wd, tle=tle, around=around, max_tracks=10,
                             workers=8, stars=w.stars, log=print)
    # some land was found (the synthetic terrain is faint: the foreground
    # detector marks only part of it, which is enough for the skyline)
    assert 0.01 < wd.fgmask().mean() < w.ground.mean() + 0.05
    cam = wd.camera()
    assert abs(cam.seconds_per_frame - w.spf) < 0.02
    assert abs(sol["lat"] - w.lat) < 0.02
    assert abs(sol["lon"] - w.lon) < 0.02
    assert abs(sol["frame0_tt"] - w.t(0).tt) * 86400 < 1.0
    assert abs(sol["E"] - w.exposure) < 0.5
    assert len(sol["ids"]) > 0.5 * n_true
    paths = report.write(wd, map=False, log=print)
    assert any(p.name == "night_satellites.jpg" for p in paths)
    # nothing written next to the clip: all of it is in the work folder
    assert sorted(p.name for p in clip.parent.iterdir()) == \
        ["night.3le", "night.mp4", "night.skyfind"]


def test_a_recorded_site_names_the_trails(rendered, tmp_path):
    """The phone's own capture: where and when are recorded (the clock
    1.3 s out here), so nothing is searched; up comes from the site, the
    clock is set by the trails, and the same trails are named."""
    import shutil
    w, clip, tle, n_true = rendered
    src = pipeline.Workdir(clip)
    if not src.has("segments.json"):
        pipeline.run_trails(src, workers=8, log=print)
    work = tmp_path / "recorded.skyfind"
    shutil.copytree(src.dir, work, ignore=shutil.ignore_patterns(
        "camera.json", "horizon.json", "solution.json", "events.jsonl",
        "date.json", "trackfit.json", "clusters.json"))
    wd = pipeline.Workdir(clip, work)
    known = dict(lat=w.lat, lon=w.lon, height_m=w.height_km * 1000,
                 frame0_unix=w.t0.timestamp() + 1.3, exposure_s=w.exposure,
                 accuracy_m=10.0)
    sol = pipeline.run_known(wd, known, tle=tle, workers=8, stars=w.stars,
                             log=print)
    assert abs(sol["frame0_tt"] - w.t(0).tt) * 86400 < 0.3
    assert sol["lat"] == w.lat and sol["lon"] == w.lon
    assert len(sol["ids"]) > 0.5 * n_true
    # up is the site's zenith through the fitted camera
    assert wd.read("horizon.json")["source"] == "site"
    assert wd.camera().refracts
    paths = report.write(wd, map=False, log=print)
    assert any(p.name == "night_satellites.jpg" for p in paths)


def test_scan_finds_the_night(rendered):
    """Only 'some night that week' is known: each night is searched on its
    own and the right one stands out."""
    w, clip, tle, _ = rendered
    wd = pipeline.Workdir(clip)
    if not wd.has("segments.json"):
        pipeline.run_trails(wd, workers=8, log=print)
    night = w.t0.date()
    first = (night - timedelta(days=1)).isoformat()
    last = (night + timedelta(days=1)).isoformat()
    s = pipeline.run_scan(wd, first, last, tle=tle, max_tracks=8, workers=8,
                          stars=w.stars, then_solve=False, log=print)
    assert s["best"]["night"] == night.isoformat()
    assert s["best"]["best"] >= 4 and s["best"]["best"] >= s["runner_up"] + 2
    assert abs(s["best"]["lat"] - w.lat) < 0.5


@pytest.fixture(scope="module")
def rendered_rig(tmp_path_factory):
    """The same world shot from a motion-control rig: it pans 30 degrees
    about the vertical over the clip, with a slow tilt wobble."""
    from remove_satellites.core.video import FrameWriter
    w = World(width=1920, height=1080, f=760.0, pan=0.05, tilt=1.0)
    d = tmp_path_factory.mktemp("rig")
    clip = d / "rig.mp4"
    by = collections.defaultdict(list)
    for s in w.segments():
        by[s["k"]].append(s)
    with FrameWriter(clip, w.width, w.height, 24, crf=12) as fw:
        for k in range(w.n_frames):
            fw.write(w.render(k, by.get(k, ())))
    tle = d / "rig.3le"
    tle.write_text(w.tle_text, encoding="ascii")
    return w, clip, tle, sum(len(v) for v in by.values())


def test_a_rig_end_to_end(rendered_rig):
    """A turning camera: found turning, followed frame by frame, its
    interval from the land and then the satellites, the trails taken out
    again along the sky's true motion, up from the ridge, and the site."""
    w, clip, tle, n_true = rendered_rig
    wd = pipeline.Workdir(clip)
    pipeline.run_trails(wd, workers=8, log=print)
    around = (w.t0 + timedelta(hours=3.7)).strftime("%Y-%m-%d %H:%M")
    sol = pipeline.run_solve(wd, tle=tle, around=around, workers=8,
                             stars=w.stars, log=print)
    cam = wd.camera()
    assert cam.moving
    # the interval refined on the joint fit's anchors: to a fiftieth of a
    # percent (the objects agreeing hold over half a percent either way)
    assert abs(cam.seconds_per_frame - w.spf) < 0.0002 * w.spf
    assert abs(sol["lat"] - w.lat) < 0.005
    assert abs(sol["lon"] - w.lon) < 0.005
    assert abs(sol["frame0_tt"] - w.t(0).tt) * 86400 < 3.0
    assert abs(sol["E"] - w.exposure) < 0.5
    assert len(sol["ids"]) > 0.5 * n_true


@pytest.fixture(scope="module")
def rendered_tracked(tmp_path_factory):
    """The same sky from a mount that follows the stars: they stand still
    in the frame, and nothing but sky is in it."""
    from remove_satellites.core.video import FrameWriter
    # towards dawn, looking east: the satellites up there are sunlit
    w = World(width=1920, height=1080, f=1100.0, az=110.0, alt=52.0,
              n_frames=240, spf=20.0, exposure=16.0, tracked=True,
              t0=World().t0 + timedelta(hours=3.3))
    assert not w.ground_at(0).any() and not w.ground_at(239).any()
    d = tmp_path_factory.mktemp("tracked")
    clip = d / "tracked.mp4"
    by = collections.defaultdict(list)
    for s in w.segments():
        by[s["k"]].append(s)
    with FrameWriter(clip, w.width, w.height, 24, crf=12) as fw:
        for k in range(w.n_frames):
            fw.write(w.render(k, by.get(k, ())))
    tle = d / "tracked.3le"
    tle.write_text(w.tle_text, encoding="ascii")
    return w, clip, tle, sum(len(v) for v in by.values())


def test_a_tracking_mount_end_to_end(rendered_tracked):
    """Stars that stand still: every frame is a capture all the same, the
    camera is taken as turning with the sky, and the interval, which the
    stars can't give, is found from the satellites over every interval
    their rates allow; then the site and the clock as ever."""
    w, clip, tle, n_true = rendered_tracked
    wd = pipeline.Workdir(clip)
    pipeline.run_trails(wd, workers=8, log=print)
    assert wd.info["clock"] is None
    around = (w.t0 - timedelta(hours=2.2)).strftime("%Y-%m-%d %H:%M")
    sol = pipeline.run_solve(wd, tle=tle, around=around, workers=8,
                             stars=w.stars, log=print)
    cam = wd.camera()
    assert cam.moving and cam.meta["tracked"]["from_satellites"]
    assert abs(cam.seconds_per_frame - w.spf) < 0.002 * w.spf
    assert abs(sol["lat"] - w.lat) < 0.05
    assert abs(sol["lon"] - w.lon) < 0.05
    assert abs(sol["frame0_tt"] - w.t(0).tt) * 86400 < 5.0
    assert len(sol["ids"]) > 0.4 * n_true
