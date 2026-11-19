"""The replay: candidate sites close in, trails get names, GIFs come out."""

import os
import shutil

import numpy as np
import pytest

from skyfind import replay


def _story(n_tracks=6):
    """Every track's object fits the true site (lat 46.55, lon 7.98, T 0)
    and, like any LEO object, once per orbit elsewhere around the globe."""
    rng = np.random.default_rng(3)
    tracks = []
    for i in range(n_tracks):
        period = 5400 + 300 * i                      # a different orbit each
        fits = []
        for m in range(-6, 7):
            fits.append(dict(lat=46.55 + rng.normal(0, 0.05),
                             lon=7.98 - m * period / 239.34
                             + rng.normal(0, 0.05),
                             T=m * period + rng.normal(0, 3), obj=f"OBJ {i}",
                             norad=1000 + i, med_err=0.05))
        tracks.append(dict(frames=[10 * i, 10 * i + 1, 10 * i + 2],
                           segs=[[100 + 30 * i, 50, 180 + 30 * i, 90]] * 3,
                           fits=fits))
    ids = [dict(seg=j, k=10 * j, name=f"OBJ {j}", norad=1000 + j,
                err_deg=0.05, dt=0.0, utc="01:00:00", x1=100.0 + 30 * j,
                y1=50.0, x2=180.0 + 30 * j, y2=90.0) for j in range(n_tracks)]
    objs = [dict(name=f"OBJ {j}", norad=1000 + j, frames=[10 * j],
                 segs=[j], utc="01:00:00", err=0.05, conf="confirmed",
                 n=j + 1) for j in range(n_tracks)]
    sol = dict(lat=46.55, lon=7.98, sig_lat=0.01, sig_lon=0.01,
               frame0_utc="2025-02-14 00:30:00", E=25.0, spf=30.0,
               anchors=[dict(name=f"OBJ {j}", frames=tracks[j]["frames"])
                        for j in range(n_tracks)])
    bg = np.full((360, 640, 3), 30, np.uint8)
    story = replay.Story(bg, tracks, sol, ids, objs, 47.0, "test")
    # the footage: 60 frames at 30 a second; the segments the solve used are
    # the tracks' and one that nothing named, beside track 2's in frame 20
    story.n_frames, story.fps = 60, 30.0
    story.rows = [dict(k=k, x1=g[0], y1=g[1], x2=g[2], y2=g[3])
                  for t in tracks for k, g in zip(t["frames"], t["segs"])]
    if n_tracks > 2:
        story.rows.append(dict(k=20, x1=400.0, y1=200.0, x2=480.0, y2=260.0))
    return story


def test_candidate_sites_close_in_on_the_answer():
    sc = replay.scenes(_story())
    counts = [len(s.sites) for s in sc if s.phase == "tracks"
              and s.sites is not None]
    assert counts[0] >= 10                       # one object: all its orbits
    assert counts[-1] <= 8                       # six agree: one place
    last = [s for s in sc if s.phase == "tracks"][-1]
    assert np.all(np.abs(last.sites[:, 0] - 46.55) < 0.5)
    assert np.all(np.abs(last.sites[:, 1] - 7.98) < 0.5)
    assert last.support == 6
    # the map zooms in as they do
    assert sc[-1].view[2] < 1 < sc[0].view[2]


def test_trails_are_named_and_the_list_grows():
    story = _story()
    sc = replay.scenes(story)
    shown = [s.shown for s in sc if s.phase in ("ids", "final")]
    assert shown == sorted(shown) and shown[-1] == len(story.ids)
    r = replay.Renderer(story, width=800)
    img = r.render(sc[-1])
    assert img.shape == (r.H, 800, 3)
    assert img[:r.fh, r.fw:].std() > 1              # the list has text


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_gif_is_written(tmp_path):
    story = _story(3)
    r = replay.Renderer(story, width=480, map_h=160)
    frames = [r.render(s) for s in replay.scenes(story)[:30]]
    out = tmp_path / "solve.gif"
    replay.write_gif(frames, out, fps=10)
    assert out.read_bytes()[:6] in (b"GIF89a", b"GIF87a")
    assert os.path.getsize(out) > 1000


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_mp4_is_written(tmp_path):
    story = _story(3)
    r = replay.Renderer(story, width=480, map_h=160)
    frames = [r.render(s) for s in replay.scenes(story)[:20]]
    out = tmp_path / "solve.mp4"
    replay.write_mp4(frames, out, fps=10)
    from remove_satellites.core import video
    info = video.probe(out)
    assert info.width == 480 and abs(info.fps - 30) < 0.1
    assert info.frame_count >= 55                 # 20 frames at 10 -> 30 fps


def test_binned_support_matches_the_dense_count():
    rng = np.random.default_rng(7)
    n = 600
    fits = [dict(lat=float(rng.uniform(30, 32)), lon=float(rng.uniform(-181, -178)),
                 T=float(rng.uniform(-200, 200)), track=int(rng.integers(0, 9)))
            for _ in range(n)]
    lat = np.array([f["lat"] for f in fits])
    lon = np.array([f["lon"] for f in fits])
    T = np.array([f["T"] for f in fits])
    tr = np.array([f["track"] for f in fits])
    dl = np.abs((lon[:, None] - lon[None] + 180) % 360 - 180)
    near = (np.abs(T[:, None] - T[None]) <= 40) & \
        (np.abs(lat[:, None] - lat[None]) <= 0.6) & (dl <= 0.8)
    dense = np.array([len(set(tr[near[i]])) for i in range(n)])
    support, top = replay.candidates(fits)
    assert np.array_equal(support, dense) and top == dense.max()


def test_dating_objects_are_placed_on_the_frame(tmp_path):
    """What dated the night gets placed by the camera model: in view, set
    behind the land, or out of the frame, including a direction so far off
    the axis that the lens polynomial folds it back into the frame."""
    import cv2

    from skyfind import report
    from skyfind.camera import vec_to_radec
    from skyfind.pipeline import Workdir

    from .synth import World
    w = World(k1=0.021, k2=-0.011)
    cam = w.true_camera()
    k, W, H = 300, cam.width, cam.height
    wd = Workdir(tmp_path / "clip.mp4")
    cam.save(wd.path("camera.json"))
    wd.write("clip.json", dict(clip=str(wd.clip), cleaned="cleaned.mp4",
                               n_frames=w.n_frames, width=W, height=H))
    fg = np.zeros((H, W), np.uint8)
    fg[int(0.8 * H):] = 255                                  # the land
    cv2.imwrite(str(wd.path("fgmask.png")), fg)
    sky = cam.unproject(np.array([[W / 2, H / 3], [W / 2, 0.9 * H]]), k)
    r = np.array([[np.tan(np.radians(72)), 0.0, 1.0]])      # far off axis
    far = cam.cam_to_gcrs(r / np.linalg.norm(r), k)
    xy, z = cam.project(far, k)
    assert cam.in_frame(xy, z).all()                         # the fold
    objs = [dict(zip(("ra", "dec"), map(float, vec_to_radec(v))))
            for v in (*sky, far[0])]
    wd.write("date.json", dict(source="comets", objects=objs, candidates=[
        dict(comets=["C/2025 A6 (Lemmon)", "C/2025 R2 (SWAN)", "12P/Far"],
             objects=[0, 1, 2])]))
    wd.write("solution.json", dict(frame0_utc="2025-10-23 19:47:59",
                                   spf=13.0))
    got = {o["name"]: o for o in report.dating(wd, k)}
    assert got["Lemmon"]["state"] == "in view"
    assert abs(got["Lemmon"]["x"] - W / 2) < 1
    assert got["SWAN"]["state"] == "set"
    assert all(y < 0.8 * H for _, y in got["SWAN"]["path"])
    assert got["12P/Far"]["state"] == "out of frame"         # no nickname
    for o in got.values():
        assert o["kind"] == "comet"
        if o["mark"] and o["state"] != "in view":
            assert o["mark"][2].endswith("UTC")
    # drawn: rings on the frame, names in the list
    story = _story(3)
    story.dating = list(got.values())
    img = replay.Renderer(story, width=800).render(replay.scenes(story)[-1])
    assert (np.abs(img.astype(int) - report.COMET).sum(2) < 60).sum() > 50


def test_the_story_plays_forward_then_recounts():
    """The footage once, forward, at its own speed; each fitted track joins
    the map as its last frame goes by (sites close in), solved when the
    last has; then every trail drawn in and named over the still, a little
    slower than the still replay does it."""
    story = _story()
    sc = replay.scenes(story, play=True)
    play = [s for s in sc if s.phase == "play"]
    assert [s.k for s in play] == list(range(60))
    assert {s.ticks for s in play} == {1.0}             # 30 fps as filmed
    ends = sorted(max(t["frames"]) for t in story.tracks)
    for s in play:
        assert s.done == sum(e <= s.k for e in ends)
        assert s.solved == (s.k >= ends[-1])
    counts = [len(s.sites) for s in play if s.sites is not None]
    assert counts[0] >= 10 and counts[-1] <= 8          # one object, then six
    rest = sc[len(play):]
    assert {s.phase for s in rest} == {"ids", "final"}
    assert all(s.k is None for s in rest)               # the still
    shown = [s.shown for s in rest]
    assert shown == sorted(shown) and shown[-1] == len(story.ids)
    assert sum(s.phase == "ids" for s in rest) == 90    # 1.5x the still's 60
    assert rest[-1].view == (46.55, 7.98, 0.35)
    story.fps = 15.0
    assert {s.ticks for s in replay.scenes(story, play=True)
            if s.phase == "play"} == {2.0}
    assert all(s.k is None for s in replay.scenes(story))   # still as before


def test_trails_together_are_ringed_together():
    story = _story(3)
    r = replay.Renderer(story, width=800)
    grey = np.full((r.fh, r.fw, 3), 90, np.uint8)
    r.source = lambda k: grey
    play = {s.k: s for s in replay.scenes(story, play=True)
            if s.phase == "play"}

    def tint(img, row):                   # just outside its ring; its middle
        (x1, y1), (x2, y2) = r._pts(row["x1"], row["y1"], row["x2"],
                                    row["y2"])
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        nx, ny = -(y2 - y1), x2 - x1
        n = np.hypot(nx, ny)
        x = int(round(mx + nx / n * (replay.RING + 1)))
        y = int(round(my + ny / n * (replay.RING + 1)))
        return (np.abs(img[y - 1:y + 2, x - 1:x + 2] - 90).sum(2).max(),
                np.abs(img[my, mx] - 90).sum())

    both = [row for row in story.rows if row["k"] == 20]
    assert len(both) == 2
    img = r.render(play[20])[:r.fh, :r.fw].astype(int)
    for row in both:
        ring, middle = tint(img, row)
        assert ring > 30 and middle == 0                # ringed, trail as is
    img = r.render(play[19])[:r.fh, :r.fw].astype(int)
    assert tint(img, both[1])[0] < 6                    # not yet
    img = r.render(play[21])[:r.fh, :r.fw].astype(int)
    now = tint(r.render(play[20])[:r.fh, :r.fw].astype(int), both[1])[0]
    assert 6 < tint(img, both[1])[0] < now              # fading


def test_what_dated_the_night_is_ringed_where_it_is_each_frame():
    from skyfind import report
    story = _story(3)
    r = replay.Renderer(story, width=800)
    r.source = lambda k: np.full((r.fh, r.fw, 3), 20, np.uint8)
    story.dating_at = {k: [dict(name="Jupiter", kind="planet", x=0.0, y=0.0,
                                state="in view", path=[],
                                mark=(100.0 + 10 * k, 120.0, "dated it"))]
                       for k in (10, 40)}
    play = {s.k: s for s in replay.scenes(story, play=True)
            if s.phase == "play"}

    def gold_near(img, x):
        near = img[:, max(int(x * r.sx) - 20, 0):int(x * r.sx) + 20]
        return (np.abs(near.astype(int) - report.PLANET).sum(2) < 60).sum()

    a = r.render(play[10])[:r.fh, :r.fw]
    b = r.render(play[40])[:r.fh, :r.fw]
    assert gold_near(a, 200) > 20 and gold_near(b, 500) > 20
    assert gold_near(a, 500) == 0 and gold_near(b, 200) == 0   # it moved


def test_levels_lift_a_dark_sky_but_not_too_far():
    rng = np.random.default_rng(5)
    sky = rng.normal(20, 3, (200, 300, 3)).clip(0, 255).astype(np.uint8)
    sky[::37, ::41] = 90                               # a few stars
    lut = replay.levels(sky)
    assert np.all(np.diff(lut.astype(int)) >= 0) and lut[255] == 255
    assert lut[15] == 0                                # the sky's black
    assert lut[60] > 60                                # lifted...
    assert lut[60] <= 255 * ((60 - 15) / 127.5) ** 0.85 + 1   # ...at most 2x


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_clip_frames_are_the_frames_asked_for(tmp_path):
    path = tmp_path / "clip.mp4"
    out = replay.VideoOut(path, 64, 48, gif=False)
    for k in range(12):
        out.add(np.full((48, 64, 3), 20 * k, np.uint8), 1)
    assert out.close() == 12
    got = replay.ClipFrames(path, (32, 24))
    for k in (0, 3, 9, 3, 5):                           # forward, and back
        assert got(k).shape == (24, 32, 3)
        assert abs(float(got(k).mean()) - 20 * k) < 4
    assert got(12) is None
    got.close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_each_writer_keeps_its_own_clock(tmp_path):
    """Still scenes last 3 ticks, played frames 1: the MP4 (30 fps) holds
    the first for 3 frames, the GIF (10 fps) keeps one played frame in 3."""
    img = np.full((48, 64, 3), 128, np.uint8)
    mp4 = replay.VideoOut(tmp_path / "a.mp4", 64, 48, gif=False)
    gif = replay.VideoOut(tmp_path / "a.gif", 64, 48, gif=True, fps=10)
    for ticks in [3] * 4 + [1] * 9:
        mp4.add(img, ticks)
        gif.add(img, ticks)
    assert mp4.close() == 21 and gif.close() == 7


def test_support_kept_up_to_date_matches_counting_afresh():
    """The replay adds a track at a time: the running count equals
    `candidates` over everything so far, at every step."""
    rng = np.random.default_rng(11)
    counter = replay.Support()
    fits = []
    for i in range(12):
        new = [dict(lat=rng.normal(-33, 1.5), lon=rng.normal(151, 2.0) % 360
                    - 180 * (rng.random() < 0.1), T=rng.normal(0, 300))
               for _ in range(rng.integers(0, 60))]
        counter.add(new, i)
        fits += [dict(f, track=i) for f in new]
        if fits:
            support, top = replay.candidates(fits)
            assert np.array_equal(counter.support, support)
            assert int(counter.support.max()) == top
