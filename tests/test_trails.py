"""Trail extraction from rendered (original, cleaned) frame pairs."""

import numpy as np
import pytest

from skyfind import trails

from .synth import World


@pytest.fixture(scope="module")
def world():
    return World()


def _frames_with_trails(w, n=3):
    segs = w.segments(range(560, 600))
    by = {}
    for s in segs:
        by.setdefault(s["k"], []).append(s)
    ks = sorted(by, key=lambda k: -len(by[k]))[:n]
    return {k: by[k] for k in ks}


def test_segments_come_out_with_their_end_points(world):
    ex = trails.Extractor(world.ground, (world.height, world.width))
    found, missed = 0, 0
    for k, truth in _frames_with_trails(world).items():
        orig = world.render(k, truth)
        clean = world.render(k)                      # same noise, no trails
        got = ex(orig, clean)
        ends = np.array([[g["x1"], g["y1"], g["x2"], g["y2"]] for g in got])
        for s in truth:
            a, b = np.array([s["x1"], s["y1"]]), np.array([s["x2"], s["y2"]])
            # worst end error against each extracted segment, either way round
            same = np.maximum(np.linalg.norm(ends[:, :2] - a, axis=1),
                              np.linalg.norm(ends[:, 2:] - b, axis=1))
            swap = np.maximum(np.linalg.norm(ends[:, 2:] - a, axis=1),
                              np.linalg.norm(ends[:, :2] - b, axis=1))
            if np.minimum(same, swap).min() < 4.0:
                found += 1
            else:
                missed += 1
        for g in got:                                # steady, neutral trails
            assert g["chroma"] < 0.1
    assert found >= 0.85 * (found + missed)


def test_ends_at_the_edge_or_skyline_are_flagged():
    w = World()
    ex = trails.Extractor(w.ground, (w.height, w.width))
    h, width = w.height, w.width
    res = np.zeros((h, width), np.float32)
    import cv2
    # one trail running off the left edge, one ending mid-sky
    cv2.line(res, (0, 300), (400, 320), 60, 2)
    cv2.line(res, (1500, 400), (1900, 380), 60, 2)
    got = sorted(ex.segments(res, np.repeat(res[..., None], 3, 2)
                             .astype(np.int16)), key=lambda g: g["x1"])
    assert len(got) == 2
    assert got[0]["clip_a"] and not got[0]["clip_b"]
    assert not got[1]["clip_a"] and not got[1]["clip_b"]


def test_link_follows_one_object_across_exposures():
    def seg(k, x0):
        return dict(x1=x0, y1=100 + 0.1 * x0, x2=x0 + 90, y2=109 + 0.1 * x0,
                    len=90.5)
    segs = {k: [seg(k, 100 + 100 * k)] for k in range(5)}
    segs[2].append(dict(x1=1000, y1=900, x2=1100, y2=800, len=141))
    tr = trails.link(segs)
    assert [len(t["frames"]) for t in tr] == [5, 1]


def test_ends_sit_at_half_maximum_whatever_the_brightness():
    """A thresholded blob runs long for bright trails; the half-maximum
    ends don't care how bright the trail is."""
    from .synth import _trail
    ex = trails.Extractor(None, (400, 800))
    lengths = {}
    for amp in (40.0, 200.0):
        img = np.zeros((400, 800), np.float32)
        _trail(img, (200.0, 200.0), (600.0, 230.0), amp)   # exact, blurred
        g = ex.segments(img, np.repeat(img[..., None], 3, 2).astype(np.int16))
        assert len(g) == 1
        lengths[amp] = np.hypot(g[0]["x2"] - g[0]["x1"],
                                g[0]["y2"] - g[0]["y1"])
        assert abs(lengths[amp] - np.hypot(400, 30)) < 0.5, amp
    assert abs(lengths[200.0] - lengths[40.0]) < 0.3


def test_clutter_is_what_lights_one_spot_frame_after_frame():
    """A lit edge or a slewing telescope flickers in one spot, every which
    way, frame after frame; a satellite moves on each frame, and a train's
    satellites pass the same spot minutes apart. Only the first is
    clutter."""
    rng = np.random.default_rng(4)
    segs: dict = {}

    def add(k, x1, y1, x2, y2):
        segs.setdefault(str(k), []).append(dict(
            x1=x1, y1=y1, x2=x2, y2=y2, len=float(np.hypot(x2 - x1, y2 - y1))))

    for k in range(10, 40):                      # flicker round (300, 300)
        a = rng.uniform(0, np.pi)
        add(k, 300 - 30 * np.cos(a), 300 - 30 * np.sin(a),
            300 + 30 * np.cos(a), 300 + 30 * np.sin(a))
    for k in range(10, 30):                      # one satellite moving on
        add(k, 100 + 60 * (k - 10), 700, 100 + 60 * (k - 10) + 55, 700)
    for sat in range(8):                         # a train along y = 900
        for i in range(5):
            k = 5 + 6 * sat + i
            add(k, 100 + 60 * i, 900, 155 + 60 * i, 900)
    grid = trails.clutter(segs, (1080, 1920))
    flick = [s for s in segs["20"] if abs(s["y1"] - 300) < 60]
    assert flick and all(trails.in_clutter(s, grid) for s in flick)
    others = [s for ss in segs.values() for s in ss
              if abs((s["y1"] + s["y2"]) / 2 - 300) > 100]
    assert not any(trails.in_clutter(s, grid) for s in others)


def test_one_frame_per_capture_when_frames_repeat():
    """25 fps put out at 60 fps: the clock stands still over repeats and
    moves by a fraction over a blend; the first frame at each whole capture
    is kept and the clock snapped there."""
    clock = [0, 0, 0, 1, 1, 1.52, 2.03, 2.03, 3.0, 3.0, 4.02, 4.02]
    keep, snapped = trails.captures(clock)
    assert keep == [0, 3, 6, 8, 10]
    assert [snapped[k] for k in keep] == [0, 1, 2, 3, 4]
    assert trails.captures([0, 1, 2, 4, 5]) == (None, [0, 1, 2, 4, 5])
    # an extra blended frame between captures: in half, out whole
    keep, snapped = trails.captures([0, 0, 1, 1, 1.55, 2.55, 2.55, 3.55])
    assert keep == [0, 2, 5, 7] and [snapped[k] for k in keep] == [0, 1, 2, 3]


def test_linking_counts_gaps_in_captures():
    """With one frame kept per capture, frames 0, 3, 5 are consecutive
    captures: a trail across them is one track."""
    seg = lambda x: dict(x1=x, y1=100.0, x2=x + 40, y2=100.0, len=40.0)  # noqa
    tracks = trails.link({0: [seg(0)], 3: [seg(45)], 5: [seg(90)]})
    assert [t["frames"] for t in tracks] == [[0, 3, 5]]


def _piece(x1, y1, x2, y2, clip_a, clip_b):
    return dict(x1=x1, y1=y1, x2=x2, y2=y2, len=float(np.hypot(x2 - x1,
                                                               y2 - y1)),
                width=2.0, flux=100.0, peak=50.0, b=10.0, g=10.0, r=10.0,
                chroma=0.0, blink=0.1, clip_a=clip_a, clip_b=clip_b)


def test_a_trail_behind_a_trunk_is_one_segment_again():
    """A trail across the frame behind a tree trunk comes out as two
    pieces, the ends facing the trunk cut; joined, it has the outer ends
    (free) and their flags. Pieces that don't face each other across land,
    or whose facing ends weren't cut, stay apart."""
    land = np.zeros((400, 800), np.uint8)
    land[:, 380:420] = 1
    left = _piece(100, 100, 360, 126, False, True)
    right = _piece(700, 160, 440, 134, False, True)    # drawn the other way
    other = _piece(100, 300, 300, 300, False, False)
    out = trails.join_pieces([left, other, right], land)
    assert len(out) == 2
    j = next(s for s in out if s.get("pieces") == 2)
    assert (j["x1"], j["y1"], j["x2"], j["y2"]) == (100, 100, 700, 160)
    assert not j["clip_a"] and not j["clip_b"]
    assert j["len"] == pytest.approx(np.hypot(600, 60))
    uncut = _piece(100, 100, 360, 126, False, False)
    assert len(trails.join_pieces([uncut, right], land)) == 2
    offset = _piece(700, 190, 440, 164, False, True)
    assert len(trails.join_pieces([left, offset], land)) == 2
    assert len(trails.join_pieces([left, right], 0 * land)) == 2


def test_colour_is_judged_against_the_stars_white():
    """A grade tints the stars and the satellites alike: a trail the colour
    of the stars is neutral whatever that colour is, and a red one
    (aircraft lights) isn't."""
    from skyfind import pipeline
    white = [0.36, 0.32, 0.32]                 # a bluish grade
    starlike = dict(b=36.0, g=32.0, r=32.0)
    assert pipeline._chroma(starlike, white) < 1e-9
    assert pipeline._chroma(starlike, [1 / 3] * 3) == pytest.approx(0.12)
    red = dict(b=20.0, g=30.0, r=50.0)
    assert pipeline._chroma(red, white) > pipeline.NEUTRAL
