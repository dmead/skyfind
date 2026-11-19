"""Blind site/time solve on the synthetic sky.

The solver gets what the pipeline would give it: trail end points with 1 px
noise, flagged where they touch the frame edge or the skyline (not where the
Earth's shadow cut them — nothing in the image says so), the camera model,
an up vector tilted 1.5 deg / 0.5 deg the way a ridge-top skyline tilts it,
and a nominal clock 3.7 h off. It searches +-12 h: every longitude.
"""

import collections
import math
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skyfind import shadow, solve, trails

from .synth import World

N_TRACKS = 8


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    w = World()
    truth = w.segments()
    ex = trails.Extractor(w.ground, (w.height, w.width))
    rng = np.random.default_rng(5)

    def cut(x, y):
        return bool(ex.cut[int(np.clip(y, 0, w.height - 1)),
                           int(np.clip(x, 0, w.width - 1))])

    rows = collections.defaultdict(list)
    for s in truth:
        x1, y1, x2, y2 = np.array([s["x1"], s["y1"], s["x2"], s["y2"]]) + \
            rng.normal(0, 1.0, 4)
        rows[s["k"]].append(dict(k=s["k"], x1=x1, y1=y1, x2=x2, y2=y2,
                                 len=math.hypot(x2 - x1, y2 - y1),
                                 clip_a=cut(x1, y1), clip_b=cut(x2, y2),
                                 truth=s["norad"]))
    tracks = trails.link(rows)
    up = Rotation.from_euler("xz", [1.5, 0.5], degrees=True).apply(w.up_cam)
    cam = w.true_camera()
    t_ref = w.t(0).tt + 3.7 / 24
    folder = tmp_path_factory.mktemp("orbits")
    nw = min(8, os.cpu_count() or 1)
    P = solve.build_problem(cam, up, w.elements, t_ref, 12 * 3600,
                            w.n_frames, folder=folder, workers=nw, log=None)
    multi = sorted([t for t in tracks if len(t["frames"]) >= 3],
                   key=lambda t: -len(t["frames"]))[:N_TRACKS]
    segs = [solve.segs_from_pixels(cam, t["segs"]) for t in multi]
    pool = solve.Workers(P, folder, nw)
    fits = solve.fit_tracks(P, segs, pool=pool, log=None)
    clusters = solve.cluster(fits)
    sol = solve.joint(P, segs, fits, clusters[0], log=None)
    allrows = [r for k in sorted(rows) for r in rows[k]]
    allsegs = solve.segs_from_pixels(cam, allrows)
    yield dict(world=w, P=P, rows=allrows, segs=allsegs, tracks=tracks,
               multi=multi, fits=fits, clusters=clusters, sol=sol, pool=pool,
               true_T=(w.t(0).tt - t_ref) * 86400)
    pool.close()
    P.orbits.close()


def test_tracks_link_one_object_each(scene):
    multi = [t for t in scene["tracks"] if len(t["frames"]) >= 3]
    pure = [len({s["truth"] for s in t["segs"]}) == 1 for t in multi]
    assert len(multi) >= 20 and np.mean(pure) > 0.95


def test_each_anchor_track_fits_its_own_object_at_the_true_site(scene):
    w = scene["world"]
    for t, r in zip(scene["multi"], scene["fits"]):
        own = [f for f in r["fits"] if f["norad"] == t["segs"][0]["truth"]
               and abs(f["T"] - scene["true_T"]) < 5]
        assert own, t["frames"]
        assert abs(own[0]["lat"] - w.lat) < 0.2
        assert abs(own[0]["lon"] - w.lon) < 0.2


def test_distinct_objects_agree_only_at_the_true_site(scene):
    top, second = scene["clusters"][:2]
    assert top["n_objects"] >= N_TRACKS - 1
    assert second["n_objects"] <= top["n_objects"] // 2


def test_blind_solution_recovers_site_clock_and_exposure(scene):
    w, sol = scene["world"], scene["sol"]
    assert abs(sol["lat"] - w.lat) < 0.005
    assert abs(sol["lon"] - w.lon) < 0.005
    assert abs(sol["T"] - scene["true_T"]) < 0.1                 # seconds
    assert abs(sol["E"] - w.exposure) < 0.1
    assert abs(sol["height_km"] - w.height_km) < 0.3
    assert sol["frame0_utc"] == w.t0.strftime("%Y-%m-%d %H:%M:%S")
    # the error bars mean something
    assert abs(sol["lat"] - w.lat) < 4 * sol["sig_lat"] + 1e-3
    assert abs(sol["lon"] - w.lon) < 4 * sol["sig_lon"] + 1e-3


def test_identifies_nearly_every_trail_correctly(scene):
    ids = solve.identify(scene["P"], scene["segs"], scene["sol"],
                         pool=scene["pool"])
    right = sum(scene["rows"][d["seg"]]["truth"] == d["norad"] for d in ids)
    assert len(ids) > 0.9 * len(scene["segs"])
    assert right > 0.99 * len(ids)
    rows = solve.objects(ids)
    assert rows[0]["n"] == 1 and {r["conf"] for r in rows} <= \
        {"confirmed", "probable"}


def test_a_wrong_clock_matches_little(scene):
    null = solve.null_rate(scene["P"], scene["segs"], scene["sol"],
                           offsets=(-600, 600), pool=scene["pool"])
    for v in null.values():
        assert v["segments"] < 0.05 * len(scene["segs"])


def test_parallel_matches_serial(scene, monkeypatch):
    """The pool splits work differently, never the answer (on the CPU:
    the GPU's arithmetic isn't bit-for-bit the CPU's; see the GPU test)."""
    monkeypatch.setenv("SKYFIND_GPU", "0")
    P, pool = scene["P"], scene["pool"]
    segs = [solve.segs_from_pixels(P.camera, t["segs"])
            for t in scene["multi"][:2]]
    serial = solve.fit_tracks(P, segs, log=None)
    assert serial == solve.fit_tracks(P, segs, pool=pool, log=None)
    some = scene["segs"][:60]
    serial = solve.identify(P, some, scene["sol"])
    assert serial == solve.identify(P, some, scene["sol"], pool=pool)
    # the joint fit's error refits, side by side
    tracks = [solve.segs_from_pixels(P.camera, t["segs"])
              for t in scene["multi"]]
    top = scene["clusters"][0]
    serial = solve.joint(P, tracks, scene["fits"], top, log=None)
    assert serial["jackknife_deg"] is not None
    assert serial == solve.joint(P, tracks, scene["fits"], top, pool=pool,
                                 log=None)


def test_parallel_propagation_matches_serial(scene, tmp_path):
    w = scene["world"]
    lines = [(e.line1, e.line2) for e in w.elements]
    t0 = w.t(0).tt
    a = solve.Orbits.propagate(lines, t0, t0 + 0.02, log=None)
    b = solve.Orbits.propagate(lines, t0, t0 + 0.02, folder=tmp_path,
                               workers=3, log=None)
    assert np.array_equal(a.r, b.r, equal_nan=True)
    assert np.array_equal(a.v, b.v, equal_nan=True)
    b.close()
    c = solve.Orbits.propagate(lines, t0, t0 + 0.02, shared=True, workers=3,
                               log=None)
    assert c.shared
    assert np.array_equal(a.r, c.r, equal_nan=True)
    assert np.array_equal(a.v, c.v, equal_nan=True)
    c.save(tmp_path / "shm")                      # a note, not the data
    assert not list((tmp_path / "shm").glob("*.npy"))
    back = solve.Orbits.load(tmp_path / "shm")
    assert np.array_equal(back.r, c.r, equal_nan=True)
    back.close()
    c.close()


def test_the_pool_builds_and_shares_the_orbit_table(scene, tmp_path):
    """The pool starts first, builds the table in shared memory, then
    solves on it: the same table and the same answers as in process."""
    w, P = scene["world"], scene["P"]
    with solve.Workers(folder=tmp_path, n=3) as pool:
        P2 = solve.build_problem(P.camera, P.site.up_cam, w.elements,
                                 P.t_ref, P.twin, w.n_frames, lat0=P.lat0,
                                 shared=True, folder=tmp_path, pool=pool,
                                 log=None)
        pool.attach(P2)
        try:
            assert np.array_equal(P2.orbits.r, P.orbits.r, equal_nan=True)
            some = scene["segs"][:40]
            assert solve.identify(P2, some, scene["sol"], pool=pool) == \
                solve.identify(P, some, scene["sol"])
        finally:
            P2.orbits.close()


def test_a_free_site_is_found_without_up(scene, tmp_path):
    """Pure sky (nothing says which way is up): lines of sight from each
    candidate object back to the ground stand in for the skyline, and the
    tracks still agree only at the true site."""
    import dataclasses
    from skyfind import horizon
    P, w = scene["P"], scene["world"]
    P2 = dataclasses.replace(P, sky=horizon.sky_rays(P.camera, w.ground == 0))
    segs = [solve.segs_from_pixels(P.camera, t["segs"])
            for t in scene["multi"][:5]]
    with solve.Workers(P2, tmp_path, min(8, os.cpu_count() or 1)) as pool:
        fits = solve.fit_tracks(P2, segs, pool=pool, log=None)
    top = solve.cluster(fits)[0]
    assert top["n_objects"] >= 4
    assert abs(top["lat"] - w.lat) < 0.3 and abs(top["lon"] - w.lon) < 0.3
    assert abs(top["T"] - scene["true_T"]) < 5


def _gpu():
    from skyfind import gpu
    return gpu.available()


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA GPU (CuPy)")
def test_the_gpu_search_finds_what_the_cpu_does(scene):
    """The free-site pass search on the GPU: single precision with margins,
    then the same double-precision tests, so the same candidates (a
    refinement step's tie can go the other way, rarely)."""
    import dataclasses
    from skyfind import gpu, horizon
    P, w = scene["P"], scene["world"]
    P2 = dataclasses.replace(P, sky=horizon.sky_rays(P.camera, w.ground == 0))
    dev = gpu.DeviceOrbits(P2.orbits)
    try:
        for t in scene["multi"][:3]:
            segs = solve.segs_from_pixels(P.camera, t["segs"])
            key = lambda c: (c[0], round(c[1] / 2))                  # noqa
            cpu = {key(c) for c in solve.free_passes(P2, segs)}
            on_gpu = {key(c) for c in gpu.free_passes(P2, dev, segs)}
            assert cpu and len(cpu ^ on_gpu) <= 0.01 * len(cpu) + 1
    finally:
        dev.free()


def _native():
    from skyfind import native
    return native.available()


@pytest.mark.skipif(not _native(),
                    reason="needs the native search library (SKYFIND_NATIVE)")
def test_the_native_search_finds_what_the_cpu_does(scene):
    """The GPU's free-site kernels compiled for the CPU (the phone's search,
    android/native/freesearch.cpp): the same candidates as Python's."""
    import dataclasses
    from skyfind import horizon, native
    P, w = scene["P"], scene["world"]
    P2 = dataclasses.replace(P, sky=horizon.sky_rays(P.camera, w.ground == 0))
    tracks = [solve.segs_from_pixels(P.camera, t["segs"])
              for t in scene["multi"][:3]]
    key = lambda c: (c[0], round(c[1] / 2))                          # noqa
    for segs, mine in zip(tracks, native.free_passes_all(P2, tracks)):
        cpu = {key(c) for c in solve.free_passes(P2, segs)}
        assert cpu and len(cpu ^ {key(c) for c in mine}) <= 0.01 * len(cpu) + 1


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA GPU (CuPy)")
def test_the_gpu_skyline_search_finds_what_the_cpu_does(scene):
    """The pass search with a skyline on the GPU: the catalogue sweep in
    single precision like the CPU's, the checks in double; the same
    candidates (a rounding tie between sample times can go the other way,
    rarely)."""
    from skyfind import gpu
    P = scene["P"]
    dev = gpu.DeviceOrbits(P.orbits)
    try:
        for t in scene["multi"][:4]:
            segs = solve.segs_from_pixels(P.camera, t["segs"])
            near = solve.near_objects(P, segs)
            J, Ts = gpu.near_objects(P, dev, segs)
            cpu_near = {(j, T) for j, S in near.items() for T in S}
            assert cpu_near and                 len(cpu_near ^ set(zip(J.tolist(), Ts.tolist()))) <=                 0.001 * len(cpu_near) + 1
            key = lambda c: (c[0], round(c[1]), c[4], c[5])          # noqa
            cpu = {key(c) for c in solve.check_passes(P, segs, near)}
            on_gpu = {key(c) for c in gpu.skyline_passes(P, dev, segs)}
            assert cpu and len(cpu ^ on_gpu) <= 0.01 * len(cpu) + 1
    finally:
        dev.free()


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA GPU (CuPy)")
def test_the_gpu_fits_find_what_the_cpu_does(scene, monkeypatch):
    """Every track's candidates fitted on the GPU at once: the same good
    fits as the CPU's batched fitter, to a hundredth of a second."""
    P, pool = scene["P"], scene["pool"]
    segs = [solve.segs_from_pixels(P.camera, t["segs"])
            for t in scene["multi"][:3]]
    monkeypatch.setenv("SKYFIND_GPU", "0")
    cpu = solve.fit_tracks(P, segs, pool=pool, log=None)
    monkeypatch.setenv("SKYFIND_GPU", "1")
    on_gpu = solve.fit_tracks(P, segs, pool=pool, log=None)
    for a, b in zip(cpu, on_gpu):
        assert a["ncand"] == b["ncand"]
        ka = {(f["norad"], round(f["T"])) for f in a["fits"]
              if f["med_err"] < solve.GOOD_FIT}
        kb = {(f["norad"], round(f["T"])) for f in b["fits"]
              if f["med_err"] < solve.GOOD_FIT}
        assert len(ka ^ kb) <= 0.01 * len(ka) + 1
        for f, g in zip(a["fits"], b["fits"]):
            if f["norad"] == g["norad"] and abs(f["T"] - g["T"]) < 1:
                assert abs(f["T"] - g["T"]) < 0.01
                assert abs(f["lat"] - g["lat"]) < 1e-3


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA GPU (CuPy)")
def test_the_gpu_join_step_matches_the_cpu(scene):
    """The joint fit's join step for several starts at once on the GPU:
    the same tracks join, with the same objects and end orders."""
    from skyfind import gpu
    P, clusters = scene["P"], scene["clusters"]
    tracks = [solve.segs_from_pixels(P.camera, t["segs"])
              for t in scene["multi"]]
    orders = solve.join_orders(scene["fits"])
    # the top cluster cut to two tracks, so the rest have to join
    tops = [dict(clusters[0], members=clusters[0]["members"][:2])] + \
        clusters[:3]
    starts = [solve.joint_start(P, tracks, c) for c in tops]
    dev = gpu.DeviceOrbits(P.orbits)
    try:
        on_gpu = gpu.join_tracks(P, dev, tracks, orders, starts)
    finally:
        dev.free()
    assert len(on_gpu[0]) >= 3
    for st, got in zip(starts, on_gpu):
        cpu = solve.join_tracks(P, tracks, orders, st)
        assert cpu.keys() == got.keys()
        for ti in cpu:
            assert cpu[ti][:2] == got[ti][:2]
            assert abs(cpu[ti][2] - got[ti][2]) < 1e-5     # deg


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA GPU (CuPy)")
def test_the_gpu_identifies_what_the_cpu_does(scene):
    """Every segment at several solutions at once (the right clock and wrong
    ones) on the GPU: the same object for each segment as on the CPU, to
    the same error."""
    from skyfind import gpu
    P, sol, segs = scene["P"], scene["sol"], scene["segs"]
    sols = [sol] + [dict(sol, T=sol["T"] + off) for off in (-600, 300)]
    cpu = solve.identify_all(P, segs, sols, pool=scene["pool"])
    dev = gpu.DeviceOrbits(P.orbits)
    try:
        on_gpu = solve.identify_all(P, segs, sols, dev=dev)
    finally:
        dev.free()
    assert len(cpu[0]) > 0.9 * len(segs)
    for a, b in zip(cpu, on_gpu):
        assert [(d["seg"], d["norad"]) for d in a] == \
            [(d["seg"], d["norad"]) for d in b]
        for d, e in zip(a, b):
            assert abs(d["err_deg"] - e["err_deg"]) < 1.5e-4   # rounding
            assert d["dt"] == e["dt"] and d["span"] == e["span"]
            assert (d["utc"], d["k"], d["clipped"]) == \
                (e["utc"], e["k"], e["clipped"])


def test_only_a_clear_winner_skips_the_tiebreak():
    """The top cluster goes straight to its joint fit only by TIE_MARGIN
    (1.5x) over the next; a narrow lead is identified like a tie."""
    from skyfind import pipeline
    assert pipeline.clear_winner(6, 4) and pipeline.clear_winner(15, 7)
    assert pipeline.clear_winner(5, 0)
    assert not pipeline.clear_winner(12, 11)
    assert not pipeline.clear_winner(11, 11)


def test_every_true_trail_could_be_a_sunlit_satellite(scene):
    """The synthetic sky draws satellites only while the Sun lights them,
    so the shadow test keeps every trail and every track (a segment at
    the slowest it could be moving, a track at its own drift)."""
    P = scene["P"]
    segs = scene["segs"]
    assert shadow.possible(P, segs, shadow.rates(P, segs)).all()
    tsegs = [solve.segs_from_pixels(P.camera, t["segs"])
             for t in scene["tracks"] if len(t["frames"]) >= 2]
    assert shadow.tracks_possible(P, tsegs).all()


def test_a_streak_in_the_midnight_shadow_is_no_satellite(scene):
    """Towards the anti-solar point the line of sight stays in the umbra
    far beyond any orbit: whatever moves there is lit by itself (an
    aircraft); a slow speck high enough to be sunlit elsewhere still
    could be a satellite."""
    P = scene["P"]
    k = P.n_frames // 2
    anti = -shadow.suns(P.t_ref)[1]
    side = np.cross(anti, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    turn = lambda v, deg: Rotation.from_rotvec(  # noqa: E731
        np.radians(deg) * np.cross(anti, side)).apply(v)
    a, b = turn(anti, -1.0), turn(anti, 1.0)
    streak = solve.Seg(k, a, b)
    assert not shadow.possible(P, [streak], np.array([0.1]))[0]
    grid = np.array(solve.SITE_GRID, float) + [P.lat0, 0.0]
    sites = P.site.gcrs(k, grid[:, 0], grid[:, 1], P.h0)
    assert shadow.fastest(sites, anti, shadow.suns(P.t_ref)) == 0.0


def test_the_umbra_starts_where_its_geometry_says():
    """Straight up from a site with the Sun 30 deg down, the umbra (its
    cone narrowing by UMBRA) ends at (R + h) (cos 30 + sin 30 tan UMBRA)
    = R: 967 km up, a little below the cylinder's 987."""
    d = np.radians(30.0)
    R = shadow.R_EARTH
    up = np.array([1.0, 0.0, 0.0])
    sun = np.array([-np.sin(d), np.cos(d), 0.0])      # 120 deg from up
    h = R / (np.cos(d) + np.sin(d) * np.tan(shadow.UMBRA)) - R
    assert 960 < h < 975
    p = (R + np.array([h - 1.0, h + 1.0]))[:, None] * up
    assert list(shadow.lit(p, sun)) == [False, True]
