"""A/B: the batched fits on the CPU (`batchfit.fit_candidates`, track by
track) against the GPU (`gpu.fit_all`, every track at once), on the same
candidates from a clip's work folder (free site: candidates from the GPU
search). Same good fits expected, and the time each took.

    python scripts/ab_gpufit.py CLIP [--workdir W] [--tle F ...]
        [--spacetrack] [--tracks 12]
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--workdir")
    ap.add_argument("--tle", action="append", default=[])
    ap.add_argument("--spacetrack", action="store_true")
    ap.add_argument("--tracks", type=int, default=12)
    ap.add_argument("--window", type=float, default=12.0)
    a = ap.parse_args()

    from skyfind import batchfit, date, gpu, pipeline, solve, tles
    wd = pipeline.Workdir(a.clip, a.workdir)
    s = pipeline._setup(wd, stars=None, workers=None, lat=None, log=print,
                        free_site=True)
    cam, up, sky, lat, n, usable = (s[k] for k in ("cam", "up", "sky", "lat",
                                                    "n", "usable"))
    t_ref = wd.read("date.json")["tt"]
    ts = date.data.timescale()
    spf = cam.seconds_per_frame
    first = ts.tt_jd(t_ref - a.window / 24).utc_datetime()
    last = ts.tt_jd(t_ref + (n * spf / 3600 + a.window) / 24).utc_datetime()
    els = tles.acquire(first, last, path=a.tle or None,
                       spacetrack=a.spacetrack)
    _, anchors = pipeline._anchors(wd, usable, None, print)
    pick = np.unique(np.linspace(0, len(anchors) - 1, a.tracks).astype(int))
    tmp = pipeline._orbit_folder(wd, len({e.norad for e in els}), n, spf,
                                 a.window, print)
    P = dev = None
    try:
        P = solve.build_problem(cam, up, els, t_ref, a.window * 3600, n,
                                lat0=lat, sky=sky, folder=tmp,
                                workers=solve.default_workers(), log=print)
        dev = gpu.DeviceOrbits(P.orbits)
        tracks = {int(i): solve.segs_from_pixels(cam, anchors[i]["segs"])
                  for i in pick}
        cands = {i: gpu.free_passes(P, dev, t) for i, t in tracks.items()}
        total = sum(len(c) for c in cands.values())
        print(f"{total} candidates over {len(tracks)} tracks")
        gpu.fit_all(P, dev, tracks, {i: c[:50] for i, c in cands.items()})
        dev.cp.cuda.Device().synchronize()                    # warm up
        tic = time.time()
        on_gpu = gpu.fit_all(P, dev, tracks, cands)
        dev.cp.cuda.Device().synchronize()
        t_gpu = time.time() - tic
        tic = time.time()
        on_cpu = {i: batchfit.fit_candidates(P, tracks[i], cands[i])
                  for i in tracks}
        t_cpu = time.time() - tic
        both = only_cpu = only_gpu = 0
        dT, dlat, dmed = [], [], []
        for i in tracks:
            for c_, g_ in zip(on_cpu[i], on_gpu[i]):
                gc = c_ is not None and c_["med_err"] < solve.GOOD_FIT
                gg = g_ is not None and g_["med_err"] < solve.GOOD_FIT
                both += gc and gg
                only_cpu += gc and not gg
                only_gpu += gg and not gc
                if gc and gg:
                    dT.append(abs(c_["T"] - g_["T"]))
                    dlat.append(abs(c_["lat"] - g_["lat"]))
                    dmed.append(g_["med_err"] - c_["med_err"])
        q = lambda v: np.percentile(v, [50, 99, 100]).round(5) if v else "-"  # noqa
        print(f"good in both {both}, only CPU {only_cpu}, only GPU "
              f"{only_gpu}; |dT| s p50/99/max {q(dT)}; |dlat| {q(dlat)}; "
              f"med_err GPU-CPU {q(dmed)}")
        print(f"CPU (one process) {t_cpu:.1f} s, GPU {t_gpu:.2f} s -> "
              f"{t_cpu / max(t_gpu, 1e-9):.0f}x")
    finally:
        if dev is not None:
            dev.free()
        if P is not None:
            P.orbits.close()
        P = None
        import gc
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
