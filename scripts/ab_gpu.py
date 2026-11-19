"""A/B: the free-site pass search on the CPU (`solve.free_passes`) against
the GPU (`gpu.free_passes`) for some of a clip's tracks: same candidates
expected, and the time each took.

    python scripts/ab_gpu.py CLIP [--workdir W] [--tle F ...] [--spacetrack]
        [--tracks 5]
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
    ap.add_argument("--tracks", type=int, default=5)
    ap.add_argument("--window", type=float, default=12.0)
    a = ap.parse_args()

    from skyfind import date, gpu, pipeline, solve, tles
    wd = pipeline.Workdir(a.clip, a.workdir)
    s = pipeline._setup(wd, stars=None, workers=None, lat=None, log=print,
                        free_site=True)
    cam, up, sky, lat, n, usable = (s[k] for k in ("cam", "up", "sky", "lat",
                                                    "n", "usable"))
    d = wd.read("date.json")
    t_ref = d["tt"]
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
        tic = time.time()
        dev = gpu.DeviceOrbits(P.orbits)
        print(f"orbit table to the GPU in {time.time() - tic:.1f} s")
        gpu.free_passes(P, dev, solve.segs_from_pixels(cam, anchors[0]["segs"]))
        t_cpu = t_gpu = 0.0
        for ti in pick:
            segs = solve.segs_from_pixels(cam, anchors[ti]["segs"])
            tic = time.time()
            c = solve.free_passes(P, segs)
            tc = time.time() - tic
            tic = time.time()
            g = gpu.free_passes(P, dev, segs)
            dev.cp.cuda.Device().synchronize()
            tg = time.time() - tic
            t_cpu += tc
            t_gpu += tg
            key = lambda r: (r[0], round(r[1] / 2))                  # noqa
            kc, kg = {key(r): r for r in c}, {key(r): r for r in g}
            both = set(kc) & set(kg)
            dT = max((abs(kc[k][1] - kg[k][1]) for k in both), default=0)
            dl = max((abs(kc[k][4] - kg[k][4]) for k in both), default=0)
            print(f"track {ti}: CPU {len(c)} candidates in {tc:.1f} s, GPU "
                  f"{len(g)} in {tg:.2f} s; matched {len(both)}, only CPU "
                  f"{len(set(kc) - set(kg))}, only GPU "
                  f"{len(set(kg) - set(kc))}; max |dT| {dT:.3f} s, |dlat| "
                  f"{dl:.4f}", flush=True)
        print(f"total: CPU {t_cpu:.0f} s (one process), GPU {t_gpu:.1f} s "
              f"-> {t_cpu / max(t_gpu, 1e-9):.0f}x")
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
