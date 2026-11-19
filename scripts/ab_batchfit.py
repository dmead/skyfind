"""A/B: `solve.fit_candidate` (SciPy, one at a time) against
`batchfit.fit_candidates` (all of a track's candidates at once) on the same
candidates from a real clip's work folder. Reports agreement on which fits
are good, how far the good ones differ, and the time each took.

    python scripts/ab_batchfit.py CLIP [--workdir W] [--tle F ...]
        [--spacetrack] [--tracks 6] [--per-track 300]
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
    ap.add_argument("--tracks", type=int, default=6)
    ap.add_argument("--per-track", type=int, default=300)
    ap.add_argument("--window", type=float, default=12.0)
    a = ap.parse_args()

    from skyfind import batchfit, date, pipeline, solve, tles
    wd = pipeline.Workdir(a.clip, a.workdir)
    s = pipeline._setup(wd, stars=None, workers=None, lat=None, log=print)
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
    rows, anchors = pipeline._anchors(wd, usable, None, print)
    pick = np.unique(np.linspace(0, len(anchors) - 1, a.tracks).astype(int))
    tmp = pipeline._orbit_folder(wd, len({e.norad for e in els}), n, spf,
                                 a.window, print)
    P = None
    rng = np.random.default_rng(0)
    try:
        P = solve.build_problem(cam, up, els, t_ref, a.window * 3600, n,
                                lat0=lat, sky=sky, folder=tmp,
                                workers=solve.default_workers(), log=print)
        tot = dict(n=0, both=0, only_old=0, only_new=0, t_old=0.0, t_new=0.0,
                   n_new=0)
        dT, dlat, dlon, dmed = [], [], [], []
        for ti in pick:
            segs = solve.segs_from_pixels(cam, anchors[ti]["segs"])
            cands = solve.candidates(P, segs)
            if not cands:
                continue
            tic = time.time()
            new = batchfit.fit_candidates(P, segs, cands)
            t_new = time.time() - tic
            sub = rng.choice(len(cands), min(a.per_track, len(cands)),
                             replace=False)
            tic = time.time()
            old = [solve.fit_candidate(P, segs, cands[i]) for i in sub]
            t_old = time.time() - tic
            tot["t_old"] += t_old * len(cands) / len(sub)   # scaled to all
            tot["t_new"] += t_new
            tot["n_new"] += len(cands)
            for i, o in zip(sub, old):
                nw = new[i]
                go = o is not None and o["med_err"] < solve.GOOD_FIT
                gn = nw is not None and nw["med_err"] < solve.GOOD_FIT
                tot["n"] += 1
                tot["both"] += go and gn
                tot["only_old"] += go and not gn
                tot["only_new"] += gn and not go
                if go and gn:
                    dT.append(abs(o["T"] - nw["T"]))
                    dlat.append(abs(o["lat"] - nw["lat"]))
                    dlon.append(abs((o["lon"] - nw["lon"] + 180) % 360 - 180))
                    dmed.append(nw["med_err"] - o["med_err"])
            print(f"track {ti}: {len(cands)} candidates; SciPy {t_old:.1f} s "
                  f"for {len(sub)}, batch {t_new:.2f} s for all", flush=True)
        q = lambda v: np.percentile(v, [50, 90, 99]).round(4) if v else "-"  # noqa
        print(f"\ncompared {tot['n']}: good in both {tot['both']}, only SciPy "
              f"{tot['only_old']}, only batch {tot['only_new']}")
        print(f"good in both, |dT| s p50/90/99 {q(dT)}; |dlat| deg {q(dlat)}; "
              f"|dlon| deg {q(dlon)}; med_err batch-SciPy {q(dmed)}")
        print(f"time for all {tot['n_new']} candidates: SciPy ~"
              f"{tot['t_old']:.0f} s (one process, extrapolated), batch "
              f"{tot['t_new']:.1f} s -> {tot['t_old'] / max(tot['t_new'], 1e-9):.0f}x")
    finally:
        if P is not None:
            P.orbits.close()
        P = None
        import gc
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
