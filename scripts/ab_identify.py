"""A/B: `solve.identify` (now vectorised over each segment's nearby
objects) against the per-object loop it replaced, at a solved clip's
solution. Same IDs expected (ties aside), and the time each took.

    python scripts/ab_identify.py CLIP [--workdir W] [--tle F ...]
        [--spacetrack] [--free-site] [--segments 150]
"""
from __future__ import annotations

import argparse
import math
import shutil
import time

import numpy as np


def old_identify(P, segs, sol, *, tol=0.25, centre_s=8.0, prefilter_deg=8.0):
    from skyfind.solve import _deg, unit
    lat, dlst, E = sol["slat"], sol["dlst"], sol["E"]
    h = sol.get("height_km", P.h0)
    T = sol["T"]
    ss = np.arange(-25, 25.25, 0.5)
    cos_pre = math.cos(math.radians(prefilter_deg))
    out = []
    for i, s in enumerate(segs):
        tk = float(P.tk(s.k))
        rs = P.site.gcrs(s.k, lat, dlst, h)
        mid = unit(s.a + s.b)
        cand = set()
        for dt in np.arange(-25, 26, 5.0):
            d = P.orbits.at(tk + (T + dt) / 86400) - rs
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            cand |= set(np.flatnonzero(d @ mid > cos_pre).tolist())
        best = None
        for j in cand:
            d = P.dirs(j, np.full(len(ss), s.k), ss, T, lat, dlst, h)
            if np.isnan(d).any():
                continue
            ca, cb = d @ s.a, d @ s.b
            ia, ib = int(np.argmax(ca)), int(np.argmax(cb))
            if ia in (0, len(ss) - 1) or ib in (0, len(ss) - 1):
                continue
            err = max(float(_deg(ca[ia])), float(_deg(cb[ib])))
            tm, span = (ss[ia] + ss[ib]) / 2, abs(ss[ib] - ss[ia])
            if s.clip_a or s.clip_b:
                inside = max(abs(ss[ia]), abs(ss[ib])) <= E / 2 + 3
            else:
                inside = abs(tm) <= centre_s
            if err < tol and inside and span <= E + 3:
                if best is None or err < best[1]:
                    best = (j, err, tm, span)
        if best:
            out.append((i, int(P.norad[best[0]]), round(best[1], 4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--workdir")
    ap.add_argument("--tle", action="append", default=[])
    ap.add_argument("--spacetrack", action="store_true")
    ap.add_argument("--free-site", action="store_true")
    ap.add_argument("--segments", type=int, default=150)
    ap.add_argument("--window", type=float, default=12.0)
    a = ap.parse_args()

    from skyfind import date, pipeline, solve, tles
    wd = pipeline.Workdir(a.clip, a.workdir)
    s = pipeline._setup(wd, stars=None, workers=None, lat=None, log=print,
                        free_site=a.free_site)
    cam, up, sky, lat, n, usable = (s[k] for k in ("cam", "up", "sky", "lat",
                                                    "n", "usable"))
    sol = wd.read("solution.json")
    d = wd.read("date.json")
    t_ref = d["tt"]
    ts = date.data.timescale()
    spf = cam.seconds_per_frame
    first = ts.tt_jd(t_ref - a.window / 24).utc_datetime()
    last = ts.tt_jd(t_ref + (n * spf / 3600 + a.window) / 24).utc_datetime()
    els = tles.acquire(first, last, path=a.tle or None,
                       spacetrack=a.spacetrack)
    rows, _ = pipeline._anchors(wd, usable, None, print)
    segs = solve.segs_from_pixels(cam, rows)
    pick = np.unique(np.linspace(0, len(segs) - 1, a.segments).astype(int))
    segs = [segs[i] for i in pick]
    tmp = pipeline._orbit_folder(wd, len({e.norad for e in els}), n, spf,
                                 a.window, print)
    P = None
    try:
        P = solve.build_problem(cam, up, els, t_ref, a.window * 3600, n,
                                lat0=lat, sky=sky, folder=tmp,
                                workers=solve.default_workers(), log=print)
        if abs(sol["frame0_tt"] - (P.t_ref + sol["T"] / 86400)) > 1e-6:
            print("note: the solution's clock was solved against another "
                  "nominal date; using its frame 0 time")
            sol = dict(sol, T=(sol["frame0_tt"] - P.t_ref) * 86400)
        tic = time.time()
        old = old_identify(P, segs, sol)
        t_old = time.time() - tic
        tic = time.time()
        new = [(d["seg"], d["norad"], d["err_deg"])
               for d in solve.identify(P, segs, sol)]
        t_new = time.time() - tic
        o = {i: (k, e) for i, k, e in old}
        nw = {i: (k, e) for i, k, e in new}
        same = sum(1 for i in o if i in nw and nw[i][0] == o[i][0])
        diff = [(i, o[i], nw[i]) for i in o if i in nw and nw[i][0] != o[i][0]]
        print(f"{len(segs)} segments: old {len(old)} IDs in {t_old:.1f} s, "
              f"new {len(new)} IDs in {t_new:.1f} s ({t_old / t_new:.0f}x); "
              f"same object {same}, different {len(diff)}, only old "
              f"{len(set(o) - set(nw))}, only new {len(set(nw) - set(o))}")
        for i, a_, b_ in diff[:5]:
            print(f"  segment {i}: old {a_} new {b_}")
    finally:
        if P is not None:
            P.orbits.close()
        P = None
        import gc
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
