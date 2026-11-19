"""A/B: `solve.joint`'s join step (other tracks joining when one of their
fitted objects agrees), batched per track now, against the one-fit-at-a-time
loop it replaced. Same joined tracks expected, and the time each took.

    python scripts/ab_joint.py CLIP [--workdir W] [--tle F ...]
        [--spacetrack] [--free-site] [--clusters 2]
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np


def old_join(P, tracks, track_fits, top, join_err=0.15):
    from skyfind.solve import Assoc, end_order, fit, residuals
    ms = top["members"]
    p0 = [float(np.median([m[k] for m in ms]))
          for k in ("slat", "dlst", "T", "E")]
    sets = {m["track"]: Assoc(m["j"], tracks[m["track"]],
                              end_order(P, m["j"], tracks[m["track"]], p0))
            for m in ms}
    p, _, _ = fit(P, list(sets.values()), p0, f_scale=0.1,
                  x_scale=(0.1, 0.1, 5, 2, 0.5))
    for ti, r in enumerate(track_fits):
        if ti in sets:
            continue
        for f in r["fits"]:
            a = Assoc(f["j"], tracks[ti], end_order(P, f["j"], tracks[ti], p))
            e = np.linalg.norm(residuals(P, [a], p).reshape(-1, 3), axis=1)
            if len(e) and np.median(e) < join_err:
                sets[ti] = a
                break
    return {ti: a.j for ti, a in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--workdir")
    ap.add_argument("--tle", action="append", default=[])
    ap.add_argument("--spacetrack", action="store_true")
    ap.add_argument("--free-site", action="store_true")
    ap.add_argument("--clusters", type=int, default=2)
    ap.add_argument("--window", type=float, default=12.0)
    a = ap.parse_args()

    from skyfind import date, pipeline, solve, tles
    wd = pipeline.Workdir(a.clip, a.workdir)
    s = pipeline._setup(wd, stars=None, workers=None, lat=None, log=print,
                        free_site=a.free_site)
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
    fits = wd.read("trackfit.json")
    assert [x["frames"] for x in anchors] == [f["frames"] for f in fits]
    tracks = [solve.segs_from_pixels(cam, t["segs"]) for t in anchors]
    tmp = pipeline._orbit_folder(wd, len({e.norad for e in els}), n, spf,
                                 a.window, print)
    P = None
    try:
        P = solve.build_problem(cam, up, els, t_ref, a.window * 3600, n,
                                lat0=lat, sky=sky, folder=tmp,
                                workers=solve.default_workers(), log=print)
        for c in wd.read("clusters.json")[:a.clusters]:
            tic = time.time()
            old = old_join(P, tracks, fits, c)
            t_old = time.time() - tic
            tic = time.time()
            sol = solve.joint(P, tracks, fits, c, log=None)
            t_new = time.time() - tic
            new = {x["track"]: x["norad"] for x in sol["anchors"]}
            oldn = {ti: int(P.norad[j]) for ti, j in old.items()}
            print(f"cluster at {c['lat']:.2f} {c['lon']:.2f}: old join "
                  f"{len(oldn)} tracks in {t_old:.0f} s; new joint (whole) "
                  f"{len(new)} tracks in {t_new:.0f} s; same "
                  f"{oldn == new}", flush=True)
            if oldn != new:
                print("  only old", sorted(set(oldn.items()) - set(new.items()))
                      [:8], " only new",
                      sorted(set(new.items()) - set(oldn.items()))[:8])
    finally:
        if P is not None:
            P.orbits.close()
        P = None
        import gc
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
