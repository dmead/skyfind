"""Trails from the compressed cleaned copy (as now) or from the median
itself (`SKYFIND_EXACT_RESIDUAL=1`): which is closer to the truth?

    python scripts/ab_residual.py synth [--crf 23]
    python scripts/ab_residual.py real [NAME ...]

synth: a rendered sky (tests/synth.py) with known trails, site, clock and
exposure, encoded like a camera's clip at `--crf`. Both ways from the same
clip: segment ends against the true ends (half maximum), then both solved
blind against the truth, every ID checked.

real: each work folder copied and its trails redone from the median, then
solved with the same options, against its current answer (no truth there:
how well the trails fit, how many are confirmed, the chance baseline).
Space-Track downloads use the account saved in the workbench.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXACT = "SKYFIND_EXACT_RESIDUAL"


def _local() -> dict:
    """`$SKYFIND_DATA/regress.json` (untracked: the clips are on loan):
    "scratch" (a folder with room) and "ab_residual": {NAME: {"clip",
    "workdir", "options"}}."""
    from skyfind import data
    f = data.data_dir() / "regress.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


OUT = Path(_local().get("scratch", ROOT / "scratch")) / "ab_residual"
# name: (clip, current work folder, solve options)
CASES = {k: (v["clip"], v["workdir"], v.get("options", []))
         for k, v in _local().get("ab_residual", {}).items()}


# ------------------------------------------------------------- matching
def match(truth: list[dict], segs: dict, perp_px: float = 3.0):
    """Each true trail's best detected segment in its frame: both ends
    within `perp_px` of the true line, overlapping it most. -> (pairs
    [(truth, seg)], detected segments matching no trail)."""
    by_k: dict[int, list] = {}
    for t in truth:
        by_k.setdefault(t["k"], []).append(t)
    pairs, used = [], set()
    for k, ts in by_k.items():
        ss = segs.get(str(k), [])
        for t in ts:
            p = np.array([t["x1"], t["y1"]])
            q = np.array([t["x2"], t["y2"]])
            L = float(np.linalg.norm(q - p))
            u = (q - p) / L
            best, best_ov = None, 0.0
            for i, s in enumerate(ss):
                a = np.array([s["x1"], s["y1"]]) - p
                b = np.array([s["x2"], s["y2"]]) - p
                if max(abs(a[0] * u[1] - a[1] * u[0]),
                       abs(b[0] * u[1] - b[1] * u[0])) > perp_px:
                    continue
                ta, tb = sorted((a @ u, b @ u))
                ov = min(tb, L) - max(ta, 0.0)
                if ov > best_ov:
                    best, best_ov = (i, ta, tb), ov
            if best is not None:
                i, ta, tb = best
                used.add((k, i))
                pairs.append((t, dict(ss[i], t_lo=ta, t_hi=tb, L=L)))
    false = sum(1 for k, ss in segs.items() for i in range(len(ss))
                if (int(k), i) not in used and int(k) in by_k) + \
        sum(len(ss) for k, ss in segs.items() if int(k) not in by_k)
    return pairs, false


def end_errors(pairs) -> np.ndarray:
    """Signed along-track errors (px, + = measured long) of the ends the
    exposure made (not cut by the frame edge, the skyline or the shadow)."""
    e = []
    for t, s in pairs:
        if not t["clip_a"]:
            e.append(-s["t_lo"])
        if not t["clip_b"]:
            e.append(s["t_hi"] - s["L"])
    return np.array(e)


def id_check(truth: list[dict], ids: list[dict]):
    """IDs against the truth: (right, wrong, on no true trail)."""
    right = wrong = none = 0
    for d in ids:
        pairs, _ = match([t for t in truth if t["k"] == d["k"]],
                         {str(d["k"]): [d]})
        if not pairs:
            none += 1
        elif any(t["norad"] == d["norad"] for t, _ in pairs):
            right += 1
        else:
            wrong += 1
    return right, wrong, none


def chance(sol: dict) -> float:
    n = sol.get("null_rate") or {}
    return float(np.mean([v["segments"] for v in n.values()])) if n else \
        float("nan")


# ------------------------------------------------------------- synthetic
def synth(crf: int, thresh: float | None = None) -> None:
    sys.path.insert(0, str(ROOT))
    from remove_satellites.core.video import FrameWriter

    from skyfind import pipeline
    from tests.synth import World

    w = World(width=1920, height=1080, f=760.0)
    d = OUT / f"synth-crf{crf}"
    d.mkdir(parents=True, exist_ok=True)
    clip = d / "night.mp4"
    truth = w.segments()
    if not clip.exists():
        print(f"rendering {w.n_frames} frames (crf {crf})...", flush=True)
        by: dict[int, list] = {}
        for s in truth:
            by.setdefault(s["k"], []).append(s)
        with FrameWriter(clip, w.width, w.height, 24, crf=crf) as fw:
            for k in range(w.n_frames):
                fw.write(w.render(k, by.get(k, ())))
    tle = d / "night.3le"
    tle.write_text(w.tle_text, encoding="ascii")
    quiet = lambda *a, **k: None                             # noqa: E731
    A = pipeline.Workdir(clip, d / "copy")
    B = pipeline.Workdir(clip, d / ("median" if thresh is None
                                      else f"median-t{thresh:g}"))
    if not A.has("tracks.json"):
        pipeline.run_trails(A, log=quiet)
    for name in ("track.npz", "fgmask.png", "cleaned.mp4", "clip.json"):
        if not B.has(name):
            shutil.copy2(A.path(name), B.path(name))
    os.environ[EXACT] = "1"
    if thresh is not None:
        os.environ["SKYFIND_TRAIL_THRESH"] = str(thresh)
    try:
        tic = time.time()
        pipeline.run_trails(B, log=quiet)
        print(f"trails from the median itself: {time.time() - tic:.0f} s")
    finally:
        del os.environ[EXACT]
        os.environ.pop("SKYFIND_TRAIL_THRESH", None)

    n_ends = sum((not t["clip_a"]) + (not t["clip_b"]) for t in truth)
    print(f"\nsynthetic sky, crf {crf}: {len(truth)} true trails, "
          f"{n_ends} ends made by the exposure")
    print(f"{'':16} {'found':>6} {'|end err| med':>14} {'p90':>6} "
          f"{'bias':>6} {'false':>6} {'segments':>9}")
    mlabel = "median itself" if thresh is None else f"median, {thresh:g}"
    for label, wd in (("compressed copy", A), (mlabel, B)):
        segs = wd.read("segments.json")
        pairs, false = match(truth, segs)
        e = end_errors(pairs)
        print(f"{label:16} {len(pairs) / len(truth):6.1%} "
              f"{np.median(np.abs(e)):11.2f} px {np.percentile(np.abs(e), 90):6.2f}"
              f" {np.mean(e):+6.2f} {false:6d} "
              f"{sum(len(v) for v in segs.values()):9d}")

    around = (w.t0 + timedelta(hours=3.7)).strftime("%Y-%m-%d %H:%M")
    print("\nsolved blind (truth: site, frame 0, exposure):")
    for label, wd in (("compressed copy", A), (mlabel, B)):
        if wd.has("solution.json") and wd is A:
            sol = wd.read("solution.json")
        else:
                sol = pipeline.run_solve(wd, tle=tle, around=around,
                                     max_tracks=10, stars=w.stars, log=quiet)
        km = math.hypot((sol["lat"] - w.lat) * 111.2,
                        (sol["lon"] - w.lon) * 111.2
                        * math.cos(math.radians(w.lat)))
        dt = (sol["frame0_tt"] - w.t(0).tt) * 86400
        r, wr, no = id_check(truth, sol["ids"])
        conf = sum(o["conf"] == "confirmed" for o in sol["objects"])
        print(f"{label:16} site {km * 1000:5.0f} m off, frame 0 {dt:+.3f} s, "
              f"exposure {sol['E'] - w.exposure:+.3f} s; {len(sol['ids'])} "
              f"IDs ({r} right, {wr} wrong, {no} on no trail), "
              f"{len(sol['objects'])}/{conf} objects; fit "
              f"{sol['median_err_deg']:.4f} deg; chance ~{chance(sol):.1f}")


# ------------------------------------------------------------- real clips
def real(names: list[str], thresh: float | None = None) -> None:
    from skyfind import sources
    _, user = sources.load()
    env = dict(os.environ, **sources.environment(user, sources.password(user)),
               PYTHONIOENCODING="utf-8")
    py = sys.executable
    rows = []
    for name in names:
        clip, cur, opts = CASES[name]
        B = OUT / (f"{name}-median" if thresh is None
                   else f"{name}-median-t{thresh:g}")
        if B.exists():
            shutil.rmtree(B)
        shutil.copytree(cur, B)
        tic = time.time()
        with open(B / "trails.log", "w", encoding="utf-8") as f:
            rc = subprocess.call([py, "-u", "-m", "skyfind", "trails", clip,
                                  "--workdir", str(B)], stdout=f,
                                 stderr=subprocess.STDOUT,
                                 env=dict(env, **{EXACT: "1"}, **(
                                     {} if thresh is None else
                                     {"SKYFIND_TRAIL_THRESH": str(thresh)})))
        t_trails = time.time() - tic
        if rc:
            print(f"{name}: trails failed, see {B / 'trails.log'}")
            continue
        with open(B / "solve.log", "w", encoding="utf-8") as f:
            rc = subprocess.call([py, "-u", "-m", "skyfind", "solve", clip,
                                  "--workdir", str(B), *opts], stdout=f,
                                 stderr=subprocess.STDOUT, env=env)
        if rc:
            print(f"{name}: solve failed, see {B / 'solve.log'}")
            continue
        for label, folder in (("copy", Path(cur)), ("median", B)):
            s = json.loads((folder / "solution.json").read_text())
            segs = json.loads((folder / "segments.json").read_text())
            tracks = json.loads((folder / "tracks.json").read_text())
            rows.append((name, label, s, sum(len(v) for v in segs.values()),
                         sum(len(t["frames"]) >= 3 for t in tracks)))
        a, b = rows[-2][2], rows[-1][2]
        km = math.hypot((a["lat"] - b["lat"]) * 111.2,
                        (a["lon"] - b["lon"]) * 111.2
                        * math.cos(math.radians(a["lat"])))
        same = {o["norad"] for o in a["objects"]} & \
            {o["norad"] for o in b["objects"]}
        print(f"{name}: trails from the median in {t_trails:.0f} s; site "
              f"{km * 1000:.0f} m apart, frame 0 "
              f"{(b['frame0_tt'] - a['frame0_tt']) * 86400:+.2f} s, exposure "
              f"{b['E'] - a['E']:+.2f} s; {len(same)} objects in both",
              flush=True)
    print(f"\n{'':12} {'':7} {'segments':>8} {'tracks3+':>8} {'objects':>8} "
          f"{'confirmed':>9} {'fit deg':>8} {'chance':>7}")
    for name, label, s, nseg, ntr in rows:
        conf = sum(o["conf"] == "confirmed" for o in s["objects"])
        print(f"{name:12} {label:7} {nseg:8d} {ntr:8d} "
              f"{len(s['objects']):8d} {conf:9d} {s['median_err_deg']:8.4f} "
              f"{chance(s):7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("synth", "real"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--thresh", type=float, default=None,
                    help="the trail threshold for the median's residual")
    a = ap.parse_args()
    if a.what == "synth":
        synth(a.crf, a.thresh)
    else:
        real(a.names or list(CASES), a.thresh)


if __name__ == "__main__":
    main()
