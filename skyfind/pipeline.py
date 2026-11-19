"""The per-clip work folder and the three stages that fill it.

`<clip dir>/<stem>.skyfind/` is the hand-off between stages (and between the
cleaner and the identifier): every stage reads what the previous ones wrote
and writes JSON a person can read.

    clip.json        the clip, the cleaned clip, frame count / size / fps
    track.npz        the sky's motion frame to frame  (trails)
    fgmask.png       camera-fixed foreground (land, trees), 255 = foreground
    cleaned.mp4      the clip with the trails taken out (trails)
    segments.json    trail segments per frame        (trails)
    tracks.json      segments linked across frames   (trails)
    stars.npz        star detections, sampled frames (solve)
    camera.json      lens, orientation, sky rotation, usable frames (solve)
    horizon.json     up vector, rough latitude       (solve)
    date.json        planets -> nominal time of frame 0 (solve)
    trackfit.json    per-track fits                  (solve)
    clusters.json    where distinct objects agree    (solve)
    solution.json    site, clock, exposure, IDs, null rate (solve)

The image engine (sky track, foreground mask, cleaner) is the
remove-satellites library, used as a library only: nothing a user runs or
sees depends on it, and everything skyfind makes stays in the work folder —
it never reads or writes files next to the clip.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from . import (calibrate, catalog, date, horizon, motion, par, platesolve,
               shadow,
               solve, tles, trails)
from . import camera as camera_mod
from .camera import Camera
from .orbits import NoRoom

NEUTRAL = 0.3                  # max chroma of a satellite (planes: coloured)
WHITE_STARS = 20               # stars it takes to say what white is
SPARSE_ANCHORS = 10            # fewer 3-frame tracks: 2-frame ones anchor too
MIN_LEN = 40.0                 # px: shorter segments are too noisy to use
# a mount that follows the stars (`_tracking`): the stars' motion below which
# the frames can't be told apart by it (px, 90th percentile of the steps);
# no satellite crosses the sky faster (deg/s: the ISS overhead is 1.2); and
# the rates most trails move at, which bound the frame interval
STILL_PX = 0.5
SAT_MAX_RATE = 3.0
TRACKED_RATES = (0.08, 1.6)
# slow movers (`_slow`): high objects creeping across (spent stages near the
# top of transfer orbits, satellites a few thousand km up), named by where
# their track lies, frame by frame: arcsec/s against the stars (under the
# first: captions, lamps, geostationary drift), frames, median miss (deg)
SLOW_RATE = (30.0, 900.0)
SLOW_FRAMES = 5
SLOW_TOL = 0.25
GEO_FRAMES = 5         # frames a track must last to time a clip as a geosat
MIN_AGREE = 3                  # independent objects agreeing to call it a site
TIE_CANDIDATES = 6             # sites tried in full when none stands out
TIE_MARGIN = 1.5               # the winner's confirmed objects over the next's
# constellations flown as trains: a plane of evenly spaced satellites fits a
# track at almost any clock (one is always about there), so they fix the
# latitude and the sky's orientation but hardly the time
TRAINS = ("STARLINK", "ONEWEB", "QIANFAN", "KUIPER", "GUOWANG")
# 4K night timelapses need a sensor clean at high ISO (full-frame DSLRs from
# about 2008): the Moon's earlier nights aren't offered for them
SHARP_NIGHT_YEAR = 2008


class Workdir:
    def __init__(self, clip, folder=None):
        self.clip = Path(clip)
        self.dir = Path(folder) if folder else \
            self.clip.with_name(self.clip.stem + ".skyfind")
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def read(self, name: str):
        return json.loads(self.path(name).read_text())

    def write(self, name: str, obj) -> Path:
        p = self.path(name)
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
        tmp.replace(p)
        return p

    @property
    def info(self) -> dict:
        info = self.read("clip.json")
        # the cleaned copy lives in the work folder (older folders recorded
        # an absolute path: a moved folder took its copy with it)
        c = Path(info["cleaned"])
        if not c.is_absolute() or not c.exists() and self.has(c.name):
            info["cleaned"] = str(self.path(c.name))
        return info

    def track(self):
        """The sky's motion frame to frame, tracked once and kept here."""
        from remove_satellites.core import track
        tr, cached = track.load_or_estimate(self.clip,
                                            cache=self.path("track.npz"))
        return tr, cached

    def fgmask(self) -> np.ndarray:
        m = cv2.imread(str(self.path("fgmask.png")), cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8)

    def camera(self) -> Camera:
        return Camera.from_dict(self.read("camera.json"))

    def cleaned_frame(self, k: int) -> np.ndarray:
        """Frame k with the trails taken out: from the cleaned copy, or
        made now from the median when there is none (trails taken from the
        median itself: SKYFIND_EXACT_RESIDUAL)."""
        from remove_satellites.core import video
        info = self.info
        if Path(info["cleaned"]).exists():
            return video.read_frame(info["cleaned"], k)
        from remove_satellites.core import pipeline as rs
        tr, _ = self.track()
        return rs.Cleaner(self.clip, tr, radius=info.get("radius", 10),
                          fg_mask=self.fgmask()).frame(k)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _workers(n: int | None) -> int:
    return n or max(1, min(16, (os.cpu_count() or 2) // 2))


# ====================================================================== trails
def _usable_cleaned(path: Path, n_frames: int) -> bool:
    from remove_satellites.core import video
    try:
        info = video.probe(path)
        video.read_frame(path, n_frames - 1)
    except (FileNotFoundError, IndexError):
        return False
    return info.frame_count == n_frames


def _thresh() -> float:
    """The residual level a trail pixel must pass (experimental override:
    SKYFIND_TRAIL_THRESH, for scripts/ab_residual.py)."""
    return float(os.environ.get("SKYFIND_TRAIL_THRESH", 18.0))


def _trail_chunk(args):
    clip, cleaned, fgmask_path, a, b = args
    from remove_satellites.core import video
    fg = (cv2.imread(fgmask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    ex = trails.Extractor(fg, fg.shape, thresh=_thresh())
    out = {}
    for k, (o, c) in enumerate(zip(video.iter_frames(clip, a, b),
                                   video.iter_frames(cleaned, a, b)), a):
        out[k] = ex(o, c)
    return out


TURNING_PROCS = 8          # chunks side by side (each ~0.5 GB)


def _trail_chunk_turning(args):
    """Frames a..b-1's segments for a turning camera: against the median
    with each frame's own land left out (`motion.FrameLand`,
    `motion.turning_frames`), the pieces of a trail the land split
    rejoined (`trails.join_pieces`)."""
    clip, tr, radius, cam, fgmask_path, a, b, threads = args
    fg = (cv2.imread(fgmask_path, cv2.IMREAD_GRAYSCALE) > 127
          ).astype(np.uint8)
    land = motion.FrameLand(clip, Camera.from_dict(cam), tr, fg)
    out = {}
    for k, o, med, land_k in motion.turning_frames(clip, tr, land, radius,
                                                   a, b, threads=threads):
        ex = trails.Extractor(land_k, land_k.shape, thresh=_thresh())
        out[k] = trails.join_pieces(ex(o, med), land_k)
        tick = _TICK.get("fn")
        if tick:
            tick()
    return out


# frames done, when the chunks are threads of this process (the phone):
# called by each as it goes, so the log can say how far along it is
_TICK: dict = {}


def _trail_chunk_exact(args):
    """Frames a..b-1's segments against the median itself, as the cleaner
    makes it: no encoded copy in between (`SKYFIND_EXACT_RESIDUAL`)."""
    clip, tr, radius, fgmask_path, a, b, gpu, threads = args
    from remove_satellites.core import pipeline as rs
    fg = (cv2.imread(fgmask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    c = rs.Cleaner(clip, tr, radius=radius, fg_mask=fg)
    ex = trails.Extractor(fg, fg.shape, thresh=_thresh())
    out = {}
    for k, o, cl in c.frames(a, b, gpu=gpu, threads=threads):
        out[k] = ex(o, cl)
        tick = _TICK.get("fn")
        if tick:
            tick()
    return out


def run_trails(wd: Workdir, *, radius: int = 10, workers: int | None = None,
               reclean: bool = False, turning: Camera | None = None,
               log=print) -> dict:
    """Track the sky, mask the land, take the trails out (`cleaned.mp4`)
    and extract them as segments. What's already in the work folder is
    reused; `reclean` starts over (after an engine update, say).
    `turning`: the camera of a turning clip (`_reclean`): the segments
    are found against each frame's own land."""
    from remove_satellites.core import foreground, track, video

    info = video.probe(wd.clip)
    n = info.frame_count
    log(f"{wd.clip.name}: {n} frames {info.width}x{info.height}")
    if reclean:
        for name in ("track.npz", "fgmask.png", "cleaned.mp4"):
            if wd.has(name):
                wd.path(name).unlink()
        log("starting over: tracking and cleaning again")
    log("sky motion...")
    tr, cached = wd.track()
    log("  " + ("from the work folder" if cached else "tracked") + ": "
        + track.summary(tr, info.width, info.height))
    if not wd.has("fgmask.png"):
        log("foreground mask...")
        fg = foreground.static_mask(wd.clip, tr)
        cv2.imwrite(str(wd.path("fgmask.png")), fg * 255)
    fg = wd.fgmask()
    log(f"  {100 * fg.mean():.1f}% of the frame is foreground")

    from remove_satellites.core import pipeline as rs
    if os.environ.get("SKYFIND_GPU") == "0":           # the cleaner's too
        os.environ["REMOVE_SATELLITES_GPU"] = "0"
    cleaned = wd.path("cleaned.mp4")
    exact = os.environ.get("SKYFIND_EXACT_RESIDUAL") == "1"
    if exact:
        pass                  # the median's frames as needed: no copy made
    elif _usable_cleaned(cleaned, n):
        log("trails taken out: from the work folder")
    else:
        log(f"taking the trails out (median of {2 * radius + 1} frames "
            "along the sky's motion)...")
        rs.Cleaner(wd.clip, tr, radius=radius, fg_mask=fg).run(
            cleaned, workers=workers)
    if not exact and not _usable_cleaned(cleaned, n):
        raise RuntimeError(f"cleaning {wd.clip.name} failed: {cleaned} "
                           "doesn't match it")
    clock = tr.clock.tolist() if getattr(tr, "mult", None) is not None \
        else None
    mot = [tr.motion_px(k, info.width, info.height)
           for k in range(len(tr.steps))]
    if clock and np.percentile(mot, 90) < STILL_PX:
        # a mount following the stars: steps that don't move aren't
        # repeated frames, none of them moves
        log("  the stars stand still in the frame (a tracking mount): every "
            "frame taken as a capture")
        clock = None
    keep, clock = trails.captures(clock) if clock else (None, None)
    if keep is not None:
        log(f"  re-timed clip: its {n} frames show {len(keep)} captures "
            "(repeated or blended frames); trails from one frame per "
            "capture")
    elif clock:
        log(f"  re-timed clip: {clock[-1] + 1 - n:.0f} dropped frames; "
            "frame k is capture clock[k]")
    wd.write("clip.json", dict(clip=str(wd.clip), cleaned=cleaned.name,
                               n_frames=n, width=info.width,
                               height=info.height, fps=info.fps,
                               clock=clock, radius=radius))

    tic = time.time()
    segs: dict[int, list] = {}
    if turning is not None:
        # each frame's own land, not the union of where it went, and the
        # trails the land split rejoined
        log("trail segments (each frame's own land left out: the camera "
            "turns)...")
        if par.processes():
            procs = min(_workers(workers), TURNING_PROCS)
            threads = max(1, (os.cpu_count() or 2) // (2 * procs))
        else:
            procs, threads = 1, os.cpu_count() or 2
        bounds = np.linspace(0, n, procs + 1).astype(int)
        jobs = [(str(wd.clip), tr, radius, turning.to_dict(),
                 str(wd.path("fgmask.png")), int(a), int(b), threads)
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with par.executor(len(jobs)) as ex:
            for part in ex.map(_trail_chunk_turning, jobs):
                segs.update(part)
    elif exact:
        # experimental (scripts/ab_residual.py): against the median as
        # made, split as the cleaner splits a pass over the clip
        log("trail segments (original minus the median itself)...")
        c = rs.Cleaner(wd.clip, tr, radius=radius, fg_mask=fg)
        w, procs, threads, gpu = c.plan(workers)
        if not par.processes():
            # threads (the phone): a few chunks side by side, each holding
            # its window, its warped copies and their stack (~1 GB in all)
            window = 3 * (2 * radius + 1) * info.width * info.height * 3
            cores = os.cpu_count() or 2
            w = procs = max(1, min(3, cores // 2, int(1.0e9 // window)))
            threads = max(1, cores // w)
        bounds = np.linspace(0, n, w + 1).astype(int)
        jobs = [(str(wd.clip), tr, radius, str(wd.path("fgmask.png")),
                 int(a), int(b), gpu, threads)
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        if not par.processes():
            import threading
            lock, done = threading.Lock(), [0]

            def tick():
                with lock:
                    done[0] += 1
                    if done[0] % 50 == 0:
                        log(f"  {done[0]}/{n} frames")
            _TICK["fn"] = tick
        try:
            with par.executor(procs) as ex:
                for part in ex.map(_trail_chunk_exact, jobs):
                    segs.update(part)
        finally:
            _TICK.pop("fn", None)
    else:
        log("trail segments (original minus cleaned)...")
        w = _workers(workers)
        bounds = np.linspace(0, n, w + 1).astype(int)
        jobs = [(str(wd.clip), str(cleaned), str(wd.path("fgmask.png")),
                 int(a), int(b))
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with par.executor(len(jobs)) as ex:
            for part in ex.map(_trail_chunk, jobs):
                segs.update(part)
    if keep is not None:                  # repeats and blends: left out
        segs = {k: segs[k] for k in keep}
    tracks = trails.link(segs)
    wd.write("segments.json", {str(k): v for k, v in sorted(segs.items())})
    wd.write("tracks.json", tracks)
    nseg = sum(len(v) for v in segs.values())
    log(f"  {nseg} segments, {len(tracks)} tracks "
        f"({sum(len(t['frames']) >= 3 for t in tracks)} over 3+ frames) in "
        f"{time.time() - tic:.0f} s")
    return dict(segments=nseg, tracks=len(tracks))


# ====================================================================== solve
def _read_frame(path, k):
    from remove_satellites.core import video
    return video.read_frame(path, k)


def _detect_one(args):
    path, k, mask_path = args
    sky = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) <= 127).astype(np.uint8)
    return k, calibrate.detect(_read_frame(path, k), sky)


def stars_and_camera(wd: Workdir, stars: catalog.Stars, *, samples: int = 40,
                     workers: int | None = None, log=print) -> Camera:
    info = wd.info
    n, W, H = info["n_frames"], info["width"], info["height"]
    sky = 1 - wd.fgmask()
    frames = sorted(set(np.linspace(0, n - 1, samples).astype(int).tolist()))
    if wd.has("stars.npz"):
        z = np.load(wd.path("stars.npz"))
        dets = {int(k[1:]): z[k] for k in z.files}
    else:
        log(f"stars in {len(frames)} sampled frames...")
        with par.executor(_workers(workers)) as ex:
            dets = dict(ex.map(_detect_one, [(str(wd.clip), k,
                                              str(wd.path("fgmask.png")))
                                             for k in frames]))
        np.savez(wd.path("stars.npz"), **{f"k{k}": d for k, d in dets.items()})
    before = int(np.median([len(d) for d in dets.values()]))
    raw = dets
    dets = _moves_with_sky(wd, dets)
    after = int(np.median([len(d) for d in dets.values()]))
    if after < before:
        log(f"  kept what moves with the sky: {after} of {before} "
            "detections per frame (median)")
    counts = {k: len(d) for k, d in dets.items()}
    log(f"  {int(np.median(list(counts.values())))} stars per frame (median)")

    # two plate solves a third of the clip either side of the middle
    order = sorted(frames, key=lambda k: abs(k - n // 3))
    a = b = None
    for k in order:
        if counts[k] < 30:
            continue
        a = platesolve.solve(dets[k], stars, W, H, frame=k, log=log)
        if a:
            break
    if a is None:
        raise RuntimeError("no frame plate-solved: is this a star field?")
    for k in sorted(frames, key=lambda k: abs(k - (2 * n) // 3)):
        if abs(k - a.frame) < n // 6 or counts[k] < 30:
            continue
        b = platesolve.solve(dets[k], stars, W, H, frame=k, log=log)
        if b:
            break
    if b is None:
        b = _carry(wd, a, frames, dets, n, log)
    if b is None:
        raise RuntimeError("only one frame plate-solved; can't measure the "
                           "sky's rotation")
    a, b = sorted([a, b], key=lambda s: s.frame)
    log(f"  plate solves: frame {a.frame} ({a.score} stars), frame {b.frame} "
        f"({b.score} stars), f ~ {a.scale:.0f} px/rad")
    k0 = (a.frame + b.frame) // 2
    cam = calibrate.initial_camera(a, b, W, H, k0, clock=info.get("clock"))
    log(f"camera fit (sky turns {cam.omega:.4f} deg/frame = "
        f"{cam.seconds_per_frame:.2f} s between frames)...")
    cam, stats = calibrate.refine(cam, dets, stars, sky_mask=sky,
                                  near=[a.frame, b.frame], log=log)
    cam, stats, clock = _check_gaps(cam, stats, info.get("clock"), a, b, W, H,
                                    k0, dets, stars, sky, log)
    tracked = _tracking(wd, cam, log)
    sols = None if tracked else _camera_turns(cam, a, raw, stars, W, H, log)
    if sols is not None:
        return _moving_camera(wd, a, raw, stars, sky, frames, log, sols=sols)
    if clock != info.get("clock"):
        wd.write("clip.json", dict(wd.read("clip.json"), clock=clock))
    # how much sky each frame shows: judged on everything detected (at dawn
    # the next sample is brighter, so the moves-with-sky test would lose
    # real stars and end the usable range early)
    frac = calibrate.match_fraction(cam, raw, stars, sky_mask=sky)
    usable = calibrate.usable_range(frac)
    yr, _ = date.pole_year(cam.axis) if stats["axis_fitted"] else (None, None)
    cam.meta.update(stats=stats, usable=list(usable),
                    match_fraction={str(k): v for k, v in frac.items()},
                    pole_year=yr)
    if tracked:
        # as the model has a turning camera, at a first guess of the
        # interval (the solve finds it: `_interval_search`); this year's
        # pole (it moves 0.006 deg a year: nothing to a site's turn)
        pole = date.data.timescale().utc(_this_year(), 7, 1).M[2]
        cam = cam.tracking(tracked["interval"], n, pole)
        cam.meta["tracked"] = tracked
    wd.write("camera.json", cam.to_dict())
    log(f"  f {cam.f:.1f} px/rad, k1 {cam.p[7]:+.4f}, k2 {cam.p[8]:+.4f}; "
        + ("on a tracking mount" if tracked else
           f"{cam.seconds_per_frame:.3f} s/frame")
        + f"; usable frames {usable[0]}-{usable[1]}")
    return cam


def _tracking(wd: Workdir, cam: Camera, log=print) -> dict | None:
    """Whether the camera rode a mount that follows the stars. The fit then
    finds the sky all but still, and its "interval" (the sky's turn at the
    sidereal rate) comes out a fraction of a second, or negative; taken at
    its word, the trails would cross the sky faster than any satellite can
    (`SAT_MAX_RATE`). A tripod's clip shot in real time looks the same to
    the stars, but its satellites creep. -> dict(step_deg: the trails'
    median travel per frame, interval: a first guess, interval_min/max:
    what the rates satellites move at allow), or None."""
    if not wd.has("tracks.json"):
        return None
    steps, east = [], []
    for t in wd.read("tracks.json"):
        if len(t["frames"]) < 3:
            continue
        ks = np.array(t["frames"], float)
        mid = np.array([[(s["x1"] + s["x2"]) / 2, (s["y1"] + s["y2"]) / 2]
                        for s in t["segs"]])
        v = cam.unproject(mid, ks)
        ds = np.maximum(np.diff(cam.s(ks)), 1e-6)
        ang = np.degrees(np.arccos(np.clip(np.sum(v[1:] * v[:-1], 1), -1, 1)))
        steps.append(float(np.median(ang / ds)))
        ra = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))
        dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        dra = float(np.median(np.diff(ra) / ds))
        ddec = float(np.median(np.diff(dec) / ds))
        if len(ks) >= GEO_FRAMES and dra > 0 and abs(ddec) < 0.05 * dra:
            east.append(dra)
    if len(steps) < 3:
        return None
    step, spf = float(np.median(steps)), cam.seconds_per_frame
    if spf > 0 and step / spf <= SAT_MAX_RATE:
        return None
    # what's fixed to the Earth times the clip itself: geostationary
    # satellites, and lights on any land the mount sweeps past, slide east
    # along their declination at the sidereal rate exactly, however far
    # off they are
    east = np.array(east)
    if len(east) >= 3:
        mid = float(np.median(east))
        same = east[np.abs(east / mid - 1) < 0.08]
        if len(same) >= 3 and len(same) >= 0.6 * len(east):
            sec = float(np.median(same)) / camera_mod.SIDEREAL_DEG_PER_DAY \
                * 86400
            log(f"  a tracking mount: the stars stand still; {len(same)} "
                "trails slide east along their declination together "
                "(fixed to the Earth: geostationary satellites, or lights "
                f"on the land): {sec:.1f} s between frames")
            return dict(step_deg=step, interval=sec, interval_min=0.9 * sec,
                        interval_max=1.1 * sec, star_interval=spf,
                        geostationary=int(len(same)))
    lo, hi = step / TRACKED_RATES[1], step / TRACKED_RATES[0]
    log(f"  a tracking mount: the stars stand still (the fit's interval, "
        f"{spf:.2f} s, would have the trails cross {step:.2f} deg a frame); "
        f"the interval comes from the satellites, {lo:.1f} to {hi:.1f} s")
    return dict(step_deg=step, interval=step / 0.5, interval_min=lo,
                interval_max=hi, star_interval=spf)


def _slow(wd: Workdir, cam: Camera, P, sol: dict, usable, ids, rows,
          log=print):
    """The slow movers the identification passes over, named by position.
    An object creeping a few pixels a frame leaves segments under
    `MIN_LEN`, lights one spot frame after frame (clutter, by that test),
    and its trail is mostly its own blur, longer than its travel, so its
    ends don't time an exposure. But its track is long: the object that
    sits within `SLOW_TOL` of it at every frame's mid-exposure, at the
    solved site and clock, is it (the next best half again as far, or
    it's left unnamed). -> (their segments as rows, their IDs: `how`
    "position", no end timing)."""
    if not wd.has("tracks.json"):
        return [], []
    named = {(d["k"], round(rows[d["seg"]]["x1"], 1),
              round(rows[d["seg"]]["y1"], 1)) for d in ids}
    lat, dlst, T = sol["slat"], sol["dlst"], sol["T"]
    h = sol.get("height_km", P.h0)
    J = np.arange(P.orbits.n_objects)
    out_rows, out_ids, names = [], [], []
    for t in wd.read("tracks.json"):
        ks = np.array(t["frames"], float)
        if len(ks) < SLOW_FRAMES or ks[0] < usable[0] or ks[-1] > usable[1]:
            continue
        if sum((k, round(s["x1"], 1), round(s["y1"], 1)) in named
               for k, s in zip(t["frames"], t["segs"])) > len(ks) / 2:
            continue
        mid = np.array([[(s["x1"] + s["x2"]) / 2, (s["y1"] + s["y2"]) / 2]
                        for s in t["segs"]])
        v = cam.unproject(mid, ks)
        dt = np.maximum(np.diff(cam.s(ks)), 1e-6) * P.spf
        rate = 3600 * float(np.median(np.degrees(np.arccos(np.clip(
            np.sum(v[1:] * v[:-1], 1), -1, 1))) / dt))
        if not SLOW_RATE[0] < rate < SLOW_RATE[1]:
            continue
        m = len(ks) // 2
        with np.errstate(invalid="ignore"):
            d0 = P.dirs(J, np.full(len(J), ks[m]), 0.0, T, lat, dlst, h)
            near = J[np.nan_to_num(d0 @ v[m], nan=-2.0)
                     > math.cos(math.radians(2.0))]
            if not len(near):
                continue
            d = P.dirs(near[:, None], np.broadcast_to(ks, (len(near), len(ks))),
                       0.0, T, lat, dlst, h)
            ang = np.degrees(np.arccos(np.clip(
                np.einsum("jkc,kc->jk", d, v), -1, 1)))
            med = np.nan_to_num(np.nanmedian(ang, 1), nan=99.0)
        order = np.argsort(med)
        best = float(med[order[0]])
        if best > SLOW_TOL or (len(order) > 1
                               and med[order[1]] < 1.5 * best):
            continue
        j = int(near[order[0]])
        names.append(P.names[j])
        for k, s in zip(t["frames"], t["segs"]):
            out_rows.append(dict(s, k=k))
            out_ids.append(dict(
                k=int(k), name=P.names[j], norad=int(P.norad[j]),
                err_deg=round(best, 4), dt=0.0, span=0.0,
                utc=P.utc(T, k)[11:], clipped=False, how="position",
                rate_arcsec_s=round(rate, 1)))
    if names:
        log(f"  slow movers named by position: {len(names)} tracks, "
            f"{len(set(names))} objects ({', '.join(sorted(set(names))[:6])}"
            + (", ..." if len(set(names)) > 6 else "") + ")")
    return out_rows, out_ids


def _tracked(cam: Camera | None) -> bool:
    return cam is not None and bool(cam.meta.get("tracked"))


MOVING_DEG = 1.0     # a fixed camera missing the frames by more: it turns


def _camera_turns(cam: Camera, a, dets, stars, W, H, log=print):
    """Whether the camera turns through the clip (a motion-control rig):
    the stars followed frame by frame from the fixed fit's prediction, and
    the best fixed camera (the sky turning about the pole) fitted to
    their orientations. A tripod's misses are under half a degree (the
    check is pinhole); a rig's, degrees (a dolly clip 6.8, a slowly tilting one
    1.6). -> the followed frames if it turns."""
    def predict(k):
        R = np.stack([camera_mod.rotate(e, cam.axis, float(cam._angle(k)))
                      for e in np.eye(3)], 1)
        return cam.M @ R

    sols = calibrate.track(a, dets, stars, W, H, predict=predict, log=None)
    if len(sols) < 6:
        return None
    miss = calibrate.fixed_misfit(sols, W, H, cam.clock)
    p90 = float(np.percentile(list(miss.values()), 90))
    if p90 <= MOVING_DEG:
        return None
    log(f"  the camera turns: a fixed camera misses the stars' frames by "
        f"{p90:.1f} deg (a motion-control rig)")
    return sols


def _moving_camera(wd: Workdir, seed, dets, stars, sky, frames, log=print,
                   sols=None) -> Camera:
    """A camera that turns through the clip. The stars are followed frame
    to frame from one plate solve, one lens and each frame's orientation
    fitted to them; the land (far enough that the rig's travel doesn't
    shift it) splits each orientation into the Earth's turn and the
    camera's, which fixes the frame interval (`motion`). That interval is
    only good to a percent or two (the rig's travel shifts nearer land a
    little): the solve refines it from the satellites."""
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation
    info = wd.info
    n, W, H = info["n_frames"], info["width"], info["height"]

    def blind(k):
        return platesolve.solve(dets[k], stars, W, H, frame=k,
                                log=lambda *a: None) \
            if len(dets[k]) >= 30 else None

    if sols is None:
        sols = calibrate.track(seed, dets, stars, W, H, blind=blind, log=log)
    if len(sols) < 4:
        raise RuntimeError("the camera moves, and the stars could be "
                           f"followed through only {len(sols)} frames")
    Q, lens, stats = calibrate.follow(sols, dets, stars, W, H, sky_mask=sky,
                                      log=log)
    ks = sorted(Q)
    prof = motion.profiles({k: _read_frame(wd.clip, k) for k in ks},
                           wd.fgmask())
    pairs = motion.land_pairs(lens, prof, ks)
    if len(pairs) < 3:
        raise RuntimeError("the camera moves, and there's too little "
                           "skyline to tell its turn from the Earth's")
    k0 = ks[len(ks) // 2]
    spf, sign, err = motion.interval(Q, pairs, k0, lambda k: k)
    log(f"  frame interval from the land: {spf:.2f} s ({len(pairs)} pairs "
        f"of frames, the land holding still to {err:.2f} deg)")
    G = motion.camera_turns(Q, spf, k0, lambda k: k, sign=sign)
    rv = np.array([Rotation.from_matrix(G[k]).as_rotvec() for k in ks])
    f = np.arange(n, dtype=float)
    turn = PchipInterpolator(ks, rv, axis=0, extrapolate=False)(
        np.clip(f, ks[0], ks[-1]))
    p = np.r_[Rotation.from_matrix(Q[k0]).as_rotvec(),
              sign * spf * camera_mod.SIDEREAL_DEG_PER_DAY / 86400.0,
              lens.p[4:9], 0.0, 0.0, 0.0]
    if info.get("clock") is not None:
        log("  (the sky tracker's dropped frames are ignored: a turning "
            "camera throws it)")
    cam = Camera(p, k0, W, H, turn=turn)
    # the rig jitters between the sampled frames: every frame the stars
    # were followed through gets its own orientation
    from remove_satellites.core import video
    cam, _ = calibrate.refine_turn(
        cam, zip(range(ks[0], ks[-1] + 1),
                 video.iter_frames(wd.clip, ks[0], ks[-1] + 1)),
        stars, sky_mask=sky, log=log)
    # a sky track made from an earlier camera is stale now (`_reclean`)
    info = wd.read("clip.json")
    if info.pop("track_from", None):
        wd.write("clip.json", info)
    frac = calibrate.match_fraction(cam, dets, stars, sky_mask=sky)
    usable = calibrate.usable_range(frac)
    cam.meta.update(stats=stats, usable=list(usable),
                    match_fraction={str(k): v for k, v in frac.items()},
                    pole_year=None,
                    moving=dict(interval_from_land=spf, land_misfit_deg=err,
                                land_pairs=len(pairs),
                                turn_deg=float(np.degrees(np.abs(rv).max()))))
    wd.write("camera.json", cam.to_dict())
    log(f"  f {cam.f:.1f} px/rad, k1 {cam.p[7]:+.4f}, k2 {cam.p[8]:+.4f}; "
        f"the camera turns up to {cam.meta['moving']['turn_deg']:.1f} deg; "
        f"usable frames {usable[0]}-{usable[1]}")
    return cam


GAP_GAIN = 0.05            # a gap stays only if the stars fit 5% better with it


def _check_gaps(cam, stats, clock, a, b, W, H, k0, dets, stars, sky,
                log=print):
    """The tracker's dropped frames, checked against the stars. A re-timed
    clip's real gaps fit the sky's rotation; a phantom one (the tracker
    thrown by moving foreground — a telescope slewing) puts every later
    frame a capture or more ahead. The camera is fitted without any gap,
    then with each put back alone; a gap stays only when the stars fit
    `GAP_GAIN` better with it. (Stars move slowly — a pixel or two over a
    whole clip for a wide lens — but a phantom gap shifts thousands of them
    the same way; satellites, far faster, then agree on one clock.) Clips
    with many gaps are genuinely re-timed and keep them all.
    -> (camera, stats, clock)."""
    if clock is None:
        return cam, stats, None
    steps = np.diff(np.asarray(clock, float))
    gaps = [int(i) for i in np.flatnonzero(steps != 1)]
    if not gaps or len(gaps) > 12:
        return cam, stats, clock

    def fit(keep):
        st = np.ones_like(steps)
        for i in keep:
            st[i] = steps[i]
        clk = np.concatenate([[0.0], np.cumsum(st)]).tolist()
        c0 = calibrate.initial_camera(a, b, W, H, k0,
                                      clock=clk if keep else None)
        c, s = calibrate.refine(c0, dets, stars, sky_mask=sky,
                                near=[a.frame, b.frame], log=lambda *x: None)
        return c, s, (clk if keep else None)

    log(f"  checking the tracker's {len(gaps)} dropped-frame gaps against "
        "the stars...")
    best = fit([])
    kept = []
    for i in gaps:
        trial = fit(kept + [i])
        if trial[1]["median_resid_px"] < \
                (1 - GAP_GAIN) * best[1]["median_resid_px"]:
            best, kept = trial, kept + [i]
    ref = stats["median_resid_px"]
    if best[1]["median_resid_px"] >= (1 - GAP_GAIN) * ref:
        log(f"  all {len(gaps)} gaps fit the stars: kept")
        return cam, stats, clock
    gone = [g for g in gaps if g not in kept]
    log(f"  gaps after frames {', '.join(str(g) for g in gone)} don't fit "
        f"the stars (median residual {ref:.2f} -> "
        f"{best[1]['median_resid_px']:.2f} px without): dropped")
    return best


class Events:
    """`events.jsonl` in the work folder: one JSON line per step of a solve
    (start, each track fitted, solution, IDs), flushed as it happens, so a
    window can follow a solve running anywhere (`skyfind gui`)."""

    def __init__(self, wd: Workdir):
        self.path = wd.path("events.jsonl")
        self.path.write_text("")
        self.t0 = time.time()

    def emit(self, kind: str, **data) -> None:
        line = json.dumps(dict(type=kind, t=round(time.time() - self.t0, 2),
                               **data), default=_json_default)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _moves_with_sky(wd: Workdir, dets: dict[int, np.ndarray],
                    tol: float = 3.0) -> dict[int, np.ndarray]:
    """Keep the detections that move with the sky: a star in one sampled
    frame reappears in the next where the sky track carries it. Whatever the foreground mask missed — floodlit patches, lamps,
    foliage at the skyline, all brighter than the stars they'd crowd out
    of the plate solve — stays put, or wanders, and fails the test."""
    from scipy.spatial import cKDTree
    try:
        tr, _ = wd.track()
    except RuntimeError:
        return dets
    ks = sorted(k for k in dets if len(dets[k]))
    if len(ks) < 2:
        return dets
    trees = {k: cKDTree(dets[k][:, :2]) for k in ks}
    out = dict(dets)
    for i, k in enumerate(ks):
        k2 = ks[i + 1] if i + 1 < len(ks) else ks[i - 1]
        xy = dets[k][:, :2].astype(np.float64).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(xy, tr.between(k, k2)).reshape(-1, 2)
        d, _ = trees[k2].query(moved)
        out[k] = dets[k][d < tol]
    return out


def _reclean(wd: Workdir, cam: Camera, *, workers=None, log=print):
    """A turning camera: the trails taken out again, now that the camera
    is known. The sky tracker took the camera's turn for bad or dropped
    frames (a dolly clip: a night's worth of steps filled in, two phantom gaps),
    so the cleaned copy was misaligned where it mattered; the sky's motion
    comes from the camera model instead (`motion.sky_track`), and the land
    mask covers the land wherever the turn took it (`motion.land_mask`:
    one for the cleaned copy), the trails found against each frame's own
    (`motion.FrameLand`)."""
    log("the camera turns: the sky's motion from the camera model, the "
        "trails taken out again...")
    n = wd.info["n_frames"]
    tr = motion.sky_track(cam, n)
    lo, hi = cam.meta.get("usable", (0, n - 1))
    ks = sorted({int(k) for k in np.linspace(lo, hi, 16)})
    fg = motion.land_mask(wd.clip, cam, tr, ks)
    if wd.has("cleaned.mp4"):             # the segments are overwritten
        wd.path("cleaned.mp4").unlink()
    tr.save(wd.path("track.npz"))
    cv2.imwrite(str(wd.path("fgmask.png")), fg * 255)
    run_trails(wd, workers=workers, turning=cam, log=log)
    wd.write("clip.json", dict(wd.read("clip.json"), track_from="camera",
                               land=LAND_PER_FRAME))


LAND_PER_FRAME = "per-frame"   # clip.json: a turning clip's trails found so
MOVING_SKYLINE = 0.2      # of the ridge's columns on one line: a skyline


def _moving_horizon(wd: Workdir, cam: Camera, log=print) -> dict:
    """Up for a turning camera: the ridge in the reference frame (the
    Earth-fixed pose, where up is defined), found as the land's motion
    was (`motion.profiles`: the foreground mask can't say where the land
    was in one frame), its lowest consistent line fitted as for a fixed
    camera. A mountain ridge is ragged, so a fifth of its columns on the
    line will do; the site search allows a few degrees either way."""
    k0 = cam.k0
    y, ok = motion.profiles({k0: _read_frame(wd.clip, k0)}, wd.fgmask())[k0]
    xs = np.arange(len(y), dtype=float)[ok][::8]
    ys = y[ok][::8]
    if len(xs) >= 20:
        up, inl = horizon.up_vector(cam, xs, ys)
        if inl.sum() >= max(20, MOVING_SKYLINE * len(xs)):
            lat = horizon.latitude(cam, up)
            roll = float(np.degrees(np.arctan2(up[0], -up[1])))
            log(f"  up from the ridge in frame {k0} ({int(inl.sum())} of "
                f"{len(xs)} columns on the line); roll {roll:+.2f} deg; "
                f"latitude {lat:.2f} deg (rough)")
            return dict(up_cam=up.tolist(), lat=lat, roll_deg=roll,
                        source=f"ridge in frame {k0}",
                        n_columns=int(len(xs)), n_inliers=int(inl.sum()),
                        n_verticals=0,
                        skyline=np.column_stack([xs, ys]).round(1).tolist())
    log("  no skyline: the site comes from the satellites alone")
    return dict(up_cam=None, lat=None, roll_deg=None, source="none",
                n_columns=int(len(xs)), n_inliers=0, n_verticals=0,
                skyline=[])


def _carry(wd: Workdir, a, frames, dets, n: int, log=print, tol=5.0):
    """A second plate solution without a second blind solve: carry frame
    a's matched stars along the sky track to another frame
    and re-match them there. For clips where noise lets only one frame
    solve blind (heavy compression, haze)."""
    from scipy.spatial import cKDTree
    tr, _ = wd.track()
    need = max(12, a.score // 2)
    for k in sorted(frames, key=lambda k: -abs(k - a.frame)):
        if abs(k - a.frame) < n // 6 or not len(dets[k]):
            continue
        h = tr.between(a.frame, k)
        xy = cv2.perspectiveTransform(a.xy.reshape(-1, 1, 2), h).reshape(-1, 2)
        d, j = cKDTree(dets[k][:, :2]).query(xy)
        m = d < tol
        if m.sum() >= need:
            log(f"  frame {k}: {int(m.sum())} of frame {a.frame}'s stars "
                "found along the sky track")
            return platesolve.PlateSolution(
                k, dets[k][j[m], :2].astype(float), a.vec[m], a.mag[m],
                [nm for nm, keep in zip(a.names, m) if keep], a.scale,
                int(m.sum()))
    return None


def _setup(wd: Workdir, *, stars, workers, lat, log, free_site=False):
    """Everything a solve needs before it knows when: the camera, which way
    is up (or a free site) and the usable frames. `free_site`: don't trust
    the skyline's up (a ragged treeline can put the latitude degrees off,
    and the pass search only looks a few degrees around it). -> dict."""
    from remove_satellites.core import video

    if not wd.has("segments.json"):
        raise RuntimeError("run `skyfind trails` first")
    info = wd.info
    n = info["n_frames"]
    if stars is None:
        stars = catalog.bright_stars(7.0)      # catalogue epoch: date unknown
    cam = wd.camera() if wd.has("camera.json") else \
        stars_and_camera(wd, stars, workers=workers, log=log)
    if cam.moving and not _tracked(cam) and (
            wd.info.get("track_from") != "camera"
            or wd.info.get("land") != LAND_PER_FRAME):
        _reclean(wd, cam, workers=workers, log=log)
        info = wd.info
    usable = tuple(cam.meta.get("usable", (0, n - 1)))
    _white(wd, usable, log)

    # ---- up, latitude, sidereal time
    if wd.has("horizon.json"):
        hz = wd.read("horizon.json")
    else:
        log("skyline...")
        if _tracked(cam):
            # any land sweeps through the frame as the mount turns: no
            # skyline to read in it (yet)
            log("  a tracking mount: the site comes from the satellites "
                "alone")
            hz = dict(up_cam=None, lat=None, roll_deg=None, source="none",
                      skyline=[])
        elif cam.moving:
            hz = _moving_horizon(wd, cam, log)
        else:
            gray = cv2.cvtColor(wd.cleaned_frame(cam.k0),
                                cv2.COLOR_BGR2GRAY)
            hz = horizon.solve(cam, wd.fgmask(), gray, log=log)
        wd.write("horizon.json", hz)
    # pure sky: no up; the site comes from the satellites alone, and a
    # fixed reference up (the image's) only anchors dlst
    skyline = hz["up_cam"] is not None and not hz.get("free", False)
    free = free_site or not skyline
    if free and hz["up_cam"] is None:
        hz = dict(hz, free=True, up_cam=[0.0, -1.0, 0.0])
        wd.write("horizon.json", hz)
    up = np.array(hz["up_cam"])
    sky = horizon.sky_rays(cam, wd.fgmask() == 0) if free else None
    if free:
        log("  free site" + ("" if skyline else ": no refraction model "
                             "(which way is up isn't known)")
            + ("; --lat ignored" if lat is not None else ""))
        lat = 0.0
    # refraction wants the stars' altitudes only roughly: a skyline's up
    # serves even when it's too far off to fix the site (--free-site).
    # Without it the sky's turn rate soaks up the refraction low down and
    # the frame interval comes out long (a comet clip: 0.3%, 13 s over it).
    if skyline and not cam.refracts:
        log("camera refit with refraction about the skyline's up...")
        z = np.load(wd.path("stars.npz"))
        dets = {int(k[1:]): z[k] for k in z.files}
        dets = {k: v for k, v in dets.items() if usable[0] <= k <= usable[1]}
        cam, _ = calibrate.add_refraction(cam, dets, stars, up,
                                          sky_mask=1 - wd.fgmask(), log=log,
                                          lens_and_rate=not cam.moving)
        wd.write("camera.json", cam.to_dict())
    return dict(cam=cam, up=up, sky=sky, free=free, lat=lat, n=n,
                usable=usable, stars=stars)


def _white(wd: Workdir, usable, log=print) -> list[float]:
    """The clip's white, as its stars show it: (b, g, r) shares. A colour
    grade or a white balance tints everything, and satellites shine by
    sunlight as the stars by their own light (a warm grade:
    blue 0.304, red 0.351; a satellite pair in a dolly clip measured 0.32 against
    grey, just over `NEUTRAL`). So a trail's colour is judged against this
    (`_chroma`). It's white at the stars' scale, which is a trail's: 4:2:0
    video keeps colour at half resolution, so a point a pixel or two
    across keeps little of its own (a 4K clip's stars come out exactly
    grey), and a thin trail loses it the same way. The brightest stars in
    a dozen frames, each its 3x3 peak over its surroundings' median,
    saturated ones left out; grey when fewer than `WHITE_STARS`. Kept in
    white.json."""
    if wd.has("white.json"):
        return wd.read("white.json")["bgr"]
    z = np.load(wd.path("stars.npz"))
    fg = wd.fgmask() if wd.has("fgmask.png") else None
    ks = sorted(k for k in (int(f[1:]) for f in z.files)
                if usable[0] <= k <= usable[1])
    ks = [ks[i] for i in np.unique(np.linspace(0, len(ks) - 1, 12).astype(int))
          ] if ks else []
    shares = []
    for k in ks:
        img = _read_frame(wd.clip, k).astype(np.float32)
        h, w = img.shape[:2]
        d = z[f"k{k}"]
        for x, y, _ in d[np.argsort(-d[:, 2])][:200]:
            xi, yi = int(round(x)), int(round(y))
            if not (6 <= xi < w - 6 and 6 <= yi < h - 6) or \
                    (fg is not None and fg[yi, xi]):
                continue
            patch = img[yi - 6:yi + 7, xi - 6:xi + 7]
            if patch.max() >= 250:
                continue
            bg = np.median(patch.reshape(-1, 3), 0)
            pk = img[yi - 1:yi + 2, xi - 1:xi + 2].reshape(-1, 3).sum(0) \
                - 9 * bg
            if pk.sum() > 60 and (pk > 0).all():
                shares.append(pk / pk.sum())
    white = np.median(shares, 0).tolist() if len(shares) >= WHITE_STARS \
        else [1 / 3] * 3
    wd.write("white.json", dict(bgr=white, stars=len(shares)))
    log(f"  the stars' white: blue {white[0]:.3f} green {white[1]:.3f} red "
        f"{white[2]:.3f} ({len(shares)} stars)")
    return white


def _chroma(s: dict, white) -> float:
    """A segment's colour against the clip's white (`_white`): the channel
    spread over the mean once the stars' tint is divided out
    (`trails.Extractor`'s `chroma` is against grey)."""
    c = np.array([s["b"], s["g"], s["r"]], float) / (3 * np.asarray(white))
    return float((c.max() - c.min()) / max(c.mean(), 1e-6))


def _anchors(wd: Workdir, usable, max_tracks, log=print, cam=None):
    """(neutral segments to identify, tracks over 3+ frames to anchor the
    solve, longest first), less what lies on clutter (`trails.clutter`:
    moving hardware, flicker). A moving camera's clutter is fixed to the
    ground, not the frame: judged there (`_ground_clutter`)."""
    segs = wd.read("segments.json")
    info = wd.info
    white = wd.read("white.json")["bgr"] if wd.has("white.json") else None

    def chroma(s):
        return s["chroma"] if white is None else _chroma(s, white)

    # (a tracking mount's clutter is fixed to its frame, like a tripod's)
    if cam is not None and cam.moving and not _tracked(cam):
        in_clutter = _ground_clutter(segs, cam)
    else:
        grid = trails.clutter(segs, (info["height"], info["width"]))

        def in_clutter(s, k):
            return trails.in_clutter(s, grid)
    rows, n_clut = [], 0
    for k, ss in segs.items():
        k = int(k)
        if usable[0] <= k <= usable[1]:
            for s in ss:
                if chroma(s) <= NEUTRAL and s["len"] >= MIN_LEN:
                    if in_clutter(s, k):
                        n_clut += 1
                    else:
                        rows.append(dict(s, k=k))
    def timed(t):
        """Enough ends at exposure boundaries to time a track: with two,
        the exposure and the clock trade off and every object along the
        line fits (a dolly clip's frames 167-168: a frame edge and the tree's sweep cut
        the outer ends)."""
        return sum((not s["clip_a"]) + (not s["clip_b"])
                   for s in t["segs"]) >= 3

    anchors, t_clut = [], 0
    for t in wd.read("tracks.json"):
        if len(t["frames"]) >= 3 and usable[0] <= t["frames"][0] \
                and t["frames"][-1] <= usable[1] and timed(t) \
                and np.median([chroma(s) for s in t["segs"]]) < NEUTRAL:
            if np.mean([in_clutter(s, k) for k, s in
                        zip(t["frames"], t["segs"])]) >= 0.5:
                t_clut += 1
            else:
                anchors.append(t)
    if len(anchors) < SPARSE_ANCHORS:
        # a long interval (a dolly clip: 44 s) carries most satellites across the
        # view in one or two frames: two-frame tracks anchor too, then
        pairs = [t for t in wd.read("tracks.json")
                 if len(t["frames"]) == 2 and usable[0] <= t["frames"][0]
                 and t["frames"][-1] <= usable[1]
                 and np.median([chroma(s) for s in t["segs"]]) < NEUTRAL
                 and min(s["len"] for s in t["segs"]) >= 30 and timed(t)
                 and not any(in_clutter(s, k) for k, s in
                             zip(t["frames"], t["segs"]))]
        if pairs:
            log(f"  only {len(anchors)} tracks over 3+ frames: {len(pairs)} "
                "two-frame tracks anchor as well")
            anchors += pairs
    anchors.sort(key=lambda t: (-len(t["frames"]),
                                -sum(s["len"] for s in t["segs"])))
    if max_tracks:
        anchors = anchors[:max_tracks]
    if n_clut or t_clut:
        log(f"  clutter (spots lit frame after frame: hardware, flicker): "
            f"left out {n_clut} segments and {t_clut} tracks")
    log(f"{len(rows)} neutral segments; {len(anchors)} tracks to anchor "
        "the solve")
    for t in anchors:
        for k, s in zip(t["frames"], t["segs"]):
            s["k"] = k
    return rows, anchors


def _ground_clutter(segs: dict, cam: Camera):
    """For a moving camera: segments carried into the Earth-fixed pose's
    frame (the land stays put there; the far land exactly, nearer things
    roughly) and clutter found there, plus what hugs the frame's own
    border (the cleaner's warped edges). -> in_clutter(segment, frame)."""
    W, H = cam.width, cam.height
    m = max(W, H) // 2                              # room to move into

    def carry(s, k):
        xy = np.array([[s["x1"], s["y1"]], [s["x2"], s["y2"]]], float)
        r = cam._unturned(cam.px_to_ray(xy), k)
        p, z = cam.ray_to_px(r)
        if (z <= 0.05).any():
            return None
        return dict(s, x1=p[0, 0] + m, y1=p[0, 1] + m, x2=p[1, 0] + m,
                    y2=p[1, 1] + m)

    # the cleaner's warped edges hug the border: one cell deep, undilated
    # (a satellite leaving the frame crosses it, and mustn't count as it)
    frame = trails.clutter(segs, (H, W), grow=0)
    edge = np.zeros_like(frame)
    edge[:1], edge[-1:], edge[:, :1], edge[:, -1:] = True, True, True, True
    frame &= edge
    # the border kept out of the ground's grid: carried there, it sweeps
    # across the sky as the camera turns
    carried = {k: [c for c in (carry(s, int(k)) for s in v
                               if not trails.in_clutter(s, frame, frac=0.7))
                   if c]
               for k, v in segs.items()}
    ground = trails.clutter(carried, (H + 2 * m, W + 2 * m))

    def in_clutter(s, k):
        # along the border (the warp's edges), not across it (a satellite
        # leaving the frame)
        c = carry(s, int(k))
        return trails.in_clutter(s, frame, frac=0.7) or (
            c is not None and trails.in_clutter(c, ground))
    return in_clutter


def run_solve(wd: Workdir, *, tle=None, around: str | None = None,
              window_h: float = 12.0, lat: float | None = None,
              max_tracks: int | None = None, workers: int | None = None,
              keep_orbits: bool = False, stars: catalog.Stars | None = None,
              spacetrack: bool = False, free_site: bool = False,
              interval: float | None = None,
              log=print) -> dict:
    """`stars`: the reference catalogue (default Hipparcos to V 7).
    `interval`: seconds between frames, for a clip from a tracking mount
    (its stars can't give it; without it the satellites are searched for
    it over every interval their rates allow).
    `tle`: element files, later ones preferred for their objects where
    current; `spacetrack`: with Space-Track's for the night too
    (downloaded once with SPACETRACK_USER / SPACETRACK_PASSWORD, cached)."""
    t_start = time.time()
    timing: dict[str, float] = {}           # stage -> seconds, in order
    mark = [t_start]

    def lap(stage: str) -> None:
        now = time.time()
        timing[stage] = round(timing.get(stage, 0.0) + now - mark[0], 1)
        mark[0] = now

    s = _setup(wd, stars=stars, workers=workers, lat=lat, log=log,
               free_site=free_site)
    cam, up, sky, free, lat, n, usable, stars = (
        s[k] for k in ("cam", "up", "sky", "free", "lat", "n", "usable",
                       "stars"))
    wide = None
    if _tracked(cam):
        tr = cam.meta["tracked"]
        if interval:
            cam = cam.with_interval(float(interval))
            cam.meta["tracked"] = dict(tr, interval=float(interval),
                                       given=True)
        elif not tr.get("from_satellites"):
            wide = (tr["interval_min"], tr["interval_max"])
    lap("camera and sky")

    # ---- segments and anchor tracks: without any, nothing to date or fetch
    rows, anchors = _anchors(wd, usable, max_tracks, log, cam)
    if not anchors:
        raise RuntimeError(
            f"no satellite trails to solve with: {len(rows)} trail segments, "
            "none tracked over 3 or more frames (too few satellites crossed, "
            "or compression wiped the trails out)")

    # the worker processes start now and import while the clip is dated and
    # the elements read; they build the orbit table, then help solve
    nw = workers or solve.default_workers()
    pool = solve.pool("skyfind-solve-", nw)
    P = dev = files = ev = None
    try:
        # ---- when (roughly)
        ts = date.data.timescale()
        if around:
            t_ref = ts.from_datetime(_parse_utc(around)).tt
            log(f"nominal frame 0: {around} UTC (given), +-{window_h:g} h")
            wd.write("date.json", dict(source="given", utc=around, tt=t_ref))
        else:
            if wd.has("date.json") and wd.read("date.json").get("source") in \
                    ("planets", "comets", "moon"):
                d = wd.read("date.json")
            else:
                log("dating from the planets...")
                z = np.load(wd.path("stars.npz"))
                dets = {int(k[1:]): z[k] for k in z.files}
                dets = {k: v for k, v in dets.items()
                        if usable[0] <= k <= usable[1]}
                objs = date.planet_candidates(cam, dets, stars, log=log)
                zen = None if free else \
                    cam.cam_to_gcrs(np.tile(up, (n, 1)), np.arange(n))
                cands = _past(date.search(
                    objs, cam.seconds_per_frame, zen,
                    years=(1990, _this_year()), usable=usable,
                    clock=cam.clock, log=log))
                source = "planets"
                cands = _unambiguous(cands, log)
                if not cands:
                    log("dating from comets...")
                    objs, cands = _comet_dates(wd, cam, stars, usable, log)
                    source = "comets"
                    cands = _unambiguous(cands, log)
                if not cands:
                    raise RuntimeError(_undated(wd, cam, usable, log))
                d = dict(source=source, objects=objs, candidates=cands[:10],
                         tt=cands[0]["tt"], utc=cands[0]["utc"])
                wd.write("date.json", d)
                if len(cands) > 1 and \
                        cands[1]["matched"] == cands[0]["matched"]:
                    log(f"  note: {len(cands)} dates fit; using the best "
                        f"({cands[0]['utc']}); --around overrides")
            t_ref = d["tt"]
            log(f"nominal frame 0: {d['utc']} UTC ({d['source']}), search "
                f"+-{window_h:g} h")
        lap("dating")

        # ---- orbital elements
        spf = cam.seconds_per_frame
        first = ts.tt_jd(t_ref - window_h / 24).utc_datetime()
        last = ts.tt_jd(t_ref + (n * spf / 3600 + window_h) / 24
                        ).utc_datetime()
        elements = tles.acquire(first, last, path=tle, spacetrack=spacetrack)
        if isinstance(tle, (str, Path)):
            tle = [tle]
        names = (["Space-Track"] if spacetrack or not tle else []) + \
            [Path(t).name for t in tle or []]
        log(f"{len(elements)} element sets from {', '.join(names)}")
        lap("orbital elements")

        tic = time.time()
        # the orbit table on the GPU too (when there is one) for the whole
        # solve: the pass search, the fits and every identification
        P, files, dev = _build(wd, cam, up, elements, t_ref, window_h, n,
                               lat=lat, sky=sky, pool=pool, log=log)
        track_segs = [solve.segs_from_pixels(cam, t["segs"]) for t in anchors]
        lap("orbit table")
        anchors, track_segs, rows, dark = _sunlit(P, cam, anchors, track_segs,
                                                  rows, log)
        log(f"fitting every candidate object to each track ({nw} "
            + ("processes" if par.processes() else "threads") + ")...")
        ev = Events(wd)
        ev.emit("start", n_tracks=len(anchors), lat0=P.lat0,
                window_h=window_h, title=wd.clip.stem)
        if dark:
            ev.emit("shadow", segs=dark)

        def on_track(i, r):
            ev.emit("track", i=i, frames=r["frames"],
                    segs=[[s["x1"], s["y1"], s["x2"], s["y2"]]
                          for s in anchors[i]["segs"]],
                    fits=[{k: f[k] for k in ("lat", "lon", "T", "obj",
                                             "norad", "med_err")}
                          for f in r["fits"] if f["med_err"] < solve.GOOD_FIT])

        if cam.moving:
            P, agree = _interval_search(P, track_segs, pool, dev, log,
                                        wide=wide)
            if agree >= MIN_AGREE:          # kept only when it's evidence
                cam = P.camera
                cam.meta["moving"] = dict(cam.meta.get("moving", {}),
                                          interval_from_satellites=P.spf)
                if _tracked(cam):
                    cam.meta["tracked"] = dict(cam.meta["tracked"],
                                               interval=P.spf,
                                               from_satellites=True)
                wd.write("camera.json", cam.to_dict())
            elif wide is not None:
                raise RuntimeError(
                    "the frame interval can't be found: the camera followed "
                    "the stars, so only the satellites can time the clip, "
                    f"and at no interval from {wide[0]:.1f} to {wide[1]:.1f} "
                    f"s do {MIN_AGREE} of them agree on a site. Give it "
                    "(--interval SECONDS) if it's known; or the date or the "
                    "elements are wrong")
        fits = solve.fit_tracks(P, track_segs, pool=pool, on_track=on_track,
                                dev=dev, log=log)
        wd.write("trackfit.json", [dict(r, frames=t["frames"])
                                   for r, t in zip(fits, anchors)])
        log(f"  track fits done in {time.time() - tic:.0f} s")
        lap("track fits")
        fam = solve.orbit_families(P)
        clusters = solve.cluster(fits, orbits=fam)
        wd.write("clusters.json", clusters)
        if not clusters:
            raise RuntimeError("no track fitted any catalogued object: wrong "
                               "date, or TLEs that don't cover it?")
        for c in clusters[:4]:
            log(f"  {c['n_objects']} objects agree on lat {c['lat']:.2f} lon "
                f"{c['lon']:.2f} (clock {c['T']:+.0f} s)")
        best = clusters[0]["n_objects"]
        second = clusters[1]["n_objects"] if len(clusters) > 1 else 0
        if best < MIN_AGREE:
            raise RuntimeError(
                f"no site stands out: at best {best} independent objects "
                "agree — that's chance. Wrong date (or night), elements that "
                "don't cover it, or trails that aren't satellites?")
        segs = solve.segs_from_pixels(cam, rows)
        if clear_winner(best, second):
            sol = solve.joint(P, track_segs, fits, clusters[0], pool=pool,
                              dev=dev, log=log)
            if cam.moving:
                P, sol = _refine_interval(wd, P, track_segs, sol, pool, log)
                cam = P.camera
            lap("site")
            ids = solve.identify(P, segs, sol, pool=pool, dev=dev)
            lap("identification")
        else:
            log(f"  {best} objects agree at one site, {second} at another: "
                "identifying every trail at each...")
            sol, ids = _tiebreak(P, track_segs, fits, clusters, segs, pool,
                                 orbits=fam, dev=dev, log=log)
            if cam.moving:
                Q, sol = _refine_interval(wd, P, track_segs, sol, pool, log)
                if Q is not P:
                    P, cam = Q, Q.camera
                    ids = solve.identify(P, segs, sol, pool=pool, dev=dev)
            lap("site (tie-break) and identification")
        ev.emit("solution", **{k: sol[k] for k in (
            "lat", "lon", "sig_lat", "sig_lon", "frame0_utc", "E", "spf",
            "n_objects", "anchors")})
        slow_rows, slow_ids = _slow(wd, cam, P, sol, usable, ids, rows, log)
        for i, d in enumerate(slow_ids):
            d["seg"] = len(rows) + i
        rows, ids = rows + slow_rows, ids + slow_ids
        for d in ids:
            s = rows[d["seg"]]
            d.update(x1=s["x1"], y1=s["y1"], x2=s["x2"], y2=s["y2"])
        objs = solve.objects(ids)
        for o in objs:
            mine = [d for d in ids if d["norad"] == o["norad"]]
            if all(d.get("how") == "position" for d in mine):
                # a place and a pace, no timed ends: sure only when it's
                # held over many frames, closely
                sure = len(mine) >= 10 and o["err"] < 0.15
                o.update(conf="confirmed" if sure else "probable",
                         how="position")
        ev.emit("ids", ids=ids, objects=objs)
        log("chance baseline (clock deliberately wrong)...")
        null = solve.null_rate(P, segs, sol, pool=pool, dev=dev)
        lap("chance check")
        log(f"  solve done in {time.time() - tic:.0f} s")
    except RuntimeError as e:
        # the story so far ends saying why (`replay`: the map never closed in)
        if ev is not None:
            ev.emit("failed", reason=str(e).split(":")[0], message=str(e))
        raise
    finally:
        if dev is not None:
            dev.free()
        pool.close()
        kept = None
        if P is not None:
            if keep_orbits:
                kept = files or _orbit_folder(
                    wd, P.orbits.n_objects, n, cam.seconds_per_frame,
                    window_h, log)
                P.orbits.save(kept, copy=True)
            P.orbits.close()          # unmap before deleting (Windows)
        P = None
        import gc
        gc.collect()
        _discard(pool.folder, files if not kept else None, log=log)
        if kept:
            log(f"orbit table kept in {kept}")
    sol.update(ids=ids, objects=objs, null_rate=null,
               n_segments=len(segs), usable=list(usable), in_shadow=dark,
               second_cluster=clusters[1]["n_objects"] if len(clusters) > 1
               else 0, elapsed_s=round(time.time() - t_start, 1),
               timing=timing)
    wd.write("solution.json", sol)
    ev.emit("done", elapsed_s=sol["elapsed_s"], timing=timing)
    _summary(sol, log)
    return sol


CLOCK_IDS = 3     # tight identifications to set a recorded clip's clock by


def _site_up(cam: Camera, lat: float, lon: float, t_ref: float, n: int,
             T: float = 0.0) -> tuple[np.ndarray, float]:
    """Up in the camera for a site that's known: the zenith of (lat, lon)
    among the stars at each frame's time (frame 0's mid-exposure at TT jd
    `t_ref`, the clock `T` s out), through the camera the stars fitted.
    -> (up, how far the frames' own differ from it at most, deg)."""
    from skyfield.api import wgs84
    ks = np.unique(np.linspace(0, n - 1, 24).astype(int))
    t = date.data.timescale().tt_jd(
        t_ref + (cam.s(ks) * cam.seconds_per_frame + T) / 86400)
    z = (wgs84.latlon(lat, lon, 1000.0).at(t).position.km
         - wgs84.latlon(lat, lon, 0.0).at(t).position.km).T
    ups = cam.gcrs_to_cam(camera_mod.unit(z), ks)
    up = camera_mod.unit(ups.mean(0))
    return up, float(np.degrees(np.arccos(np.clip(ups @ up, -1, 1))).max())


def run_known(wd: Workdir, known: dict, *, tle=None, spacetrack: bool = False,
              workers: int | None = None, stars: catalog.Stars | None = None,
              log=print) -> dict:
    """The trails named for a clip whose camera recorded where and when it
    was (the phone app's own captures: GPS and clock): no search. `known`:
    lat, lon (deg), height_m, frame0_unix (UTC s of frame 0's
    mid-exposure), exposure_s, and accuracy_m (the fix's, for the error
    bars). The stars still give the lens, the pointing and the interval;
    up is the site's zenith, not the skyline's. The clock is then set by
    the satellites themselves when `CLOCK_IDS` trails are named tightly (a
    phone's frame times are good to a few tenths of a second, a satellite
    crosses 0.1 deg in as much), the site turning with it."""
    from datetime import datetime, timezone
    t_start = time.time()
    if not wd.has("segments.json"):
        raise RuntimeError("run `skyfind trails` first")
    n = wd.info["n_frames"]
    lat, lon = float(known["lat"]), float(known["lon"])
    h = float(known.get("height_m") or 0.0) / 1000
    E = float(known["exposure_s"])
    ts = date.data.timescale()
    when = datetime.fromtimestamp(float(known["frame0_unix"]), timezone.utc)
    t_ref = ts.from_datetime(when).tt
    utc = when.strftime("%Y-%m-%d %H:%M:%S")
    log(f"site and time as recorded: {abs(lat):.4f} {'N' if lat >= 0 else 'S'}"
        f" {abs(lon):.4f} {'E' if lon >= 0 else 'W'}, {h * 1000:.0f} m; "
        f"frame 0 {utc} UTC")
    wd.write("capture.json", known)
    wd.write("date.json", dict(source="given", utc=utc, tt=t_ref))
    for old in ("trackfit.json", "clusters.json"):    # an earlier search's
        if wd.has(old):
            wd.path(old).unlink()

    if stars is None:
        stars = catalog.bright_stars(7.0)
    cam = wd.camera() if wd.has("camera.json") else \
        stars_and_camera(wd, stars, workers=workers, log=log)
    up, spread = _site_up(cam, lat, lon, t_ref, n)
    log(f"  up from the site's zenith (the frames' own within {spread:.3f} "
        "deg of it)")
    wd.write("horizon.json", dict(
        up_cam=up.tolist(), lat=lat, source="site",
        roll_deg=float(np.degrees(np.arctan2(up[0], -up[1]))), skyline=[]))
    s = _setup(wd, stars=stars, workers=workers, lat=lat, log=log)
    cam, usable = s["cam"], s["usable"]
    rows, _ = _anchors(wd, usable, None, log, cam)
    if not rows:
        raise RuntimeError("no satellite trails found in the clip")

    nw = workers or solve.default_workers()
    pool = solve.pool("skyfind-solve-", nw)
    P = dev = files = None
    ev = Events(wd)
    try:
        spf = cam.seconds_per_frame
        first = ts.tt_jd(t_ref - 1 / 24).utc_datetime()
        last = ts.tt_jd(t_ref + (n * spf / 3600 + 1) / 24).utc_datetime()
        elements = tles.acquire(first, last, path=tle, spacetrack=spacetrack)
        log(f"{len(elements)} element sets")
        P, files, dev = _build(wd, cam, up, elements, t_ref, 120 / 3600, n,
                               lat=lat, sky=None, pool=pool, log=log)
        ev.emit("start", n_tracks=0, lat0=lat, window_h=0.0,
                title=wd.clip.stem)
        segs = solve.segs_from_pixels(cam, rows)
        lit = shadow.possible(P, segs, shadow.rates(P, segs))
        dark = [[r["k"], r["x1"], r["y1"], r["x2"], r["y2"]]
                for r, k in zip(rows, lit) if not k]
        if dark:
            log(f"  Earth's shadow: {len(dark)} of {len(rows)} segments move "
                "too fast for any satellite the Sun lights there (aircraft, "
                "most likely): left out")
            ev.emit("shadow", segs=dark)
            rows = [r for r, k in zip(rows, lit) if k]
            segs = [g for g, k in zip(segs, lit) if k]

        sol = dict(slat=lat, dlst=0.0, T=0.0, E=E, height_km=h)
        ids = solve.identify(P, segs, sol, pool=pool, dev=dev)
        sig_T, n_clock = 0.5, 0
        for _ in range(3):
            dt = np.array([d["dt"] for d in ids
                           if d["err_deg"] < 0.12 and not d["clipped"]])
            if len(dt) < CLOCK_IDS:
                break
            n_clock = len(dt)
            sig_T = float(1.4826 * np.median(np.abs(dt - np.median(dt)))
                          / math.sqrt(len(dt))) + 0.02
            step = float(np.median(dt))
            if abs(step) < 0.05:
                break
            # the clock moves the satellites; the site, fixed to the Earth,
            # sits that much further round among the stars
            T = sol["T"] + step
            sol.update(T=T, dlst=T * camera_mod.SIDEREAL_DEG_PER_DAY / 86400)
            ids = solve.identify(P, segs, sol, pool=pool, dev=dev)
        if n_clock:
            log(f"  clock from {n_clock} trails: {sol['T']:+.2f} s on the "
                "recorded time")
        else:
            log(f"  fewer than {CLOCK_IDS} trails named tightly: the "
                "recorded clock kept as it is")
        acc = float(known.get("accuracy_m") or 30.0) / 111320
        sol.update(
            lat=lat, lon=lon, height_fitted=True, sig_lat=acc,
            sig_lon=acc / max(math.cos(math.radians(lat)), 0.01),
            sig_height_km=3 * acc * 111.32, height_sys_deg=[0.0, 0.0],
            jackknife_deg=None, sig_T=sig_T, sig_E=0.0, spf=P.spf,
            frame0_utc=P.utc(sol["T"]), frame0_tt=P.t_ref + sol["T"] / 86400,
            median_err_deg=float(np.median([d["err_deg"] for d in ids]))
            if ids else None, n_objects=0, anchors=[], site_from="recorded")
        ev.emit("solution", **{k: sol[k] for k in (
            "lat", "lon", "sig_lat", "sig_lon", "frame0_utc", "E", "spf",
            "n_objects", "anchors")})
        for d in ids:
            r = rows[d["seg"]]
            d.update(x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"])
        objs = solve.objects(ids)
        ev.emit("ids", ids=ids, objects=objs)
        null = solve.null_rate(P, segs, sol, pool=pool, dev=dev)
    except RuntimeError as e:
        ev.emit("failed", reason=str(e).split(":")[0], message=str(e))
        raise
    finally:
        if dev is not None:
            dev.free()
        pool.close()
        if P is not None:
            P.orbits.close()
        P = None
        import gc
        gc.collect()
        _discard(pool.folder, files, log=log)
    sol.update(ids=ids, objects=objs, null_rate=null, n_segments=len(segs),
               usable=list(usable), in_shadow=dark, second_cluster=0,
               elapsed_s=round(time.time() - t_start, 1), timing={})
    wd.write("solution.json", sol)
    ev.emit("done", elapsed_s=sol["elapsed_s"], timing={})
    _summary(sol, log)
    return sol


SEARCH_TRACKS = 16          # the longest anchors fitted per trial interval


def _sunlit(P, cam, anchors, track_segs, rows, log=print):
    """Leave out what moves too fast for any satellite the Sun could light
    there (`shadow`: aircraft, most likely). -> (anchors, track_segs,
    rows, [[k, x1, y1, x2, y2] of the segments left out]). A free site
    isn't tested (no latitude)."""
    if P.free:
        return anchors, track_segs, rows, []
    keep_t = shadow.tracks_possible(P, track_segs)
    segs = solve.segs_from_pixels(cam, rows)
    keep_r = shadow.possible(P, segs, shadow.rates(P, segs))
    dark = [[r["k"], r["x1"], r["y1"], r["x2"], r["y2"]]
            for r, k in zip(rows, keep_r) if not k]
    if dark or not keep_t.all():
        log(f"  Earth's shadow: {len(dark)} of {len(rows)} segments and "
            f"{int((~keep_t).sum())} of {len(anchors)} tracks move too fast "
            "for any satellite the Sun lights there (aircraft, most likely): "
            "left out")
    if not keep_t.any():
        raise RuntimeError(
            "no satellite trails to solve with: every track moves too fast "
            "for a sunlit satellite there (Earth's shadow): aircraft?")
    return ([t for t, k in zip(anchors, keep_t) if k],
            [t for t, k in zip(track_segs, keep_t) if k],
            [r for r, k in zip(rows, keep_r) if k], dark)


def _interval_search(P, track_segs, pool, dev, log=print, wide=None):
    """A moving camera's frame interval. The land gives it only to a few
    percent (the rig's travel shifts nearer land; smooth hills barely pin
    a sideways shift), and satellites need it to a tenth of that: a
    percent is a minute of clock drift over a clip. So the longest anchor
    tracks are fitted at trial intervals across +-`solve.INTERVAL_SPAN`
    in half-percent steps (fine enough for any one track: its few frames
    drift a fraction of a second), then in tenths and fiftieths about the
    best; where the most independent objects agree on one site (then the
    most tracks, then the smallest errors) wins. A trial costs little:
    the stars fix each frame's view, so another interval only moves the
    camera's turn about the pole (`Problem.with_interval`). -> (the
    problem at that interval, attached to the pool; how many independent
    objects agree there).

    `wide` (lowest, highest, s): a tracking mount's clip, with no land's
    interval to start from: every interval the satellites' rates allow is
    tried first, a percent apart (the count of agreeing objects holds over
    about that), then about the best as above."""
    order = sorted(range(len(track_segs)), key=lambda i: -len(track_segs[i]))
    segs = [track_segs[i] for i in order[:SEARCH_TRACKS]]
    fam = solve.orbit_families(P)
    # about the land's interval, whatever an earlier search kept
    s0 = P.camera.meta.get("moving", {}).get("interval_from_land", P.spf)
    seen = {}

    def score(s):
        s = round(float(s), 6)
        if s not in seen:
            Q = P.with_interval(s)
            pool.attach(Q)
            fits = solve.fit_tracks(Q, segs, pool=pool, dev=dev, log=None)
            cl = solve.cluster(fits, orbits=fam)
            if cl:
                c = cl[0]
                err = sum(m["med_err"] for m in c["members"])
                seen[s] = ((c["n_objects"], c["n_tracks"], -err), Q, c)
            else:
                seen[s] = ((0, 0, 0.0), Q, None)
        return seen[s]

    span = solve.INTERVAL_SPAN
    if wide is not None:
        log(f"  the frame interval from the satellites ({len(segs)} tracks, "
            f"{wide[0]:.1f} to {wide[1]:.1f} s)...")
        tic = time.time()
        for s in np.exp(np.arange(math.log(wide[0]), math.log(wide[1]),
                                  0.01)):
            score(s)
        s0 = max(seen, key=lambda s: seen[s][0])
        top = sorted(seen, key=lambda s: seen[s][0], reverse=True)[:5]
        log("    most objects agreeing: " + "  ".join(
            f"{s:.2f} s:{seen[s][0][0]}" for s in top)
            + f" ({len(seen)} intervals in {time.time() - tic:.0f} s)")
        span = 0.01
    else:
        log(f"  the frame interval from the satellites ({len(segs)} tracks, "
            f"{s0:.2f} s +-{span:.0%})...")
    # half a percent is fine enough for any one track; a tenth, then a
    # fiftieth, stop the clock drifting across the clip (a tenth of a
    # percent over 300 frames of 30 s is 9 s)
    for step, lo, hi in ((0.005, -span, span), (0.001, -0.004, 0.004),
                         (0.0002, -0.001, 0.001)):
        base = s0 if step == 0.005 else best
        for f in np.arange(lo, hi + step / 2, step):
            score(base * (1 + f))
        best = max(seen, key=lambda s: seen[s][0])
    line = "  ".join(f"{s:.2f}:{seen[s][0][0]}" for s in sorted(seen)
                     if abs(s / best - 1) < 0.021)
    log(f"    objects agreeing by interval (s): {line}")
    key, Q, c = seen[best]
    log(f"  interval {best:.3f} s"
        + ("" if wide is not None else
           f" ({(best / s0 - 1):+.1%} on the land's)")
        + f": {key[0]} objects agree"
        + (f" at lat {c['lat']:.2f} lon {c['lon']:.2f}" if c else ""))
    pool.attach(Q)
    return Q, key[0]


REFINE_STEPS = ((0.003, 0.0005), (0.0006, 0.0001), (0.00012, 0.00002))


def _refine_interval(wd, P, track_segs, sol, pool, log=print):
    """A moving camera's interval, sharpened on the joint fit. The search
    (`_interval_search`) counts agreeing objects, and that count holds
    over a few hundredths of a percent: seconds of clock drift across the
    clip (the synthetic rig: 0.1%, 16 s at frame 0). The joint fit's own
    anchors, refitted at intervals about it, say where they fit best (in
    tenths of a percent, then hundredths, then fiftieths of those; each
    start carried over, the anchors' middle frame keeping its time and
    site). At the best, the joint fit again, error bars and all.
    -> (problem, solution): the same ones when the search's stands."""
    idx = {int(nr): j for j, nr in enumerate(P.norad)}
    fitted_h = bool(sol.get("height_fitted"))
    p = [sol["slat"], sol["dlst"], sol["T"], sol["E"]] + \
        ([sol["height_km"]] if fitted_h else [])
    A = []
    for a in sol["anchors"]:
        j, segs = idx[int(a["norad"])], track_segs[a["track"]]
        A.append(solve.Assoc(j, segs, solve.end_order(P, j, segs, p)))
    # the ends the joint fit found cut short stay out
    e = np.concatenate([x.end_errors(P, p).ravel() for x in A])
    lim = max(4 * float(np.nanmedian(e)), 0.1)
    for x in A:
        with np.errstate(invalid="ignore"):
            x.cut = x.end_errors(P, p) > lim
    kmid = int(np.median([s.k for x in A for s in x.segs]))
    zra = float(P.site.zenith_ra(kmid))
    seen: dict = {}

    def cost(sp):
        sp = round(float(sp), 6)
        if sp not in seen:
            Q = P.with_interval(sp)
            q = list(p)
            q[2] = p[2] + (P.spf - sp) * float(P.s(kmid))
            q[1] = p[1] + math.degrees(zra - float(Q.site.zenith_ra(kmid)))
            try:
                q, _, errs = solve.fit(Q, A, q, height=fitted_h,
                                       f_scale=0.1,
                                       x_scale=(0.1, 0.1, 5, 2, 0.5))
                seen[sp] = float(np.mean(errs ** 2))
            except (ValueError, np.linalg.LinAlgError):
                seen[sp] = np.inf
        return seen[sp]

    best = P.spf
    for span, step in REFINE_STEPS:
        for f in np.arange(-span, span + step / 2, step):
            cost(best * (1 + f))
        best = min(seen, key=seen.get)
    if abs(best / P.spf - 1) < REFINE_STEPS[-1][1] / 2:
        return P, sol
    log(f"  the interval refined on the {len(A)} anchors: {best:.4f} s "
        f"({best / P.spf - 1:+.3%}; rms end error "
        f"{math.sqrt(seen[round(P.spf, 6)]):.3f} -> "
        f"{math.sqrt(seen[best]):.3f} deg)")
    Q = P.with_interval(best)
    pool.attach(Q)
    q = list(p)
    q[2] = p[2] + (P.spf - best) * float(P.s(kmid))
    q[1] = p[1] + math.degrees(zra - float(Q.site.zenith_ra(kmid)))
    sets = {a["track"]: solve.Assoc(x.j, x.segs, solve.end_order(
        Q, x.j, x.segs, q)) for a, x in zip(sol["anchors"], A)}
    sol2, state = solve.joint_finish(Q, track_segs, (sets, q[:4]), {},
                                     log=None)
    sol2 = solve.joint_errors(Q, sol2, state, pool=pool)
    cam = Q.camera
    cam.meta["moving"] = dict(cam.meta.get("moving", {}),
                              interval_from_satellites=best)
    wd.write("camera.json", cam.to_dict())
    return Q, sol2


def clear_winner(best: int, second: int) -> bool:
    """Does the top cluster (`best` independent objects) go straight to its
    joint fit? Only by `TIE_MARGIN` over the next: Starlink trains agree by
    chance along a whole line of sites and clocks, so a one-object lead
    can be the wrong end of it (a comet clip: 12 vs 11 took the site 70 km off,
    where identifying everything confirms 52 objects, not 81); the rest go
    to `_tiebreak`."""
    return best >= TIE_MARGIN * second


def _tiebreak(P, track_segs, fits, clusters, segs, pool, *, orbits,
              dev=None, log=print):
    """Candidate sites that tie on agreeing objects, settled by identifying
    every trail at each: the right site and clock confirm far more than a
    chance one. The candidates are the top clusters, and the top ones among
    objects not flown in trains (`TRAINS`: those agree by chance along a
    whole line of sites and clocks; lone satellites are what fix the
    clock). Every candidate's joint fit runs at once, one per worker, then
    every candidate's identification together (on the GPU with `dev`).
    -> (solution, ids); raises when no site wins by `TIE_MARGIN`."""
    lone = [dict(r, fits=[f for f in r["fits"]
                          if not f["obj"].upper().startswith(TRAINS)])
            for r in fits]
    cands = clusters[:TIE_CANDIDATES] + [
        c for c in solve.cluster(lone, orbits=orbits)[:TIE_CANDIDATES]
        if c["n_objects"] >= MIN_AGREE]
    orders = solve.join_orders(fits)
    # each site's joint fit in three steps, every site at once in each: its
    # cluster fitted (side by side), the other tracks joining (on the GPU,
    # or side by side), the refits (side by side)
    starts = [st for st in pool.map(solve._job_joint_start,
                                    [(track_segs, c) for c in cands])
              if st is not None]
    if dev is not None:
        from . import gpu
        joined = gpu.join_tracks(P, dev, track_segs, orders, starts)
    else:
        joined = list(pool.map(solve._job_join, [(track_segs, orders, st)
                                                 for st in starts]))
    sols, tops = [], []
    for fitted in pool.map(solve._job_joint_finish,
                           [(track_segs, st, jn)
                            for st, jn in zip(starts, joined)]):
        # unfittable, or the same site found twice (as a cluster and among
        # lone objects)
        sol = fitted[0] if fitted else None
        if sol is None or any(
                abs(sol["lat"] - t["lat"]) < 0.5 and
                abs((sol["lon"] - t["lon"] + 180) % 360 - 180) < 0.5 and
                abs(sol["T"] - t["T"]) < 60 for t in sols):
            continue
        sols.append(sol)
        tops.append(fitted[1])
    trials = []
    for state, sol, ids in zip(tops, sols, solve.identify_all(
            P, segs, sols, pool=pool, dev=dev)):
        objs = solve.objects(ids)
        conf = sum(o["conf"] == "confirmed" for o in objs)
        log(f"    lat {sol['lat']:.2f} lon {sol['lon']:.2f} (frame 0 "
            f"{sol['frame0_utc'][:16]}): {len(objs)} objects, {conf} "
            "confirmed")
        trials.append((conf, sol, ids, state))
    trials.sort(key=lambda t: -t[0])
    if not trials:
        raise RuntimeError("no site stands out: no candidate site could be "
                           "fitted")
    top = trials[0][0]
    nxt = trials[1][0] if len(trials) > 1 else 0
    if top < MIN_AGREE or top < TIE_MARGIN * nxt:
        raise RuntimeError(
            f"no site stands out: with every trail identified the best site "
            f"confirms {top} objects and the next {nxt} — that's chance. "
            "Wrong date (or night), elements that don't cover it, or trails "
            "that aren't satellites?")
    log(f"  the site confirming {top} objects wins (next best {nxt})")
    # its error bars now (the candidates were fitted without them)
    return solve.joint_errors(P, trials[0][1], trials[0][3], pool=pool), \
        trials[0][2]


def find_moon(wd: Workdir, cam: Camera, usable, log=print,
              samples: int = 24) -> dict | None:
    """The Moon in the clip (`moon.find`), kept in moon.json."""
    from remove_satellites.core import video
    from . import moon
    if wd.has("moon.json"):
        return wd.read("moon.json").get("moon")
    ks = sorted({int(k) for k in np.linspace(usable[0], usable[1], samples)})
    frames = {k: video.read_frame(str(wd.clip), k) for k in ks}
    m = moon.find(cam, frames, wd.fgmask() == 0, log=log)
    wd.write("moon.json", dict(moon=m))
    return m


def moon_nights(wd: Workdir, cam: Camera, m: dict, first: str, last: str,
                log=print) -> list[dict]:
    """When the Moon stood where the clip shows it, frame 0 between UTC
    dates `first` and `last` ('YYYY-MM-DD'; `moon.search`)."""
    from datetime import date as Date, timedelta
    from . import moon
    ts = date.data.timescale()
    d0 = Date.fromisoformat(first)
    d1 = Date.fromisoformat(last) + timedelta(days=1)
    return moon.search(m, cam.seconds_per_frame, clock=cam.clock,
                       first_tt=ts.utc(d0.year, d0.month, d0.day).tt,
                       last_tt=ts.utc(d1.year, d1.month, d1.day).tt, log=log)


def _undated(wd, cam, usable, log) -> str:
    """Why the clip couldn't be dated, and what would do it: with the Moon
    in view, when it stood there (moon.json)."""
    msg = ("couldn't date the clip from planets or comets; pass the rough "
           "UTC of the first frame with --around 'YYYY-MM-DD HH:MM'")
    log("looking for the Moon...")
    m = find_moon(wd, cam, usable, log)
    if m is None:
        return msg
    from collections import Counter
    from datetime import date as Date
    today = Date.today()                     # the clip exists: shot by now
    y0, y1 = 1990, today.year
    last = min(Date(y1, 12, 31), today).isoformat()
    nights = moon_nights(wd, cam, m, f"{y0}-01-01", last, log)
    wd.write("moon.json", dict(moon=m, nights=nights))
    # a 4K night timelapse needs a sensor clean at high ISO: earlier nights
    # are listed in moon.json, not in the message
    info = wd.info
    floor = SHARP_NIGHT_YEAR if max(info["width"], info["height"]) >= 3000 \
        else y0
    shown = [n for n in nights if int(n["utc"][:4]) >= floor]
    # the Moon only gets that far north or south in some years (its
    # 18.6-year cycle): say which
    years = Counter(int(n["utc"][:4]) for n in shown)
    by_year = ", ".join(f"{y}: {c}" for y, c in sorted(years.items()))
    since = (f" (a {info['width']}x{info['height']} night timelapse: from "
             f"{floor}; earlier ones in moon.json)" if floor > y0 else "")
    return (f"couldn't date the clip from planets or comets. The Moon is in "
            f"view: it stood there on {len(shown)} nights between "
            f"{max(y0, floor)} and {last[:4]}{since}, once a lunar month, by "
            f"year {by_year} (with its phase in moon.json). Give the rough "
            "UTC of the first frame (--around) or a span to search (skyfind "
            "scan --from --to): the scan then tries only the Moon's "
            "nights.")


def _comet_dates(wd: Workdir, cam: Camera, stars, usable, log=print,
                 samples: int = 24):
    """Comets in the cleaned clip (`comet.candidates`) and the dates their
    positions give (`comet.search`) -> (objects, candidates)."""
    from remove_satellites.core import video
    from . import comet
    ks = sorted({int(k) for k in np.linspace(usable[0], usable[1], samples)})
    frames = {k: wd.cleaned_frame(k) for k in ks}
    objs = comet.candidates(cam, frames, stars, wd.fgmask() == 0, log=log)
    years = (1990, _this_year())          # the clip exists: shot by now
    return objs, _past(comet.search(objs[:6], cam.seconds_per_frame,
                                    years=years, clock=cam.clock, log=log))


def _unambiguous(cands: list[dict], log=print) -> list[dict]:
    """The candidate dates, or none when far-apart dates fit as well as the
    best: one object lined up with a planet or a comet by chance somewhere
    in the decades searched (a glow matched to Mercury gave 1995, 2041,
    2028...), not a date."""
    if not cands:
        return cands
    best = cands[0]
    rivals = [c for c in cands[1:] if c["matched"] == best["matched"] and
              abs(c["tt"] - best["tt"]) > 2.0]
    if not rivals:
        return cands
    if log:
        log(f"  {len(rivals) + 1} far-apart dates fit as well ("
            + ", ".join(c["utc"][:10] for c in [best] + rivals[:3])
            + (", ..." if len(rivals) > 3 else "")
            + "): a chance line-up, not a date")
    return []


def _this_year() -> int:
    from datetime import date as Date
    return Date.today().year


def _past(cands: list[dict]) -> list[dict]:
    """Dates of frame 0 no later than now: the clip exists (a comet's or
    planets' next return fits the sky as well as a past one)."""
    now = date.data.timescale().now().tt + 1.0
    return [c for c in cands if c["tt"] <= now]


def run_scan(wd: Workdir, first: str, last: str, *, tle=None,
             spacetrack: bool = False, max_tracks: int = 20,
             workers: int | None = None, stars: catalog.Stars | None = None,
             then_solve: bool = True, free_site: bool = False,
             moon_lit: float = 0.0, log=print) -> dict:
    """Which night between `first` and `last` (UTC dates, 'YYYY-MM-DD') the
    clip was shot on, when all anyone remembers is roughly when. With the
    Moon in view only its nights are tried (`moon_lit`: those with at
    least this fraction of it lit, when it's plainly lighting the land).

    Orbital elements go stale in days, so one search over weeks would miss
    even the right answer; instead each night is searched on its own
    (+-12 h around 00:00 UTC: every longitude) against elements current
    for it, fitting the `max_tracks` longest tracks. On the right night
    many independent objects agree on one site and time; on the others
    only a handful, by chance. Space-Track's elements (`spacetrack`, or
    no files given) come down a week at a time, once. With `then_solve`
    the best night gets the full solve. -> scan summary."""
    from datetime import date as Date, timedelta
    s = _setup(wd, stars=stars, workers=workers, lat=None, log=log,
               free_site=free_site)
    cam, up, sky, lat, n, usable = (s[k] for k in ("cam", "up", "sky", "lat",
                                                   "n", "usable"))
    _, anchors = _anchors(wd, usable, max_tracks, log, cam)
    track_segs = [solve.segs_from_pixels(cam, t["segs"]) for t in anchors]
    d0, d1 = Date.fromisoformat(first), Date.fromisoformat(last)
    ts = date.data.timescale()
    spf = cam.seconds_per_frame
    files = [tle] if isinstance(tle, (str, Path)) else list(tle or [])
    # with the Moon in view, only the stretches it stood where it's seen
    log("looking for the Moon...")
    m = find_moon(wd, cam, usable, log)
    moon_plan = moon_nights(wd, cam, m, first, last, log) if m else None
    if moon_plan is not None and moon_lit > 0:
        n_all = len(moon_plan)
        moon_plan = [c for c in moon_plan
                     if c.get("lit", 1.0) >= moon_lit - 1e-9]
        log(f"  {len(moon_plan)} of those {n_all} with the Moon at least "
            f"{moon_lit * 100:.0f}% lit")
    if moon_plan is not None:
        if not moon_plan:
            raise RuntimeError(
                f"the Moon in view never stood there between {first} and "
                f"{last}: wrong span?")
        return _scan_moon(wd, s, anchors, track_segs, moon_plan, files,
                          spacetrack, workers, then_solve, free_site, first,
                          last, log)
    nights = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    chunks = []
    if spacetrack or not files:
        from datetime import datetime, timezone
        t0 = datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc)
        t1 = datetime(d1.year, d1.month, d1.day, tzinfo=timezone.utc)
        chunks = tles.acquire_span(t0 - timedelta(hours=12),
                                   t1 + timedelta(hours=12) +
                                   timedelta(seconds=n * spf), log=log)
    mine = tles.read_files(files) if files else []
    nw = workers or solve.default_workers()
    log(f"scanning {len(nights)} nights, {first} to {last}, with the "
        f"{len(anchors)} longest tracks ({nw} processes)")
    out = []
    for i, night in enumerate(nights):
        tic = time.time()
        log(f"scan: night {i + 1} of {len(nights)} ({night})")
        t_ref = ts.utc(night.year, night.month, night.day).tt
        base = sum((tles.read_files([p], first_rank=0) for a, b, p in chunks
                    if a.date() - timedelta(days=2) <= night <= b.date()),
                   tles.Catalog.empty())
        pool = solve.pool("skyfind-scan-", nw)
        P = files = dev = None
        try:
            P, files, dev = _build(wd, cam, up, base + mine, t_ref, 12.0, n,
                                   lat=lat, sky=sky, pool=pool, log=None)
            # each night's Sun: what's too fast for its sunlit sky left out
            lit = track_segs if P.free else \
                [t for t, k in zip(track_segs,
                                   shadow.tracks_possible(P, track_segs)) if k]
            if cam.moving:        # each night finds the interval its own way
                P, _ = _interval_search(P, lit, pool, dev,
                                        log=lambda *a, **k: None)
            fits = solve.fit_tracks(P, lit, pool=pool, dev=dev, log=None)
            cl = solve.cluster(fits, orbits=solve.orbit_families(P))
            top = cl[0] if cl else None
            r = dict(night=str(night), best=top["n_objects"] if top else 0,
                     second=cl[1]["n_objects"] if len(cl) > 1 else 0,
                     lat=top["lat"] if top else None,
                     lon=top["lon"] if top else None,
                     utc=P.utc(top["T"]) if top else None,
                     sunlit_tracks=len(lit), seconds=round(time.time() - tic))
            if cam.moving:
                r["spf"] = round(P.spf, 3)
        finally:
            if dev is not None:
                dev.free()
            pool.close()
            if P is not None:
                P.orbits.close()
            P = None
            import gc
            gc.collect()
            _discard(pool.folder, files, log=log)
        out.append(r)
        log(f"  {night}: {r['best']} objects agree"
            + (f" at lat {r['lat']:.2f} lon {r['lon']:.2f}, frame 0 "
               f"{r['utc']} UTC" if r["best"] else "")
            + f" (next best {r['second']}; {r['seconds']} s)")
        wd.write("scan.json", dict(first=first, last=last,
                                   tracks=len(anchors), nights=out))
    ranked = sorted(out, key=lambda r: -r["best"])
    win = ranked[0]
    runner = ranked[1]["best"] if len(ranked) > 1 else 0
    log(f"best night: {win['night']} — {win['best']} objects agree (the "
        f"next-best night: {runner})")
    summary = dict(first=first, last=last, tracks=len(anchors), nights=out,
                   best=win, runner_up=runner)
    wd.write("scan.json", summary)
    if win["best"] < 4 or win["best"] < runner + 2:
        raise RuntimeError(
            f"no night between {first} and {last} stands out ({win['best']} "
            f"objects agree at best, {runner} on the next night): wrong "
            "month, or elements that don't cover it?")
    if then_solve:
        log(f"full solve on {win['night']}...")
        run_solve(wd, tle=[p for _, _, p in chunks] + files,
                  around=f"{win['night']} 00:00", window_h=12.0,
                  workers=workers, stars=s["stars"], free_site=free_site,
                  log=log)
    return summary


def _scan_moon(wd, s, anchors, track_segs, plan, files, spacetrack,
               workers, then_solve, free_site, first, last, log) -> dict:
    """`run_scan` over the Moon's stretches only: each centred where the
    Moon puts frame 0, searched as far as it allows, against elements
    fetched for it alone."""
    cam, up, sky, lat, n = (s[k] for k in ("cam", "up", "sky", "lat", "n"))
    ts = date.data.timescale()
    spf = cam.seconds_per_frame
    mine = tles.read_files(files) if files else []
    nw = workers or solve.default_workers()
    log(f"scanning the Moon's {len(plan)} nights between {first} and "
        f"{last}, with the {len(anchors)} longest tracks ({nw} processes)")
    out = []
    for i, c in enumerate(plan):
        tic = time.time()
        win = max(4.0, c["span_h"] / 2 + 1.0)
        log(f"scan: {i + 1} of {len(plan)} (frame 0 ~ {c['utc']} UTC "
            f"+-{win:.0f} h, Moon {c['phase']})")
        a = ts.tt_jd(c["tt"] - win / 24).utc_datetime()
        b = ts.tt_jd(c["tt"] + (n * spf / 3600 + win) / 24).utc_datetime()
        base = tles.acquire(a, b) if (spacetrack or not files) else []
        pool = solve.pool("skyfind-scan-", nw)
        P = files = dev = None
        try:
            P, files, dev = _build(wd, cam, up, base + mine, c["tt"], win, n,
                                   lat=lat, sky=sky, pool=pool, log=None)
            fits = solve.fit_tracks(P, track_segs, pool=pool, dev=dev,
                                    log=None)
            cl = solve.cluster(fits, orbits=solve.orbit_families(P))
            top = cl[0] if cl else None
            r = dict(night=c["utc"][:10], around=c["utc"], window_h=win,
                     phase=c["phase"],
                     best=top["n_objects"] if top else 0,
                     second=cl[1]["n_objects"] if len(cl) > 1 else 0,
                     lat=top["lat"] if top else None,
                     lon=top["lon"] if top else None,
                     utc=P.utc(top["T"]) if top else None,
                     seconds=round(time.time() - tic))
        finally:
            if dev is not None:
                dev.free()
            pool.close()
            if P is not None:
                P.orbits.close()
            P = None
            import gc
            gc.collect()
            _discard(pool.folder, files, log=log)
        out.append(r)
        log(f"  {r['around']}: {r['best']} objects agree"
            + (f" at lat {r['lat']:.2f} lon {r['lon']:.2f}, frame 0 "
               f"{r['utc']} UTC" if r["best"] else "")
            + f" (next best {r['second']}; {r['seconds']} s)")
        wd.write("scan.json", dict(first=first, last=last, moon=True,
                                   tracks=len(anchors), nights=out))
    ranked = sorted(out, key=lambda r: -r["best"])
    win_ = ranked[0]
    runner = ranked[1]["best"] if len(ranked) > 1 else 0
    log(f"best night: {win_['around']} — {win_['best']} objects agree (the "
        f"next-best: {runner})")
    summary = dict(first=first, last=last, moon=True, tracks=len(anchors),
                   nights=out, best=win_, runner_up=runner)
    wd.write("scan.json", summary)
    if win_["best"] < 4 or win_["best"] < runner + 2:
        raise RuntimeError(
            f"no night between {first} and {last} stands out ({win_['best']} "
            f"objects agree at best, {runner} on the next): wrong span, or "
            "elements that don't cover it?")
    if then_solve:
        log(f"full solve around {win_['around']}...")
        # the Moon dated it (with the satellites choosing the night): say
        # so, so a report rings it and a solve again keeps the date
        m = find_moon(wd, cam, s["usable"], log=None)
        c = next(c for c in plan if c["utc"] == win_["around"])
        wd.write("date.json", dict(
            source="moon", utc=c["utc"], tt=c["tt"],
            objects=[dict(ra=m["ra"], dec=m["dec"], frames=m["frames"])],
            candidates=[dict(tt=c["tt"], utc=c["utc"], objects=[0],
                             names=["Moon"], phase=c["phase"],
                             nights_tried=len(plan))]))
        run_solve(wd, tle=files or None, window_h=win_["window_h"],
                  workers=workers, stars=s["stars"],
                  spacetrack=spacetrack or not files, free_site=free_site,
                  log=log)
    return summary


def _build(wd: Workdir, cam, up_cam, elements, t_ref: float,
           window_h: float, n: int, *, lat, sky, pool, log=print):
    """The problem, its orbit table in shared memory where there's room
    (else as files on a disk that has it), built by `pool`'s processes and
    handed to them, and sent to the GPU block by block as they finish
    (when there's one with room). -> (problem, the table's folder or None,
    DeviceOrbits or None)."""
    up = None
    if solve.gpu_wanted():
        from . import gpu
        up = gpu.Upload()
    kw = dict(lat0=lat, sky=sky, pool=pool, on_block=up, log=log)
    say = log or (lambda *a, **k: None)
    if pool.folder is None:              # threads: the table in this process
        P = solve.build_problem(cam, up_cam, elements, t_ref,
                                window_h * 3600, n, **dict(kw, pool=None))
        pool.attach(P)
        return P, None, (up.done(P.orbits, say) if up is not None else None)
    try:
        P = solve.build_problem(cam, up_cam, elements, t_ref,
                                window_h * 3600, n, shared=True,
                                folder=pool.folder, **kw)
        files = None
    except NoRoom as err:
        say(f"  no room in memory for the orbit table ({err}); using a disk")
        files = _orbit_folder(wd, tles.n_objects(elements), n,
                              cam.seconds_per_frame, window_h, say)
        if up is not None:
            up = kw["on_block"] = gpu.Upload()     # start again
        P = solve.build_problem(cam, up_cam, elements, t_ref,
                                window_h * 3600, n, folder=files, **kw)
    pool.attach(P)
    return P, files, (up.done(P.orbits, say) if up is not None else None)


def _discard(*folders, log=print) -> None:
    for f in folders:
        if f is None:
            continue
        shutil.rmtree(f, ignore_errors=True)
        if Path(f).exists() and log:
            log(f"  couldn't delete {f}; remove it by hand")


def _orbit_folder(wd: Workdir, n_elements: int, n_frames: int, spf: float,
                  window_h: float, log=print) -> Path:
    """Somewhere with room for the orbit table (positions + velocities,
    float32, 20 s grid; ~3.3 GB for today's catalogue over +-12 h): the
    temp dir if it has plenty, else the work folder next to the clip."""
    n_t = (2 * window_h * 3600 + 1.5 * n_frames * spf + 360) / 20 + 2
    need = int(2 * n_t * n_elements * 3 * 4)
    for base in (Path(tempfile.gettempdir()), wd.dir):
        free = shutil.disk_usage(base).free
        if free > 1.5 * need + (1 << 30):
            return Path(tempfile.mkdtemp(prefix="skyfind-orbits-", dir=base))
        log(f"  {base}: {free / 2**30:.1f} GB free, the orbit table needs "
            f"{need / 2**30:.1f} GB")
    raise RuntimeError("no disk has room for the orbit table: free some "
                       "space, or narrow --window")


def _summary(sol: dict, log=print) -> None:
    ns = "N" if sol["lat"] >= 0 else "S"
    ew = "E" if sol["lon"] >= 0 else "W"
    log(f"\nsite  {abs(sol['lat']):.4f} {ns} +- {sol['sig_lat']:.4f}, "
        f"{abs(sol['lon']):.4f} {ew} +- {sol['sig_lon']:.4f}"
        + (f", height {sol['height_km']:.2f} +- {sol['sig_height_km']:.2f} km"
           if sol.get("height_fitted") else
           f" (height not determined: assumed {sol['height_km']:.0f} +- 1 km,"
           f" included in the error)"))
    log(f"clock frame 0 mid-exposure {sol['frame0_utc']} UTC (+- "
        f"{sol['sig_T']:.2f} s); exposure {sol['E']:.2f} s every "
        f"{sol['spf']:.2f} s")
    objs = sol["objects"]
    conf = sum(o["conf"] == "confirmed" for o in objs)
    null = sol["null_rate"]
    chance = np.mean([v["segments"] for v in null.values()])
    log(f"{len(objs)} objects ({conf} confirmed) on {len(sol['ids'])} of "
        f"{sol['n_segments']} segments; "
        + ("site and time as recorded" if sol.get("site_from") == "recorded"
           else f"{sol['n_objects']} anchored the solve")
        + f"; with the clock wrong ~{chance:.0f} segments match by chance")
    if "elapsed_s" in sol:
        from .report import took
        log(f"solved in {took(sol['elapsed_s'])}: " + ", ".join(
            f"{k} {took(v)}" for k, v in sol.get("timing", {}).items()))


def _parse_utc(s: str):
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"can't read {s!r} as a UTC time (YYYY-MM-DD HH:MM)")
