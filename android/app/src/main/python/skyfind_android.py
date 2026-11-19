"""The app's side of skyfind: the desktop pipeline (trails, solve, report,
replay: `skyfind run` and `skyfind replay`) with the phone's settings.

- No other processes (`par`): the solve's workers are threads, remove-
  satellites runs in this one (REMOVE_SATELLITES_WORKERS=1).
- No GPU, and no video encoder for a cleaned copy: the trails come from the
  median itself (SKYFIND_EXACT_RESIDUAL), its residual thresholded at 16 —
  the level a copy's encoding made 18 into (scripts/ab_residual.py).
- The orbit table on a 120 s grid (a sixth of the desktop's memory; Hermite
  interpolation keeps LEO positions to metres), propagated by libsgp4c.so
  (`sgp4lib`); a free site's pass search in libskyfind_native.so (the GPU's
  kernels in C++ threads: `native`).
- The replay is an MP4 made by the phone's encoder (`Mp4Writer`).

What goes wrong is said in the app's words (`AppError`, `LoginNeeded`):
its login fields and buttons, not the desktop's options.
"""

from __future__ import annotations

import os
import time


def configure(data_dir: str, lib_dir: str) -> None:
    """Before anything imports numpy: the phone's settings."""
    os.environ.update(
        SKYFIND_DATA=data_dir,
        SKYFIND_PROCESSES="0",
        REMOVE_SATELLITES_WORKERS="1",
        SKYFIND_GPU="0",
        REMOVE_SATELLITES_GPU="0",
        SKYFIND_EXACT_RESIDUAL="1",
        SKYFIND_TRAIL_THRESH="16",
        SKYFIND_GRID_DT="120",
        SKYFIND_SGP4C=os.path.join(lib_dir, "libsgp4c.so"),
        SKYFIND_NATIVE=os.path.join(lib_dir, "libskyfind_native.so"),
        OPENBLAS_NUM_THREADS="2",
    )
    # downloads (Space-Track, fresh comet orbits) check certificates
    # against certifi's list: Android's own aren't where Python looks
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


class AppError(RuntimeError):
    """A failure the app shows as it is (its message is for the phone)."""


class LoginNeeded(AppError):
    """The Space-Track login is missing or wrong: the app points at it."""


def _app_words(e) -> AppError:
    """TLEsNeeded's advice for the phone: the login fields and the
    element-file button, not environment variables and --tle."""
    from skyfind import tles
    if e.reason == "login":
        return LoginNeeded(
            "Naming the satellites needs Space-Track's orbital elements for "
            "the night. Type your Space-Track login in the app (a free "
            f"account: {tles.ST_SIGNUP}) and solve again: the trails are "
            "kept, so it goes straight to the solve. Or sign in on the "
            f"website, save\n{e.url}\nand add that file with Add orbital "
            "elements.")
    if e.reason == "refused":
        return LoginNeeded(
            f"Space-Track refused the sign-in: {e.detail}. Check the user "
            "name and password in the app (they're your Space-Track "
            f"account's: try them at {tles.ST_BASE}).")
    return AppError(str(e))


def _offline(e) -> str:
    return f"couldn't reach Space-Track ({getattr(e, 'reason', e)})"


def check_login(user: str, password: str) -> str:
    """Signs in to Space-Track (nothing downloaded; the password goes to
    Space-Track only): "" when it works, else what's wrong."""
    from skyfind import tles
    try:
        tles.sign_in(user, password)
    except tles.TLEsNeeded as e:
        return str(_app_words(e))
    except OSError as e:                  # urllib's URLError: no network
        return _offline(e).capitalize() + ": is the phone online?"
    return ""


_LIVE: dict = {}


def live_frame(clip: str, work: str, out: str, since_ms: int = 0,
               width: int = 960) -> bool:
    """The solve story so far, drawn as the desktop's live window draws it
    (`skyfind watch`: the still story, its latest step) into the PNG `out`.
    False while there's nothing to draw (the solve hasn't begun: an event
    log from before `since_ms`, the run's start, is an earlier solve's) or
    nothing new since the last call. The backgrounds and the map layers
    are kept between calls: the app asks every few seconds."""
    import cv2
    from skyfind import pipeline, replay
    wd = pipeline.Workdir(clip, work)
    if not wd.has("events.jsonl"):
        return False
    stamp = wd.path("events.jsonl").stat().st_mtime_ns
    if stamp < since_ms * 1_000_000:
        return False
    c = _LIVE.setdefault(work, dict(stamp=None, frames={}, map=None))
    if c["stamp"] == stamp and os.path.exists(out):
        return False
    story = replay.Story.from_events(wd, frames=c["frames"], footage=False)
    if story is None:
        return False
    sc = replay.scenes(story)
    if not sc:
        return False
    r = replay.Renderer(story, width=width)
    if c["map"] is None:
        c["map"] = r.map
    r.map = c["map"]                     # its layers loaded once
    img = r.render(sc[-1])
    cv2.imwrite(out + ".part.png", img)
    os.replace(out + ".part.png", out)
    c["stamp"] = stamp
    return True


def dump_frame(clip: str, out: str, k: int = 0) -> str:
    """A check over adb: frame k of `clip` as the pipeline reads it, to a
    PNG (the phone's decoder: colour order, size)."""
    import cv2
    from remove_satellites.core import video
    f = video.read_frame(clip, k)
    cv2.imwrite(out, f)
    return f"{f.shape} {f.dtype} -> {out}"


def focus_ready() -> None:
    """numpy and OpenCV loaded before the first focus stop needs them."""
    from skyfind import focus  # noqa: F401


def _luma(y, w: int, h: int):
    import numpy as np
    return np.frombuffer(bytes(y), np.uint8).reshape(int(h), int(w))


def focus_sweep(min_focus: float, hyperfocal: float):
    """The focus run for a lens that goes from 0 (the maker's infinity) to
    `min_focus` dioptres: the far end of its travel, a few times past the
    hyperfocal position (closer than that the stars are discs too faint to
    measure)."""
    from skyfind import focus
    hi = min(float(min_focus), max(1.0, 3 * float(hyperfocal)))
    return focus.Sweep(0.0, hi)


def focus_add(sweep, frames, w: int, h: int) -> str:
    """The frames (luma bytes) taken at the sweep's stop; -> what they
    showed, for the screen."""
    s = sweep.add([_luma(y, w, h) for y in frames])
    if s["hfr"] is None:
        return f"lens {s['x']:.3f}: no stars measured"
    return f"lens {s['x']:.3f}: stars {s['hfr']:.2f} px ({s['n']} of them)"


def focus_check(y, w: int, h: int, focused_hfr: float) -> str:
    """While recording: "" while the stars are as small as the focus run
    left them, else a warning."""
    from skyfind import focus
    m = focus.measure(_luma(y, w, h))
    if focus.drifted(m["hfr"], focused_hfr):
        return (f"focus has drifted: stars {m['hfr']:.2f} px, were "
                f"{focused_hfr:.2f}; stop and focus again")
    return ""


def _recorded(capture: str | None):
    """A clip recorded by the app: (where and when for `run_known`, or
    None without a GPS fix or a held exposure; the time for --around)."""
    if not capture or not os.path.exists(capture):
        return None, None
    import json
    from datetime import datetime, timezone
    with open(capture, encoding="utf-8") as f:
        c = json.load(f)
    if "frame0_unix" not in c:
        return None, None
    when = datetime.fromtimestamp(c["frame0_unix"], timezone.utc)
    known = c if all(k in c for k in ("lat", "lon", "exposure_s")) else None
    return known, when.strftime("%Y-%m-%d %H:%M:%S")


def solve(clip: str, work: str, tle_paths, around: str | None,
          free_site: bool, st_user: str | None, st_password: str | None,
          sink, capture: str | None = None) -> list:
    """trails + solve + report + replay for `clip`, everything in `work`.
    `sink`: the app's listener (`line(text)`, `stage(name)`). `capture`:
    the app's own record of a clip it shot (where, when, the exposure):
    the trails are then named there with no search (`run_known`); without
    a GPS fix its time still dates the blind solve. Returns the files made
    (pictures and the replay), or raises with what went wrong (which the
    app shows)."""
    def log(*args, **_):
        sink.line(" ".join(str(a) for a in args))

    from skyfind import pipeline, replay, report, tles as tles_

    tic = time.time()
    # a wrong login found now, not after the trails (a quarter of an hour)
    if st_user and st_password:
        try:
            tles_.sign_in(st_user, st_password)
            log("Space-Track: signed in")
        except tles_.TLEsNeeded as e:
            raise _app_words(e) from None
        except OSError as e:              # the night's may be saved already
            log(f"Space-Track: {_offline(e)}; going on")
    wd = pipeline.Workdir(clip, work)
    if all(wd.has(f) for f in ("clip.json", "segments.json", "tracks.json")):
        log("trails: from the work folder (a clip solved before)")
    else:
        sink.stage("finding the trails")
        pipeline.run_trails(wd, log=log)
        log(f"[trails done in {time.time() - tic:.0f} s]")

    sink.stage("solving")
    if st_user and st_password:           # this process only, never logged
        os.environ["SPACETRACK_USER"] = st_user
        os.environ["SPACETRACK_PASSWORD"] = st_password
    tles = [p for p in (tle_paths or [])]
    known, shot = _recorded(capture)
    try:
        if known is not None:
            pipeline.run_known(wd, known, tle=tles or None,
                               spacetrack=bool(tles and st_user
                                               and st_password), log=log)
        else:
            pipeline.run_solve(wd, tle=tles or None,
                               around=around or shot or None,
                               free_site=bool(free_site),
                               spacetrack=bool(tles and st_user
                                               and st_password),
                               log=log)
    except tles_.TLEsNeeded as e:
        raise _app_words(e) from None
    finally:
        os.environ.pop("SPACETRACK_PASSWORD", None)
    log(f"[solved at {time.time() - tic:.0f} s]")

    sink.stage("drawing the results")
    out = [str(p) for p in report.write(wd, log=log)]
    sink.stage("making the replay")
    mp4 = wd.path(f"{wd.clip.stem}_solve.mp4")
    replay.replay(wd, None, mp4=mp4, log=log)
    out.append(str(mp4))
    log(f"[all done in {time.time() - tic:.0f} s]")
    return out
