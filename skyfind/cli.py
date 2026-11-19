"""skyfind — where and when a star-field timelapse was shot, and which
satellites crossed it.

    skyfind trails CLIP     trail segments (the clip, less its trails)
    skyfind solve CLIP      stars, skyline, planets -> site, clock, IDs
    skyfind scan CLIP       ... when only the month is known: night by night
    skyfind report CLIP     trail graphic, close-up, object list, site map
    skyfind run CLIP        all three
    skyfind gui [FOLDER]    the workbench: all of it in a window

Everything lands in a work folder next to the clip, `<stem>.skyfind/`.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)

_WORKERS = typer.Option(None, "--workers", "-j",
                        help="processes (default: half the cores, max 16)")
_WORKDIR = typer.Option(None, "--workdir",
                        help="work folder (default: <stem>.skyfind next to "
                             "the clip)")


def _wd(clip: Path, workdir: Path | None):
    from .pipeline import Workdir
    return Workdir(clip, workdir)


@app.command()
def trails(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    radius: int = typer.Option(10, help="frames either side in the median "
                               "that takes the trails out"),
    reclean: bool = typer.Option(False, help="start over: track the sky and "
                                 "take the trails out again even if the "
                                 "work folder has them"),
    workers: int = _WORKERS,
    workdir: Path = _WORKDIR,
):
    """Find the trail segments: the clip minus a copy with the trails taken
    out, linked over frames."""
    from .pipeline import run_trails
    run_trails(_wd(clip, workdir), radius=radius, workers=workers,
               reclean=reclean, log=typer.echo)


@app.command()
def solve(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    tle: list[Path] = typer.Option(None, exists=True, dir_okay=False,
                                   help="orbital elements (2/3-line TLE "
                                   "text) covering the night; repeat to add "
                                   "sources, later ones preferred for their "
                                   "objects where current; default: cached or "
                                   "Space-Track with SPACETRACK_USER/"
                                   "PASSWORD"),
    around: str = typer.Option(None, help="rough UTC of the first frame "
                               "('YYYY-MM-DD HH:MM'), instead of dating it "
                               "from planets or comets in view"),
    window: float = typer.Option(12.0, help="hours either side of that time "
                                 "to search (12 = every longitude)"),
    lat: float = typer.Option(None, help="start latitude instead of the "
                              "skyline's (still solved for)"),
    max_tracks: int = typer.Option(None, help="use only the longest N "
                                   "tracks to anchor the solve"),
    keep_orbits: bool = typer.Option(False, help="keep the orbit table"),
    spacetrack: bool = typer.Option(False, help="with --tle: put the files "
                                    "over Space-Track's elements for the "
                                    "night (cached, or downloaded with "
                                    "SPACETRACK_USER/PASSWORD)"),
    free_site: bool = typer.Option(False, "--free-site", help="ignore the "
                                   "skyline's up and find the site from the "
                                   "satellites alone (a ragged treeline can "
                                   "put the latitude degrees off)"),
    interval: float = typer.Option(None, help="seconds between frames, "
                                   "for a clip shot on a tracking mount "
                                   "(its stars stand still and can't give "
                                   "it); otherwise searched for"),
    recorded: Path = typer.Option(None, exists=True, dir_okay=False,
                                  help="the phone app's record of a clip it "
                                  "shot (JSON: lat, lon, height_m, "
                                  "frame0_unix, exposure_s): the trails are "
                                  "named there, nothing is searched"),
    workers: int = _WORKERS,
    workdir: Path = _WORKDIR,
):
    """Blind site/time solve, then identify every trail."""
    from .pipeline import run_known, run_solve
    from .tles import TLEsNeeded
    try:
        if recorded is not None:
            import json
            run_known(_wd(clip, workdir), json.loads(recorded.read_text()),
                      tle=tle, spacetrack=spacetrack, workers=workers,
                      log=typer.echo)
            return
        run_solve(_wd(clip, workdir), tle=tle, around=around, window_h=window,
                  lat=lat, max_tracks=max_tracks, workers=workers,
                  keep_orbits=keep_orbits, spacetrack=spacetrack,
                  free_site=free_site, interval=interval, log=typer.echo)
    except (TLEsNeeded, RuntimeError) as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)


@app.command()
def moon(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    first: str = typer.Option("1990-01-01", "--from", help="first date to "
                              "list (UTC, YYYY-MM-DD)"),
    last: str = typer.Option("2049-12-31", "--to", help="last date"),
    workdir: Path = _WORKDIR,
):
    """When the Moon stood where the clip shows it (once a lunar month):
    the nights `scan` would try."""
    from .pipeline import _setup, find_moon, moon_nights
    wd = _wd(clip, workdir)
    s = _setup(wd, stars=None, workers=None, lat=None, log=typer.echo)
    m = find_moon(wd, s["cam"], s["usable"], typer.echo)
    if m is None:
        typer.echo("no Moon in view", err=True)
        raise typer.Exit(1)
    for n in moon_nights(wd, s["cam"], m, first, last, typer.echo):
        typer.echo(f"  frame 0 ~ {n['utc']} UTC (+-{n['span_h'] / 2:.1f} h),"
                   f" Moon {n['phase']}, {n['sep_deg']:.2f} deg off")


@app.command()
def scan(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    first: str = typer.Option(..., "--from", help="first night it could "
                              "be (UTC date, YYYY-MM-DD)"),
    last: str = typer.Option(..., "--to", help="last night it could be"),
    tle: list[Path] = typer.Option(None, exists=True, dir_okay=False,
                                   help="element files (repeatable); "
                                   "without any, Space-Track's"),
    spacetrack: bool = typer.Option(False, help="Space-Track's elements "
                                    "as well as the files"),
    max_tracks: int = typer.Option(20, help="longest tracks to fit each "
                                   "night"),
    free_site: bool = typer.Option(False, "--free-site", help="ignore the "
                                   "skyline's up (see solve --free-site)"),
    moon_lit: float = typer.Option(0.0, "--moon-lit", help="with the Moon "
                                   "in view: only nights with at least this "
                                   "fraction of it lit (0.5 when it plainly "
                                   "lights the land)"),
    workers: int = _WORKERS,
    workdir: Path = _WORKDIR,
):
    """Which night, when it's only roughly known: each night from --from to
    --to is searched on its own; the best then gets the full solve."""
    from .pipeline import run_scan
    from .tles import TLEsNeeded
    try:
        run_scan(_wd(clip, workdir), first, last, tle=tle,
                 spacetrack=spacetrack, max_tracks=max_tracks,
                 workers=workers, free_site=free_site,
                 moon_lit=moon_lit, log=typer.echo)
    except (TLEsNeeded, RuntimeError, ValueError) as e:
        typer.echo(f"\n{e}", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    workdir: Path = _WORKDIR,
):
    """Write the trail graphic, close-up, object list and site map."""
    from . import report as rep
    for p in rep.write(_wd(clip, workdir), log=typer.echo):
        typer.echo(f"wrote {p}")


@app.command()
def replay(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    gif: Path = typer.Option(None, help="output GIF (default: "
                             "<stem>_solve.gif in the work folder)"),
    mp4: Path = typer.Option(None, help="also (or only, with --no-gif) "
                             "an MP4 video"),
    no_gif: bool = typer.Option(False, "--no-gif"),
    width: int = typer.Option(960, help="width in pixels"),
    fps: int = typer.Option(10),
    track_frames: int = typer.Option(5, help="frames for each track's step "
                                     "on the map (more = slower)"),
    still: bool = typer.Option(False, "--still", help="everything over one "
                               "still, trail-free frame, a track at a time, "
                               "instead of the footage"),
    workdir: Path = _WORKDIR,
):
    """A GIF (and/or MP4) of the solve, on the footage at its own speed:
    every trail ringed as it goes by, the candidate sites on the world map
    closing in as the tracks are fitted; then every trail drawn in and
    named, the object list filling up."""
    from . import replay as rp
    wd = _wd(clip, workdir)
    gif = None if no_gif else gif or wd.path(f"{clip.stem}_solve.gif")
    rp.replay(wd, gif, mp4=mp4, width=width, fps=fps,
              track_frames=track_frames, still=still, log=typer.echo)


@app.command()
def gui(
    start: Path = typer.Argument(None, exists=True,
                                 help="a folder of clips, or one clip"),
):
    """The workbench: every step on a folder of clips, in one window."""
    try:
        from . import workbench
    except ImportError as e:
        typer.echo(f"the workbench needs the gui extra ({e}): "
                   "uv sync --extra gui", err=True)
        raise typer.Exit(1)
    raise typer.Exit(workbench.run(start))


@app.command()
def watch(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    solve: bool = typer.Option(False, "--solve", help="start the solve too"),
    tle: list[Path] = typer.Option(None, exists=True, dir_okay=False),
    around: str = typer.Option(None),
    workdir: Path = _WORKDIR,
):
    """Watch one clip's solve live (or replay a finished one)."""
    from . import gui as g
    args = None
    if solve:
        args = [*[a for t in tle or [] for a in ("--tle", str(t))],
                *(["--around", around] if around else [])]
    raise typer.Exit(g.run(_wd(clip, workdir), solve_args=args))


@app.command()
def run(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False),
    tle: list[Path] = typer.Option(None, exists=True, dir_okay=False),
    around: str = typer.Option(None),
    window: float = typer.Option(12.0),
    workers: int = _WORKERS,
    workdir: Path = _WORKDIR,
):
    """trails + solve + report."""
    trails(clip, radius=10, reclean=False, workers=workers, workdir=workdir)
    # every option given: a default left out here is typer's OptionInfo,
    # not its value (a truthy one: it turned --free-site on)
    solve(clip, tle=tle, around=around, window=window, lat=None,
          max_tracks=None, keep_orbits=False, spacetrack=False,
          free_site=False, interval=None, recorded=None,
          workers=workers,
          workdir=workdir)
    report(clip, workdir=workdir)


def main():
    app()


if __name__ == "__main__":
    main()
