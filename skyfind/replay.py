"""Watch a solve: trails get their names, the object list grows, and the
candidate sites on the world map close in on the answer.

One renderer draws every view; the GIF writer (`skyfind replay --gif`) and
the live window (`skyfind gui`) both use it, so they look the same.

The story, from a work folder:

1. Each anchor track is fitted against the catalogue. Every good fit is a
   candidate site (one object fits many: once per orbit, around the globe).
   A site stays in the running while it is backed by as many *distinct*
   objects as the best site so far — so the dots start everywhere and fall
   away as independent objects agree, and the map zooms in after them.
2. The joint fit pins the site (a dot with its 2-sigma ellipse).
3. Every trail is identified: named on the frame, added to the list.

A replay plays the footage forward once at its own speed: every trail the
solve used ringed as it goes by (all at once when several do), each
fitted track joining the map as its last frame passes, the map closing in
on the site when the last has; then the recount over one trail-free still,
every trail drawn in and named. What dated the night stays ringed where it
is, frame by frame. `still` (and the live window) tell it all over the
still, a track at a time.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import data
from .report import (NE, _label, colour, dating, dating_colour,
                     draw_dating, took)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 20, 18)
INK = (225, 225, 225)
DIM = (130, 130, 130)
FIT = (60, 200, 255)          # BGR amber: the track being fitted
DONE = (150, 120, 90)          # fitted tracks
CAND = (60, 170, 255)          # candidate sites
GONE = (80, 80, 80)            # eliminated sites
SITE = (60, 60, 230)           # the answer
RING = 5                       # px: a ring this far round a real trail
UNNAMED = (170, 170, 170)      # a trail the solve used that nothing named
SLOW = (255, 255, 255)         # a slow mover (a high orbit), named by position
SHADOW = (90, 90, 200)         # too fast for a sunlit satellite: aircraft


def short(name: str, n: int = 16) -> str:
    s = name.replace("STARLINK-", "SL-")
    return s if len(s) <= n else s[:n - 1] + "."


# ================================================================ story
@dataclass
class Story:
    bg: np.ndarray                         # BGR frame (cleaned) behind it all
    tracks: list                           # [{frames, segs [[x1,y1,x2,y2]], fits}]
    solution: dict | None = None
    ids: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    lat0: float = 0.0                      # skyline latitude (starting band)
    title: str = ""
    expected: int = 0                      # tracks the solve will fit
    done: bool = True                      # False while a solve is running
    failed: str | None = None              # why a solve gave up (no site)
    # what dated the night (planets or comets), placed on the background:
    # [{name, kind, x, y, state}] (`report.dating`)
    dating: list = field(default_factory=list)
    k: int = 0                             # the clip frame `bg` shows
    # for playing the footage: its rate and length, the segments the solve
    # used ([{k, x1, y1, x2, y2}], `pipeline._anchors`) and what dated the
    # night placed frame by frame ({k: dating list}, `report.dating_frames`)
    fps: float = 30.0
    n_frames: int = 0
    rows: list = field(default_factory=list)
    dating_at: dict = field(default_factory=dict)
    # segments too fast for any satellite the Sun lit there, left out
    # (`shadow`): [[k, x1, y1, x2, y2]]
    dark: list = field(default_factory=list)

    def _footage(self, wd, sol) -> None:
        info = wd.info
        self.fps = float(info.get("fps") or 30.0)
        self.n_frames = int(info["n_frames"])
        try:
            from .pipeline import _anchors
            usable = sol["usable"] if sol and sol.get("usable") else \
                (0, self.n_frames - 1)
            self.rows, _ = _anchors(wd, usable, 0, log=lambda *a: None)
        except (FileNotFoundError, KeyError):
            self.rows = []

    @classmethod
    def from_events(cls, wd, bg=None, *, frames: dict | None = None,
                    footage: bool = True) -> "Story | None":
        """The story so far, from `events.jsonl` (a solve may still be
        running). None when there's no event log yet. `frames`: the still
        backgrounds made so far, by frame (a live view asks again and
        again, and on a phone each is a median to take); `footage` False
        leaves out what only playing the footage needs."""
        if not wd.has("events.jsonl"):
            return None
        ev = [json.loads(ln) for ln in
              wd.path("events.jsonl").read_text().splitlines() if ln.strip()]
        start = next((e for e in ev if e["type"] == "start"), None)
        if start is None:
            return None
        tracks = [dict(frames=e["frames"], segs=e["segs"], fits=e["fits"])
                  for e in sorted((e for e in ev if e["type"] == "track"),
                                  key=lambda e: e["i"])]
        sol = next((e for e in ev if e["type"] == "solution"), None)
        ids = next((e for e in ev if e["type"] == "ids"), None)
        k = _background_frame(wd, ids["ids"] if ids else [])
        if bg is None:
            bg = frames.get(k) if frames is not None else None
        if bg is None:
            bg = wd.cleaned_frame(k)
            if frames is not None:
                frames[k] = bg
        story = cls(bg, tracks, sol, ids["ids"] if ids else [],
                    ids["objects"] if ids else [], start["lat0"],
                    start.get("title", ""))
        story.expected = start["n_tracks"]
        done = next((e for e in ev if e["type"] == "done"), None)
        story.done = done is not None
        dark = next((e for e in ev if e["type"] == "shadow"), None)
        story.dark = dark["segs"] if dark else []
        failed = next((e for e in ev if e["type"] == "failed"), None)
        if failed is not None:
            story.failed = failed["reason"]
            story.done = True
        if done is not None and sol is not None and "elapsed_s" in done:
            sol = story.solution = dict(sol, elapsed_s=done["elapsed_s"])
        story.dating = dating(wd, k)
        story.k = k
        if footage:
            story._footage(wd, sol)
        return story

    @classmethod
    def from_workdir(cls, wd) -> "Story":
        live = cls.from_events(wd)
        if live is not None and live.done:
            return live
        from remove_satellites.core import video
        info = wd.info
        sol = wd.read("solution.json") if wd.has("solution.json") else None
        ids = sol["ids"] if sol else []
        k = int(np.median([d["k"] for d in ids])) if ids else \
            wd.camera().k0 if wd.has("camera.json") else info["n_frames"] // 2
        bg = wd.cleaned_frame(k)
        tracks = []
        if wd.has("trackfit.json"):
            all_tracks = {tuple(t["frames"]): t for t in wd.read("tracks.json")}
            for r in wd.read("trackfit.json"):
                t = all_tracks.get(tuple(r["frames"]), {"segs": []})
                tracks.append(dict(
                    frames=r["frames"],
                    segs=[[s["x1"], s["y1"], s["x2"], s["y2"]]
                          for s in t["segs"]],
                    fits=[f for f in r["fits"] if f["med_err"] < 0.25]))
        hz = wd.read("horizon.json") if wd.has("horizon.json") else {}
        story = cls(bg, tracks, sol, ids, sol["objects"] if sol else [],
                    hz.get("lat") or 0.0, wd.clip.stem)
        story.dark = (sol or {}).get("in_shadow", [])
        story.dating = dating(wd, k)
        story.k = k
        story._footage(wd, sol)
        return story


def _background_frame(wd, ids) -> int:
    """The cleaned frame behind the replay: where the identified trails
    are, else the middle of the clip."""
    n = wd.info["n_frames"]
    k = int(np.median([d["k"] for d in ids])) if ids else n // 2
    return min(k, n - 1)


def candidates(fits: list[dict], dT: float = 40.0, dlat: float = 0.6,
               dlon: float = 0.8):
    """(support per fit, max support): how many distinct tracks have a fit
    agreeing with each fit (same test as `solve.cluster`). Fits are binned
    in (dT, dlat, dlon) cells and compared cell by cell with the
    neighbouring cells only (a free site leaves 10^5 fits)."""
    if not fits:
        return np.zeros(0, int), 0
    lat = np.array([f["lat"] for f in fits])
    lon = np.array([f["lon"] for f in fits])
    T = np.array([f["T"] for f in fits])
    _, tr = np.unique([f["track"] for f in fits], return_inverse=True)
    n_lon = math.ceil(360 / dlon)
    keys = np.stack([np.floor(T / dT), np.floor(lat / dlat),
                     np.floor(((lon + 180) % 360) / dlon)], 1).astype(np.int64)
    cells: dict[tuple, list] = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    cells = {k: np.array(v) for k, v in cells.items()}
    support = np.zeros(len(fits), int)
    n_tr = int(tr.max()) + 1
    for (a, b, c), rows in cells.items():
        cols = np.concatenate([cells[k] for k in {
            (a + i, b + j, (c + m) % n_lon) for i in (-1, 0, 1)
            for j in (-1, 0, 1) for m in (-1, 0, 1)} if k in cells])
        dl = np.abs((lon[rows, None] - lon[None, cols] + 180) % 360 - 180)
        near = (np.abs(T[rows, None] - T[None, cols]) <= dT) & \
            (np.abs(lat[rows, None] - lat[None, cols]) <= dlat) & (dl <= dlon)
        onehot = np.zeros((len(cols), n_tr), np.int32)
        onehot[np.arange(len(cols)), tr[cols]] = 1
        support[rows] = ((near.astype(np.int32) @ onehot) > 0).sum(1)
    return support, int(support.max())


class Support:
    """`candidates` kept up to date as tracks are added: each new fit's
    neighbours (in its and the neighbouring cells) give its own support,
    and every older fit near one of them gains the new track. Adding all
    of a clip's tracks costs about what one `candidates` over them all does
    (a free site leaves 10^5 fits, and the replay steps once per track)."""

    def __init__(self, dT: float = 40.0, dlat: float = 0.6,
                 dlon: float = 0.8):
        self.dT, self.dlat, self.dlon = dT, dlat, dlon
        self.n_lon = math.ceil(360 / dlon)
        self.cells: dict[tuple, list] = {}
        self.lat = np.zeros(0)
        self.lon = np.zeros(0)
        self.T = np.zeros(0)
        self.tr = np.zeros(0, int)
        self.support = np.zeros(0, int)
        self.near: list[set] = []

    def _key(self, i) -> tuple:
        return (int(math.floor(self.T[i] / self.dT)),
                int(math.floor(self.lat[i] / self.dlat)),
                int(math.floor(((self.lon[i] + 180) % 360) / self.dlon)))

    def add(self, fits: list[dict], track: int) -> None:
        if not fits:
            return
        n0 = len(self.lat)
        self.lat = np.r_[self.lat, [f["lat"] for f in fits]]
        self.lon = np.r_[self.lon, [f["lon"] for f in fits]]
        self.T = np.r_[self.T, [f["T"] for f in fits]]
        self.tr = np.r_[self.tr, np.full(len(fits), track)]
        self.support = np.r_[self.support, np.zeros(len(fits), int)]
        keys = []
        for i in range(n0, len(self.lat)):
            k = self._key(i)
            keys.append(k)
            self.cells.setdefault(k, []).append(i)
        touched: set = set()
        for i, (a, b, c) in zip(range(n0, len(self.lat)), keys):
            cols = np.fromiter((j for k in {
                (a + x, b + y, (c + z) % self.n_lon) for x in (-1, 0, 1)
                for y in (-1, 0, 1) for z in (-1, 0, 1)} if k in self.cells
                for j in self.cells[k]), int)
            dl = np.abs((self.lon[i] - self.lon[cols] + 180) % 360 - 180)
            near = cols[(np.abs(self.T[i] - self.T[cols]) <= self.dT)
                        & (np.abs(self.lat[i] - self.lat[cols]) <= self.dlat)
                        & (dl <= self.dlon)]
            self.near.append(set(self.tr[near].tolist()) | {track})
            touched.update(near[near < n0].tolist())
        for j in touched:
            if track not in self.near[j]:
                self.near[j].add(track)
        self.support = np.array([len(x) for x in self.near])

    def points(self) -> np.ndarray:
        return np.column_stack([self.lat, self.lon])


# ================================================================ scenes
@dataclass
class Scene:
    phase: str
    fitting: int | None = None             # track being fitted
    done: int = 0                          # tracks fitted so far
    sites: np.ndarray | None = None        # (m, 2) lat, lon still in
    gone: np.ndarray | None = None         # eliminated
    support: int = 0
    view: tuple = (0.0, 0.0, 60.0)         # lat, lon, half-height (deg)
    solved: bool = False
    shown: int = 0                         # identified segments shown
    fresh: int = 0                         # of which just added
    k: int | None = None                   # the footage's frame (played)
    ticks: float = 3                       # on screen for ticks / TICKS s
    slow: int = 0                          # slow movers picked out so far


def _view_of(pts: np.ndarray, lat0: float, floor: float) -> tuple:
    if pts is None or not len(pts):
        return (lat0, 0.0, 60.0)
    lat = pts[:, 0]
    lon = pts[:, 1]
    # longitudes may straddle the date line: centre on the circular mean
    c = math.degrees(math.atan2(np.sin(np.radians(lon)).mean(),
                                np.cos(np.radians(lon)).mean()))
    dl = (lon - c + 180) % 360 - 180
    kx = max(math.cos(math.radians(float(np.clip(lat.mean(), -70, 70)))), 0.3)
    half = max((lat.max() - lat.min()) / 2, (dl.max() - dl.min()) / 2 * kx / 3,
               floor) * 1.35
    return (float((lat.max() + lat.min()) / 2), float(c + (dl.max() + dl.min()) / 2),
            float(min(half, 60.0)))


TRACK_FRAMES = 5          # frames per fitted track: the map's steps
ZOOM_FRAMES = 30          # frames to zoom from the last candidates to the site
TICKS = 30                # the replay's clock: ticks per second (MP4 rate)
LINGER_TICKS = 9          # a trail's ring fades out over this after its frame
VIEW_TICKS = 10           # the map follows the candidates within about this
RECOUNT = 1.5             # the footage's recount: this much slower than still


def scenes(story: Story, *, per_track=TRACK_FRAMES, zoom=ZOOM_FRAMES,
           id_frames=60, hold=24, play: bool = False, tick: float = 3):
    """The replay as a list of Scenes (one per output frame), each lasting
    `tick` ticks.

    Still (the default, and the live window): over one still frame,
    `per_track` frames for each track's step on the map, `zoom` for the
    move in onto the site, then the trails drawn in and named `id_frames`
    at a time. `play`: the footage instead (`_played`)."""
    if play:
        return _played(story, zoom=zoom,
                       id_frames=int(round(id_frames * RECOUNT)),
                       hold=int(round(hold * RECOUNT)), tick=tick)
    out: list[Scene] = []
    lat0 = story.lat0
    world = (float(np.clip(lat0, -30, 30)), 0.0, 60.0)
    out += [Scene("tracks", view=world, ticks=tick)] * 8
    counter = Support()
    view = world
    sites = gone = None
    sup = 0
    n_tr = len(story.tracks)
    for i, t in enumerate(story.tracks):
        new = t["fits"]
        n_steps = per_track if new else 1
        counter.add(new, i)
        if len(counter.support):
            support = counter.support
            sup = int(support.max())
            keep = support >= max(sup, 1)
            pts = counter.points()
            sites, gone = pts[keep], pts[~keep]
            target = _view_of(sites, lat0, 0.4)
        else:
            target = view
        for s in range(n_steps):
            a = (s + 1) / n_steps
            view = _ease(view, target, a)
            out.append(Scene("tracks", fitting=i, done=i, sites=sites,
                             gone=gone, support=sup, view=view, ticks=tick))
        view = target
    sol = story.solution
    if sol:
        final = (sol["lat"], sol["lon"], 0.35)
        for s in range(zoom):
            out.append(Scene("solved", done=n_tr, sites=sites, gone=gone,
                             support=sup,
                             view=_ease(view, final, (s + 1) / zoom),
                             solved=True, ticks=tick))
        view = final
        out += _recount(story, n_tr, sites, gone, sup, view, final, zoom,
                        id_frames, hold, tick)
    return out


def _played(story: Story, *, zoom, id_frames, hold, tick) -> list[Scene]:
    """The footage played forward once at its own speed, every trail the
    solve used ringed in the frames that caught it (all at once when they
    come together); each fitted track joins the map as its last frame goes
    by, and when the last one has, the map closes in on the site. Then the
    recount over the still: every trail drawn in and named, the list
    filling, a little slower than the still replay's."""
    out: list[Scene] = []
    lat0 = story.lat0
    world = (float(np.clip(lat0, -30, 30)), 0.0, 60.0)
    n_tr = len(story.tracks)
    sol = story.solution
    final = (sol["lat"], sol["lon"], 0.35) if sol else world
    ends = sorted((max(t["frames"]), i) for i, t in enumerate(story.tracks)
                  if t["frames"])
    k_solved = ends[-1][0] if ends else 0
    n = story.n_frames or 1 + max(
        [d["k"] for d in story.rows] + [e for e, _ in ends] or [0])
    per = TICKS / max(story.fps, 1.0)           # ticks a frame: as filmed
    follow = min(1.0, per / VIEW_TICKS)
    counter = Support()
    sites = gone = None
    sup = done = j = 0
    view = target = world
    for k in range(n):
        while j < len(ends) and ends[j][0] <= k:
            i = ends[j][1]
            j += 1
            done += 1
            counter.add(story.tracks[i]["fits"], i)
            if len(counter.support):
                sup = int(counter.support.max())
                keep = counter.support >= max(sup, 1)
                pts = counter.points()
                sites, gone = pts[keep], pts[~keep]
                target = _view_of(sites, lat0, 0.4)
        solved = bool(sol) and k >= k_solved
        view = _toward(view, final if solved else target, follow)
        out.append(Scene("play", done=done, sites=sites, gone=gone,
                         support=sup, view=view, solved=solved, k=k,
                         ticks=per))
    if sol:
        out += _recount(story, n_tr, sites, gone, sup, view, final, zoom,
                        id_frames, hold, tick)
    return out


def _recount(story, n_tr, sites, gone, sup, view, final, zoom, id_frames,
             hold, tick) -> list[Scene]:
    """Every identified trail drawn in and named over the still, `id_frames`
    steps, the list filling (the map finishing its move in, `zoom` steps);
    then held."""
    out = []
    n = len(story.ids)
    for s in range(id_frames):
        shown = int(round(n * (s + 1) / id_frames))
        prev = int(round(n * s / id_frames))
        v = _ease(view, final, (s + 1) / zoom) if s < zoom else final
        out.append(Scene("ids", done=n_tr, sites=sites, gone=gone,
                         support=sup, view=v, solved=True, shown=shown,
                         fresh=shown - prev, ticks=tick))
    out += [Scene("final", done=n_tr, sites=sites, gone=gone, support=sup,
                  view=final, solved=True, shown=n, ticks=tick)] * hold
    # a last chapter for the slow movers (high orbits, named by position:
    # `pipeline._slow`), lost among the hundreds: one at a time, on their own
    n_slow = len(slow_movers(story.ids))
    for i in range(n_slow):
        out += [Scene("slow", done=n_tr, sites=sites, gone=gone, support=sup,
                      view=final, solved=True, shown=n, slow=i + 1,
                      ticks=tick)] * SLOW_FRAMES
    if n_slow:
        out += [out[-1]] * hold
    return out


SLOW_FRAMES = 14          # frames each slow mover has to itself


def slow_movers(ids: list) -> list:
    """The objects named by position, in order of appearance: [(norad,
    name, their segments' IDs, arcsec/s)]."""
    by: dict = {}
    for d in ids:
        if d.get("how") == "position":
            by.setdefault(d["norad"], []).append(d)
    out = [(no, ds[0]["name"], sorted(ds, key=lambda d: d["k"]),
            float(np.median([d.get("rate_arcsec_s", 0.0) for d in ds])))
           for no, ds in by.items()]
    return sorted(out, key=lambda r: r[2][0]["k"])


def _ease(a: tuple, b: tuple, f: float) -> tuple:
    """Blend two views; the zoom blends in log space so it glides."""
    f = 0.5 - 0.5 * math.cos(math.pi * min(max(f, 0), 1))
    return _toward(a, b, f)


def _toward(a: tuple, b: tuple, f: float) -> tuple:
    """`f` of the way from view `a` to `b` (the zoom in log space)."""
    dl = (b[1] - a[1] + 180) % 360 - 180
    return (a[0] + (b[0] - a[0]) * f, a[1] + dl * f,
            math.exp(math.log(a[2]) + (math.log(b[2]) - math.log(a[2])) * f))


# ================================================================ map
class WorldMap:
    """Natural Earth coastlines and borders at a detail that suits the
    zoom (110m / 50m / 10m, fetched once into the data directory)."""

    def __init__(self, log=None):
        self.layers: dict = {}
        self.log = log

    def _layer(self, scale: str, kind: str):
        key = (scale, kind)
        if key not in self.layers:
            name = {"coast": f"ne_{scale}_coastline.geojson",
                    "borders": f"ne_{scale}_admin_0_boundary_lines_land.geojson",
                    "states": f"ne_{scale}_admin_1_states_provinces_lines.geojson",
                    "places": f"ne_{scale}_populated_places_simple.geojson"}[kind]
            try:
                fc = json.loads(data.fetch(NE + name, name, quiet=True)
                                .read_text(encoding="utf-8"))
            except OSError:
                fc = {"features": []}
            lines, places = [], []
            for f in fc["features"]:
                g = f.get("geometry")
                if not g:
                    continue
                if g["type"] == "Point":
                    p = f["properties"]
                    places.append((p.get("scalerank", 10), p.get("name", ""),
                                   g["coordinates"][1], g["coordinates"][0]))
                    continue
                parts = [g["coordinates"]] if g["type"] == "LineString" \
                    else g["coordinates"]
                for ln in parts:
                    a = np.asarray(ln, float)
                    if len(a) > 1:
                        lines.append((a, a[:, 1].min(), a[:, 1].max(),
                                      a[:, 0].min(), a[:, 0].max()))
            self.layers[key] = places if kind == "places" else lines
        return self.layers[key]

    def draw(self, img: np.ndarray, view: tuple, *, sites=None, gone=None,
             solved=None) -> None:
        H, W = img.shape[:2]
        lat_c, lon_c, half = view
        kx = max(math.cos(math.radians(float(np.clip(lat_c, -70, 70)))), 0.3)
        s = H / (2 * half)                                   # px per degree

        def P(lat, lon):
            dl = (np.asarray(lon) - lon_c + 180) % 360 - 180
            return np.stack([W / 2 + dl * kx * s,
                             H / 2 - (np.asarray(lat) - lat_c) * s], -1)

        img[:] = (46, 30, 18)                                # night ocean
        scale = "110m" if half > 12 else "50m" if half > 2.5 else "10m"
        # graticule
        step = 30 if half > 25 else 10 if half > 6 else 2 if half > 1.5 else 0.5
        for la in np.arange(-90, 90.1, step):
            y = int(P(la, lon_c)[1])
            if 0 <= y < H:
                cv2.line(img, (0, y), (W, y), (70, 52, 38), 1)
        span_lon = W / (kx * s) / 2
        for lo in np.arange(math.floor((lon_c - span_lon) / step) * step,
                            lon_c + span_lon + step, step):
            x = int(P(lat_c, lo)[0])
            if 0 <= x < W:
                cv2.line(img, (x, 0), (x, H), (70, 52, 38), 1)
        lat_lo, lat_hi = lat_c - half * 1.1, lat_c + half * 1.1
        for kind, col, w in (("states", (95, 85, 80), 1),
                             ("borders", (120, 110, 105), 1),
                             ("coast", (170, 170, 160), 1)):
            if kind == "states" and half > 8:
                continue
            for a, la0, la1, lo0, lo1 in self._layer(scale, kind):
                if la1 < lat_lo or la0 > lat_hi:
                    continue
                if half < 60 and not _lon_overlap(lo0, lo1, lon_c, span_lon):
                    continue
                pts = P(a[:, 1], a[:, 0])
                # break lines where they wrap round the map edge
                jumps = np.flatnonzero(np.abs(np.diff(pts[:, 0])) > W / 2) + 1
                for part in np.split(pts, jumps):
                    if len(part) > 1:
                        cv2.polylines(img, [part.astype(np.int32)], False,
                                      col, w, cv2.LINE_AA)
        if half < 4:                                         # towns
            towns = [(r, n, la, lo) for r, n, la, lo in
                     self._layer("10m" if half < 1.5 else "50m", "places")
                     if abs(la - lat_c) < half and
                     abs((lo - lon_c + 180) % 360 - 180) < span_lon]
            for _, name, la, lo in sorted(towns)[:10]:
                x, y = P(la, lo).astype(int)
                cv2.circle(img, (x, y), 3, (200, 200, 200), -1, cv2.LINE_AA)
                cv2.putText(img, name, (x + 6, y + 4), FONT, 0.42,
                            (200, 200, 200), 1, cv2.LINE_AA)
        if gone is not None and len(gone):
            # up to 10^5 of them: 3x3 dots in one go, not a circle each
            xy = P(gone[:, 0], gone[:, 1]).astype(int)
            ok = (xy[:, 0] >= 1) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 1) & \
                (xy[:, 1] < H - 1)
            x, y = xy[ok, 0], xy[ok, 1]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    img[y + dy, x + dx] = GONE
        if sites is not None and len(sites):
            r = 4 if len(sites) < 50 else 3
            for x, y in P(sites[:, 0], sites[:, 1]).astype(int):
                cv2.circle(img, (x, y), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), r, CAND, -1, cv2.LINE_AA)
        if solved:
            x, y = P(solved["lat"], solved["lon"]).astype(int)
            ax = max(3, int(2 * solved["sig_lon"] * kx * s))
            ay = max(3, int(2 * solved["sig_lat"] * s))
            cv2.ellipse(img, (x, y), (ax, ay), 0, 0, 360, SITE, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), 5, SITE, -1, cv2.LINE_AA)
            ns = "N" if solved["lat"] >= 0 else "S"
            ew = "E" if solved["lon"] >= 0 else "W"
            _text(img, f"{abs(solved['lat']):.3f}{ns} {abs(solved['lon']):.3f}{ew}",
                  (x + 10, y - 10), 0.55, (120, 120, 255))


def _lon_overlap(lo0, lo1, lon_c, span):
    if lo1 - lo0 > 180:                       # wraps: always draw
        return True
    d0 = (lo0 - lon_c + 180) % 360 - 180
    d1 = (lo1 - lon_c + 180) % 360 - 180
    return not ((d0 > span and d1 > span) or (d0 < -span and d1 < -span))


def _text(img, t, org, scale, col, thick=1):
    _label(img, t, org, scale, col, thick)


def _blend_line(img, p1, p2, col, thick: int, alpha: float) -> None:
    """A line over `img` at opacity `alpha` (a band over a trail leaves the
    trail showing through). Only the line's bounding box is blended."""
    if alpha <= 0:
        return
    h, w = img.shape[:2]
    r = thick + 2
    x0, x1 = max(min(p1[0], p2[0]) - r, 0), min(max(p1[0], p2[0]) + r + 1, w)
    y0, y1 = max(min(p1[1], p2[1]) - r, 0), min(max(p1[1], p2[1]) + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.line(over, (p1[0] - x0, p1[1] - y0), (p2[0] - x0, p2[1] - y0), col,
             thick, cv2.LINE_AA)
    img[y0:y1, x0:x1] = cv2.addWeighted(over, alpha, roi, 1 - alpha, 0)


def _ring(img, p1, p2, col, alpha: float = 0.9) -> None:
    """A capsule `RING` px round a trail, the trail itself (everything
    inside) left as the footage has it."""
    h, w = img.shape[:2]
    r = RING + 3
    x0, x1 = max(min(p1[0], p2[0]) - r, 0), min(max(p1[0], p2[0]) + r + 1, w)
    y0, y1 = max(min(p1[1], p2[1]) - r, 0), min(max(p1[1], p2[1]) + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return
    q1, q2 = (p1[0] - x0, p1[1] - y0), (p2[0] - x0, p2[1] - y0)
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.line(over, q1, q2, col, 2 * RING + 3, cv2.LINE_AA)
    inside = np.zeros(roi.shape[:2], np.uint8)
    cv2.line(inside, q1, q2, 255, 2 * RING - 1, cv2.LINE_8)
    over[inside > 0] = roi[inside > 0]
    img[y0:y1, x0:x1] = cv2.addWeighted(over, alpha, roi, 1 - alpha, 0)


def levels(img: np.ndarray, gain_max: float = 2.0, gamma: float = 0.85
           ) -> np.ndarray:
    """A lookup table that stretches a night frame's levels — the sky's
    black up to its brightest few pixels, at most `gain_max`, a little
    gamma — so faint trails still show once the footage is shrunk to the
    replay's size (averaging thins a 1-2 px trail out)."""
    small = cv2.resize(img, (max(img.shape[1] // 4, 1),
                             max(img.shape[0] // 4, 1)),
                       interpolation=cv2.INTER_AREA)
    lum = small.mean(2) if small.ndim == 3 else small
    lo, hi = float(np.percentile(lum, 5)), float(np.percentile(lum, 99.7))
    hi = max(hi, lo + 255 / gain_max)
    v = np.clip((np.arange(256) - lo) / (hi - lo), 0, 1) ** gamma
    return (v * 255).round().astype(np.uint8)


class ClipFrames:
    """The clip's frames as a replay plays them: read forward through the
    clip (a step back seeks), shrunk to the renderer's frame size, through
    `lut` (`levels`, one for the whole clip so it doesn't flicker) when
    given. Called with a frame number: that frame, or None past the end."""

    def __init__(self, path, size: tuple[int, int],
                 lut: np.ndarray | None = None):
        self.path, self.size, self.lut = str(path), size, lut
        self.cap = None
        self.next = 0                         # the frame a read gives next
        self.last: tuple = (None, None)

    def __call__(self, k: int) -> np.ndarray | None:
        if self.last[0] == k:
            return self.last[1]
        if self.cap is None or k < self.next:
            if self.cap is not None:
                self.cap.release()
            from remove_satellites.core import video
            self.cap = video.capture(self.path)
            if k:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            self.next = k
        while self.next < k:                  # decoded, not converted
            if not self.cap.grab():
                return None
            self.next += 1
        ok, img = self.cap.read()
        if not ok:
            return None
        self.next += 1
        small = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        f = small if self.lut is None else cv2.LUT(small, self.lut)
        self.last = (k, f)
        return f

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ================================================================ renderer
class Renderer:
    """Scene -> BGR image: the frame (top left), the object list (top
    right), the world map (bottom)."""

    def __init__(self, story: Story, width: int = 960, list_w: int = 300,
                 map_h: int = 300, source=None):
        self.story = story
        # the footage (`ClipFrames`: clip frame -> frame at this size), for
        # scenes that show one; without it everything goes over `story.bg`
        self.source = source
        self.W = width
        self.fw = width - list_w
        h, w = story.bg.shape[:2]
        self.fh = int(round(self.fw * h / w))
        self.sx = self.fw / w
        self.map_h = map_h
        self.H = self.fh + map_h
        self.list_w = list_w
        self.bg = (cv2.resize(story.bg, (self.fw, self.fh),
                              interpolation=cv2.INTER_AREA).astype(np.float32)
                   * 0.6).astype(np.uint8)
        self.map = WorldMap()
        # identified segments in the order they get revealed (by frame)
        ids = sorted(range(len(story.ids)), key=lambda i: story.ids[i]["k"])
        self.id_order = [story.ids[i] for i in ids]
        by_norad = {o["norad"]: o for o in story.objects}
        self.obj_of = [by_norad.get(d["norad"]) for d in self.id_order]
        # the segments the solve used, by frame, each in the colour it's
        # ringed in on the footage: the fitted tracks' amber, a named one
        # its object's, the rest grey
        key = lambda k, x, y: (int(k), round(x, 1), round(y, 1))  # noqa: E731
        named = {key(d["k"], d["x1"], d["y1"]): o
                 for d, o in zip(self.id_order, self.obj_of) if o}
        fitted = {key(k, g[0], g[1]) for t in story.tracks
                  for k, g in zip(t["frames"], t["segs"])}
        dark = {key(d[0], d[1], d[2]) for d in story.dark}
        self.slow = slow_movers(story.ids)
        slow_keys = {key(d["k"], d["x1"], d["y1"])
                     for _, _, ds, _ in self.slow for d in ds}
        have = {key(r["k"], r["x1"], r["y1"]) for r in story.rows}
        # (ringed on the footage too: their segments aren't among the ones
        # the solve timed)
        extra = [dict(k=d["k"], x1=d["x1"], y1=d["y1"], x2=d["x2"],
                      y2=d["y2"]) for _, _, ds, _ in self.slow for d in ds
                 if key(d["k"], d["x1"], d["y1"]) not in have]
        self.rows = sorted(list(story.rows) + extra, key=lambda r: r["k"])
        self.row_k = np.array([r["k"] for r in self.rows], int)
        self.row_col = []
        for r in self.rows:
            q = key(r["k"], r["x1"], r["y1"])
            self.row_col.append(FIT if q in fitted else
                                SLOW if q in slow_keys else
                                colour(named[q]) if q in named else
                                SHADOW if q in dark else UNNAMED)

    def _frame(self, sc: Scene) -> tuple[np.ndarray, bool]:
        """The picture behind a scene: its clip frame from the footage, else
        the still. -> (frame, from the footage)."""
        if self.source is not None and sc.k is not None:
            f = self.source(sc.k)
            if f is not None:
                return f.copy(), True
        return self.bg.copy(), False

    def render(self, sc: Scene) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[:] = BG
        frame, real = self._frame(sc)
        s = self.sx
        st = self.story
        if sc.phase == "play":
            self._activity(frame, sc)          # the footage, trails ringed
        # tracks: fitted ones dim, the current one bright
        elif sc.phase == "tracks" or not sc.solved:
            for i, t in enumerate(st.tracks[:sc.done + 1]):
                col = FIT if i == sc.fitting else DONE
                w = 2 if i == sc.fitting else 1
                for x1, y1, x2, y2 in t["segs"]:
                    cv2.line(frame, (int(x1 * s), int(y1 * s)),
                             (int(x2 * s), int(y2 * s)), col, w, cv2.LINE_AA)
        elif sc.phase == "solved":
            # the tracks whose objects agreed on the site, named
            for a in (st.solution or {}).get("anchors", []):
                t = next((t for t in st.tracks
                          if t["frames"] == a["frames"]), None)
                if not t or not t["segs"]:
                    continue
                c = colour(dict(name=a["name"], conf="confirmed"))
                for x1, y1, x2, y2 in t["segs"]:
                    cv2.line(frame, (int(x1 * s), int(y1 * s)),
                             (int(x2 * s), int(y2 * s)), c, 2, cv2.LINE_AA)
                x1, y1, x2, y2 = t["segs"][len(t["segs"]) // 2]
                _text(frame, short(a["name"], 14),
                      (int((x1 + x2) / 2 * s) + 5, int((y1 + y2) / 2 * s) - 5),
                      0.4, c)
        if sc.phase == "slow":
            # everything named so far faint, the slow movers on top: those
            # picked out so far, the newest ringed in white
            frame[:] = (frame.astype(np.float32) * 0.55).astype(np.uint8)
            for d in self.id_order:
                if d.get("how") != "position":
                    cv2.line(frame, *self._pts(d["x1"], d["y1"], d["x2"],
                                               d["y2"]), (70, 70, 70), 1,
                             cv2.LINE_AA)
            for i, (_, name, ds, rate) in enumerate(self.slow[:sc.slow]):
                new = i == sc.slow - 1
                for d in ds:
                    p1, p2 = self._pts(d["x1"], d["y1"], d["x2"], d["y2"])
                    if new:
                        cv2.line(frame, p1, p2, FIT, 5, cv2.LINE_AA)
                    cv2.line(frame, p1, p2, SLOW, 2, cv2.LINE_AA)
                d = ds[len(ds) // 2]
                p1, p2 = self._pts(d["x1"], d["y1"], d["x2"], d["y2"])
                _text(frame, f"{short(name, 18)}  {rate:.0f}\"/s",
                      ((p1[0] + p2[0]) // 2 + 6, (p1[1] + p2[1]) // 2 - 6),
                      0.42, FIT if new else SLOW)
        # identified trails, drawn in and named
        labelled = set()
        for j, d in enumerate(self.id_order[:sc.shown]
                              if sc.phase not in ("play", "slow") else []):
            o = self.obj_of[j]
            c = colour(o) if o else INK
            fresh = j >= sc.shown - sc.fresh
            p1 = (int(d["x1"] * s), int(d["y1"] * s))
            p2 = (int(d["x2"] * s), int(d["y2"] * s))
            if fresh:
                cv2.line(frame, p1, p2, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(frame, p1, p2, c, 2, cv2.LINE_AA)
            if o and o["norad"] not in labelled and (fresh or sc.phase == "final"):
                labelled.add(o["norad"])
                mx, my = (p1[0] + p2[0]) // 2 + 4, (p1[1] + p2[1]) // 2 - 4
                _text(frame, short(o["name"], 14), (mx, my), 0.38, c)
        # what dated the night: a ring and a name, all along (on the
        # footage where it is in that frame)
        for o in st.dating_at.get(sc.k, []) if real else st.dating:
            draw_dating(frame, o, s, 0.9)
        img[:self.fh, :self.fw] = frame
        self._list(img, sc)
        self._map(img, sc)
        return img

    def _pts(self, x1, y1, x2, y2):
        s = self.sx
        return (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))

    def _activity(self, frame, sc: Scene) -> None:
        """Every trail the solve used that the footage is showing, ringed,
        all at once (a train's side by side); each ring fades out over
        LINGER_TICKS after its frame."""
        k = sc.k
        linger = max(1, int(round(LINGER_TICKS / max(sc.ticks, 1e-3))))
        a = int(np.searchsorted(self.row_k, k - linger, "right"))
        b = int(np.searchsorted(self.row_k, k, "right"))
        for j in range(a, b):
            r = self.rows[j]
            age = k - r["k"]
            alpha = 0.9 if age == 0 else 0.6 * (1 - age / (linger + 1))
            _ring(frame, *self._pts(r["x1"], r["y1"], r["x2"], r["y2"]),
                  self.row_col[j], alpha)

    def _list(self, img, sc: Scene):
        x0, W = self.fw + 12, self.list_w - 18
        st = self.story
        cv2.putText(img, "skyfind", (x0, 26), FONT, 0.75, INK, 2, cv2.LINE_AA)
        cv2.putText(img, st.title[:28], (x0, 48), FONT, 0.45, DIM, 1,
                    cv2.LINE_AA)
        # of all the solve will fit (a live one's so far are fewer)
        n_tr = max(len(st.tracks), st.expected or 0)
        if not sc.solved:
            lines = [f"tracks fitted {sc.done}/{n_tr}" if sc.phase == "play"
                     else f"fitting track {min(sc.done + 1, n_tr)}/{n_tr}",
                     f"{len(sc.sites)} candidate sites"
                     if sc.sites is not None else "no candidate sites yet",
                     f"best: {sc.support} objects agree" if sc.support
                     else ""]
            if st.failed and sc.done >= n_tr:
                lines.append(f"{st.failed}: no solution")
        else:
            sol = st.solution
            lines = [f"{sc.support} objects agree",
                     f"frame 0 {sol['frame0_utc'][:16]} UTC",
                     f"exposure {sol['E']:.1f} s / {sol['spf']:.1f} s"]
            if "elapsed_s" in sol:
                lines.append(f"solved in {took(sol['elapsed_s'])}")
        y = 74
        for ln in lines:
            cv2.putText(img, ln, (x0, y), FONT, 0.45, INK, 1, cv2.LINE_AA)
            y += 20
        if st.dark:
            cv2.putText(img, f"{len(st.dark)} too fast for the sunlit sky",
                        (x0, y), FONT, 0.42, SHADOW, 1, cv2.LINE_AA)
            y += 20
        if st.dating:
            c = dating_colour(st.dating[0])
            kind = st.dating[0]["kind"]
            what = "the Moon" if kind == "moon" else f"{kind}s:"
            cv2.putText(img, f"dated by {what}", (x0, y),
                        FONT, 0.42, c, 1, cv2.LINE_AA)
            y += 18
            for o in (st.dating if kind != "moon" else []):
                cv2.putText(img, f"  {o['name'][:24]}", (x0, y), FONT,
                            0.42, c, 1, cv2.LINE_AA)
                y += 18
        y += 6
        cv2.line(img, (x0, y), (x0 + W, y), (70, 70, 70), 1)
        y += 18
        if sc.phase == "slow":
            cv2.putText(img, f"{len(self.slow)} slow movers: high orbits,",
                        (x0, y - 4), FONT, 0.42, DIM, 1, cv2.LINE_AA)
            cv2.putText(img, "named by where they are", (x0, y + 12), FONT,
                        0.42, DIM, 1, cv2.LINE_AA)
            y += 34
            cap = max(1, (self.fh - y) // 17)
            lo = max(0, sc.slow - cap)
            for i, (_, name, ds, rate) in enumerate(self.slow[lo:sc.slow], lo):
                new = i == sc.slow - 1
                if new:
                    cv2.rectangle(img, (x0 - 4, y - 12), (x0 + W, y + 4),
                                  (60, 60, 60), -1)
                cv2.putText(img, f"{rate:4.0f}\"/s {short(name, 18)}",
                            (x0, y), FONT, 0.42, FIT if new else SLOW, 1,
                            cv2.LINE_AA)
                y += 17
            return
        # objects in order of first identification
        seen, rows = set(), []
        for j in range(sc.shown):
            o = self.obj_of[j]
            if o and o["norad"] not in seen:
                seen.add(o["norad"])
                rows.append((o, j >= sc.shown - sc.fresh))
        cap = max(1, (self.fh - y) // 17)
        if len(rows) > cap:
            rows = rows[-cap:]
        head = f"{len(seen)} objects identified" if sc.shown else \
            "objects: naming them next" if sc.solved else \
            "objects: waiting for the site"
        cv2.putText(img, head, (x0, y - 4), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        y += 10
        for o, fresh in rows:
            c = colour(o)
            if fresh:
                cv2.rectangle(img, (x0 - 4, y - 12), (x0 + W, y + 4),
                              (60, 60, 60), -1)
            cv2.putText(img, f"{o['utc'][:5]} {short(o['name'], 20)}",
                        (x0, y), FONT, 0.42, c, 1, cv2.LINE_AA)
            y += 17

    def _map(self, img, sc: Scene):
        m = img[self.fh:, :]
        self.map.draw(m, sc.view, sites=sc.sites, gone=sc.gone,
                      solved=self.story.solution if sc.solved else None)
        n = len(sc.sites) if sc.sites is not None else 0
        label = ("site found" if sc.solved else
                 f"{n} candidate site{'s' if n != 1 else ''}"
                 if sc.sites is not None else "fitting tracks...")
        cv2.rectangle(m, (0, 0), (230, 56), (40, 26, 16), -1)
        cv2.putText(m, label, (12, 24), FONT, 0.6, INK, 1, cv2.LINE_AA)
        cv2.putText(m, f"view +-{sc.view[2]:.2f} deg" if sc.view[2] < 10
                    else "world", (12, 46), FONT, 0.45, DIM, 1, cv2.LINE_AA)


# ================================================================ output
class VideoOut:
    """Frames piped to ffmpeg as they're rendered: a GIF at `fps` or an MP4
    at TICKS a second (players and social sites dislike low frame rates).
    Each frame is `add`ed with how many ticks it lasts; the writer puts out
    the frames that fall on its own clock (an MP4 holds a 3-tick frame for
    3 frames, a 10 fps GIF keeps every third 1-tick frame)."""

    def __init__(self, path, w: int, h: int, *, gif: bool, fps: int = 10,
                 width: int | None = None, crf: int = 18):
        from remove_satellites.core.video import ANDROID
        self.jw = None
        if ANDROID and not gif:
            # the phone: its own H.264 encoder (the app's Mp4Writer), fed
            # I420 frames padded to whole macroblocks
            from java import jclass
            self.gif, self.rate = False, TICKS
            self.t = self.sent = 0
            self.w16, self.h16 = -(-w // 16) * 16, -(-h // 16) * 16
            self.jw = jclass("com.skyfind.Mp4Writer")(str(path), self.w16,
                                                      self.h16, TICKS)
            return
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"writing {'a GIF' if gif else 'a video'} "
                               "needs ffmpeg on PATH")
        self.gif = gif
        self.rate = fps if gif else TICKS
        self.t = self.sent = 0
        head = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                "bgr24", "-s", f"{w}x{h}", "-r", str(self.rate), "-i", "-"]
        if gif:
            scale = f"scale={width}:-2:flags=lanczos," if width else ""
            tail = ["-vf", f"{scale}split[a][b];[a]palettegen=max_colors=160:"
                    "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:"
                    "bayer_scale=4:diff_mode=rectangle", "-loop", "0"]
        else:
            tail = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        self.p = subprocess.Popen(head + tail + [str(path)],
                                  stdin=subprocess.PIPE)

    def add(self, img: np.ndarray, ticks: float) -> None:
        self.t += ticks
        due = int(round(self.t * self.rate / TICKS)) - self.sent
        if due > 0 and self.jw is not None:
            h, w = img.shape[:2]
            if (h, w) != (self.h16, self.w16):
                pad = np.empty((self.h16, self.w16, 3), np.uint8)
                pad[:] = BG
                pad[:h, :w] = img
                img = pad
            self.jw.write(cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).tobytes(),
                          due)
            self.sent += due
        elif due > 0:
            b = np.ascontiguousarray(img).tobytes()
            for _ in range(due):
                self.p.stdin.write(b)
            self.sent += due

    def close(self) -> int:
        """-> frames written."""
        if self.jw is not None:
            self.jw.close()
            return self.sent
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg failed writing the "
                               + ("GIF" if self.gif else "video"))
        return self.sent


def write_gif(frames, path, fps: int = 10, width: int | None = None) -> None:
    """Frames (BGR, one each at `fps`) -> an optimised GIF through ffmpeg's
    palette filters."""
    frames = list(frames)
    h, w = frames[0].shape[:2]
    out = VideoOut(path, w, h, gif=True, fps=fps, width=width)
    for f in frames:
        out.add(f, TICKS / fps)
    out.close()


def write_mp4(frames, path, fps: int = 10, crf: int = 18) -> None:
    """Frames (BGR, one each at `fps`) -> H.264 MP4 at 30 fps, each frame
    held."""
    frames = list(frames)
    h, w = frames[0].shape[:2]
    out = VideoOut(path, w, h, gif=False, crf=crf)
    for f in frames:
        out.add(f, TICKS / fps)
    out.close()


def replay(wd, gif=None, *, mp4=None, width: int = 960, fps: int = 10,
           track_frames: int = TRACK_FRAMES, still: bool = False,
           log=print) -> float:
    """Render a work folder's solve; write a GIF and/or an MP4 when given.

    The story plays on the footage, at its own speed: every trail the solve
    used ringed as it goes by, the candidate sites closing in as each
    fitted track ends; then every trail drawn in and named over the still,
    the list filling. `still`: all of it over the still, a track at a time
    (the live window's way). Returns the replay's length in seconds."""
    story = Story.from_workdir(wd)
    r = Renderer(story, width=width)
    sc = scenes(story, per_track=track_frames, play=not still,
                tick=TICKS / fps)
    if not still:
        from .report import dating_frames
        story.dating_at = dating_frames(
            wd, sorted({s.k for s in sc if s.k is not None}))
        r.source = ClipFrames(wd.clip, (r.fw, r.fh), lut=levels(story.bg))
    outs = [(path, VideoOut(path, r.W, r.H, gif=gif_, fps=fps))
            for path, gif_ in ((gif, True), (mp4, False)) if path]
    for s in sc:
        img = r.render(s)
        for _, out in outs:
            out.add(img, s.ticks)
    seconds = sum(s.ticks for s in sc) / TICKS
    if r.source is not None:
        r.source.close()
    for path, out in outs:
        n = out.close()
        if log:
            log(f"wrote {path} ({n} frames, {seconds:.0f} s)")
    return seconds
