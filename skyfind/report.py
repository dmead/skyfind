"""Deliverables: the identified trails drawn on the cleaned sky with a
numbered object list, a close-up of the busiest patch, a JSON list, and a
map of the solved site with its error ellipse and the camera's view.

Map layers are Natural Earth (public domain) coastlines, state/province
borders and towns, downloaded once into the data directory; without them
(offline) the map is skipped, the rest still written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import data

FONT = cv2.FONT_HERSHEY_SIMPLEX
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
LAYERS = {"coast": "ne_10m_coastline.geojson",
          "borders": "ne_10m_admin_1_states_provinces_lines.geojson",
          "countries": "ne_10m_admin_0_boundary_lines_land.geojson",
          "places": "ne_10m_populated_places_simple.geojson"}


def took(seconds: float) -> str:
    """'48 s', '2 min 31 s', '1 h 05 min'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def colour(o: dict) -> tuple:
    n = o["name"].upper()
    if "STARLINK" in n:
        c = (255, 200, 60)                 # BGR sky blue
    elif "R/B" in n or "DEB" in n:
        c = (60, 150, 255)                 # orange: rocket bodies, debris
    else:
        c = (90, 255, 170)                 # green: other payloads
    return c if o["conf"] == "confirmed" else tuple(int(v * 0.6) for v in c)


PLANET = (0, 215, 255)                 # BGR gold: a planet that dated it
COMET = (230, 110, 255)                # magenta: a comet that did
MOON = (200, 235, 255)                 # pale: the Moon (with the scan)


def dating_colour(o: dict) -> tuple:
    return {"planet": PLANET, "comet": COMET, "moon": MOON}[o["kind"]]


def _comet_label(name: str) -> str:
    """'C/2025 A6 (Lemmon)' -> 'Lemmon'; '12P/Pons-Brooks' as it is."""
    return name[name.index("(") + 1:name.rindex(")")] if "(" in name \
        else name


def dating(wd, k: int, samples: int = 80) -> list[dict]:
    """The planets or comets that dated the night (`date.json`), placed by
    the camera model: [{name, kind, x, y, state, path, mark}]. (x, y,
    state) is where it is at frame k — state "in view", "set" (behind the
    land) or "out of frame"; `path` is its [x, y] across the clip where
    it's in view; `mark` is (x, y, note) where to ring it on frame k's
    picture: where it is, else where it last was (or first will be), with
    when (a comet low in the west has often set by the replay's frame).
    Empty when the date was given or scanned."""
    return dating_frames(wd, [k], samples).get(k, [])


def dating_frames(wd, ks, samples: int = 80) -> dict[int, list[dict]]:
    """`dating` for many frames at once (each object placed once per frame):
    {k: [...]} for the frames `ks` (a replay playing the clip through)."""
    ks = [int(k) for k in ks]
    if not ks or not (wd.has("date.json") and wd.has("camera.json")):
        return {}
    d = wd.read("date.json")
    src = d.get("source")
    if src not in ("planets", "comets", "moon") or not d.get("candidates"):
        return {}
    c, objs = d["candidates"][0], d.get("objects", [])
    if src == "planets":
        pairs = [(int(i), n.split()[0].capitalize())
                 for i, n in c.get("planets", {}).items()]
    elif src == "moon":
        pairs = [(0, "Moon")]
    else:
        pairs = [(int(i), _comet_label(n))
                 for i, n in zip(c.get("objects", []), c.get("comets", []))]
    from .camera import radec_to_vec
    kind = "moon" if src == "moon" else src[:-1]
    cam = wd.camera()
    fg = wd.fgmask() if wd.has("fgmask.png") else None
    n = wd.info["n_frames"]
    grid = [int(q) for q in np.unique(np.linspace(0, n - 1, samples)
                                      .astype(int))]
    utc = _frame_utc(wd, cam)

    def place(v, kk):
        r = cam.rays(v, kk)
        xy, z = cam.ray_to_px(r)
        x, y = (float(q) for q in xy[0])
        # far off the axis the lens polynomial folds back into the frame:
        # a real pixel maps back to the ray it came from
        back = float(cam.px_to_ray(xy)[0] @ r[0]) / np.linalg.norm(r[0])
        if not (z[0] > 0 and 0 <= x < cam.width and 0 <= y < cam.height
                and back > np.cos(np.radians(0.5))):
            return x, y, "out of frame"
        if fg is not None and fg[int(y) * fg.shape[0] // cam.height,
                                 int(x) * fg.shape[1] // cam.width]:
            return x, y, "set"
        return x, y, "in view"

    out: dict[int, list[dict]] = {k: [] for k in ks}
    for i, name in sorted(pairs):
        if i >= len(objs):
            continue
        v = radec_to_vec(objs[i]["ra"], objs[i]["dec"])[None]
        at = {kk: place(v, kk) for kk in sorted(set(grid) | set(ks))}
        for k in ks:
            x, y, state = at[k]
            seen = [kk for kk in sorted(set(grid) | {k})
                    if at[kk][2] == "in view"]
            before = [kk for kk in seen if kk < k]
            after = [kk for kk in seen if kk > k]
            if state == "in view":
                mark = (x, y, f"dated the night ({kind})")
            elif before:
                kk = before[-1]
                gone = "set" if state == "set" else "left the frame"
                mark = at[kk][:2] + (f"{gone} {utc(kk)}".strip(),)
            elif after:
                kk = after[0]
                come = "rose" if state == "set" else "came into frame"
                mark = at[kk][:2] + (f"{come} {utc(kk)}".strip(),)
            else:
                mark = None
            out[k].append(dict(name=name, kind=kind, x=x, y=y, state=state,
                               path=[at[kk][:2] for kk in seen], mark=mark))
    return out


def _frame_utc(wd, cam):
    """frame -> 'HH:MM UTC' once solved, else ''."""
    if not wd.has("solution.json"):
        return lambda k: ""
    from datetime import datetime, timedelta
    sol = wd.read("solution.json")
    t0 = datetime.strptime(sol["frame0_utc"][:19], "%Y-%m-%d %H:%M:%S")
    return lambda k: (t0 + timedelta(seconds=float(cam.s(k)) * sol["spf"])
                      ).strftime("%H:%M UTC")


def draw_dating(img, o: dict, sx: float, size: float):
    """One planet or comet from `dating` on a picture of the frame scaled
    `sx`: its path through the clip dotted, a ring where `mark` says, its
    name and the note; `size` 1 suits a 1000-px-wide picture."""
    c = dating_colour(o)
    dim = tuple(int(v * 0.7) for v in c)
    for x, y in o["path"]:
        cv2.circle(img, (int(x * sx), int(y * sx)), max(1, int(2 * size)),
                   dim, -1, cv2.LINE_AA)
    if not o["mark"]:
        return
    x, y, note = o["mark"]
    q, r = (int(x * sx), int(y * sx)), int(13 * size)
    th = max(1, int(round(2 * size)))
    cv2.circle(img, q, r, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.circle(img, q, r, c, th, cv2.LINE_AA)
    # the label on the side with room, kept inside the picture (a mark can
    # sit on the edge its object left by)
    lines = [(o["name"], 0.5 * size, th, 0),
             (note, 0.36 * size, max(1, th // 2), int(16 * size))]
    w = max(cv2.getTextSize(t, FONT, fs, t_)[0][0] for t, fs, t_, _ in lines)
    right = q[0] + r + w + int(10 * size) < img.shape[1]
    tx = q[0] + r + int(5 * size) if right else q[0] - r - int(5 * size) - w
    ty = min(max(q[1] - int(3 * size), int(18 * size)),
             img.shape[0] - int(20 * size))
    for t, fs, t_, dy in lines:
        _label(img, t, (tx, ty + dy), fs, c, t_)


def _label(img, text, org, scale, c, thick):
    """Text with a black outline. The outline is the same text nudged
    around, not thicker: OpenCV 5 sets thickness 1 in a narrower face than
    2+, so a thick outline runs past the letters."""
    x, y = org
    d = thick + 1
    for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d),
                   (-d, -d), (d, d), (-d, d), (d, -d)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0),
                    thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, c, thick, cv2.LINE_AA)


def trails_image(bg: np.ndarray, sol: dict, title: str,
                 dated=()) -> np.ndarray:
    """The trails, numbered, on the frame; `dated` (from `dating`) are the
    planets or comets that dated the night, ringed where they're in view."""
    H, W = bg.shape[:2]
    s = H / 2160
    PW = int(1500 * s)
    ids = {d["seg"]: d for d in sol["ids"]}
    canvas = np.zeros((H, W + PW, 3), np.uint8)
    canvas[:, :W] = (bg.astype(np.float32) * 0.55).astype(np.uint8)
    canvas[:, W:] = (22, 18, 16)
    placed = []
    for o in sol["objects"]:
        c = colour(o)
        segs = [ids[i] for i in o["segs"]]
        for d in segs:
            cv2.line(canvas, (int(d["x1"]), int(d["y1"])),
                     (int(d["x2"]), int(d["y2"])), c,
                     max(1, int((5 if o["conf"] == "confirmed" else 3) * s)),
                     cv2.LINE_AA)
        d = segs[len(segs) // 2]
        x = int((d["x1"] + d["x2"]) / 2 + 10 * s)
        y = int((d["y1"] + d["y2"]) / 2 - 10 * s)
        for _ in range(12):                              # nudge labels apart
            if all(abs(x - px) > 46 * s or abs(y - py) > 30 * s
                   for px, py in placed):
                break
            y -= int(30 * s)
        placed.append((x, y))
        _label(canvas, str(o["n"]), (x, y), 0.95 * s, c, max(1, int(2 * s)))
    if dated:
        frame = np.ascontiguousarray(canvas[:, :W])
        for o in dated:
            draw_dating(frame, o, 1.0, 2 * s)
        canvas[:, :W] = frame
    x0, y = W + int(40 * s), int(70 * s)
    cv2.putText(canvas, title, (x0, y), FONT, 1.25 * s, (235, 235, 235),
                max(1, int(2 * s)), cv2.LINE_AA)
    y += int(50 * s)
    ns, ew = ("N" if sol["lat"] >= 0 else "S"), ("E" if sol["lon"] >= 0 else "W")
    sub = (f"{sol['frame0_utc'][:10]} UTC   site {abs(sol['lat']):.3f}{ns} "
           f"{abs(sol['lon']):.3f}{ew} (blind solve)   "
           f"{len(sol['objects'])} objects"
           + (f"   solved in {took(sol['elapsed_s'])}"
              if "elapsed_s" in sol else ""))
    cv2.putText(canvas, sub, (x0, y), FONT, 0.7 * s, (170, 170, 170), 1,
                cv2.LINE_AA)
    y += int(40 * s)
    x = x0
    for c, t in [((255, 200, 60), "Starlink"),
                 ((60, 150, 255), "rocket body / debris"),
                 ((90, 255, 170), "other satellite")]:
        cv2.line(canvas, (x, y - int(8 * s)), (x + int(40 * s), y - int(8 * s)),
                 c, max(1, int(5 * s)))
        cv2.putText(canvas, t, (x + int(50 * s), y), FONT, 0.65 * s,
                    (200, 200, 200), 1, cv2.LINE_AA)
        x += int((60 + 14 * len(t)) * s)
    y += int(30 * s)
    cv2.putText(canvas, "bright = confirmed (multi-frame or tight fit)   "
                "dim = probable", (x0, y), FONT, 0.6 * s, (150, 150, 150), 1,
                cv2.LINE_AA)
    y += int(30 * s)
    if dated:
        c = dating_colour(dated[0])
        cv2.circle(canvas, (x0 + int(20 * s), y - int(8 * s)), int(14 * s), c,
                   max(1, int(4 * s)), cv2.LINE_AA)
        names = ", ".join(o["name"] for o in dated)
        kind = dated[0]["kind"]
        what = ("dated the night (the Moon; the satellites chose which)"
                if kind == "moon" else f"dated the night ({kind}s): {names}")
        cv2.putText(canvas, what, (x0 + int(50 * s), y), FONT, 0.65 * s, c,
                    1, cv2.LINE_AA)
        y += int(30 * s)
    rows = sol["objects"]
    ncol = 1 if len(rows) <= 40 else 2
    per = max(1, math.ceil(len(rows) / ncol))
    lh = min(int(40 * s), (H - y - int(30 * s)) // per)
    fs = min(0.62 * s, lh / 36)
    for i, o in enumerate(rows):
        cx = x0 + (i // per) * (PW // ncol)
        cy = y + lh * (i % per + 1)
        name = o["name"] if len(o["name"]) <= 20 else o["name"][:19] + "."
        cv2.putText(canvas, f"{o['n']:>2} {o['utc'][:5]} {name}", (cx, cy),
                    FONT, fs, colour(o), 1, cv2.LINE_AA)
    return canvas


def busiest(sol: dict, W: int, H: int, fw: float = 1 / 3, fh: float = 0.45):
    """(x0, y0, x1, y1) window holding the most identified trail midpoints."""
    mids = np.array([((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2)
                     for d in sol["ids"]]) if sol["ids"] else np.zeros((0, 2))
    ww, hh = int(W * fw), int(H * fh)
    best = (-1, 0, 0)
    for x in range(0, W - ww + 1, max(1, ww // 8)):
        for y in range(0, H - hh + 1, max(1, hh // 8)):
            n = int(((mids[:, 0] >= x) & (mids[:, 0] < x + ww) &
                     (mids[:, 1] >= y) & (mids[:, 1] < y + hh)).sum())
            if n > best[0]:
                best = (n, x, y)
    _, x, y = best
    return x, y, x + ww, y + hh


def closeup(bg: np.ndarray, sol: dict, box, scale: int = 2) -> np.ndarray:
    X0, Y0, X1, Y1 = box
    s = bg.shape[0] / 2160
    crop = cv2.resize((bg[Y0:Y1, X0:X1].astype(np.float32) * 0.55
                       ).astype(np.uint8), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)
    ids = {d["seg"]: d for d in sol["ids"]}
    placed = []
    for o in sol["objects"]:
        segs = [ids[i] for i in o["segs"]
                if X0 <= (ids[i]["x1"] + ids[i]["x2"]) / 2 < X1
                and Y0 <= (ids[i]["y1"] + ids[i]["y2"]) / 2 < Y1]
        if not segs:
            continue
        c = colour(o)
        for d in segs:
            cv2.line(crop, (int((d["x1"] - X0) * scale), int((d["y1"] - Y0) * scale)),
                     (int((d["x2"] - X0) * scale), int((d["y2"] - Y0) * scale)),
                     c, max(1, int((6 if o["conf"] == "confirmed" else 3) * s)),
                     cv2.LINE_AA)
        d = segs[len(segs) // 2]
        x = int(((d["x1"] + d["x2"]) / 2 - X0) * scale + 12 * s)
        y = int(((d["y1"] + d["y2"]) / 2 - Y0) * scale - 12 * s)
        for _ in range(20):
            if all(abs(x - px) > 90 * s or abs(y - py) > 36 * s
                   for px, py in placed):
                break
            y -= int(36 * s)
        placed.append((x, y))
        _label(crop, f"{o['n']} {o['name'].replace('STARLINK-', 'SL-')}",
               (x, y), 0.8 * s, c, max(1, int(2 * s)))
    return crop


# ---------------------------------------------------------------- map
def _layers(log=print) -> dict | None:
    out = {}
    for k, name in LAYERS.items():
        try:
            out[k] = json.loads(data.fetch(NE + name, name).read_text(
                encoding="utf-8"))
        except OSError as e:
            if log:
                log(f"  map layer {name} unavailable ({e}); no map")
            return None
    return out


def view_wedge(cam, sol: dict, site) -> tuple[float, float, float]:
    """(left, centre, right) azimuths of the camera's view at the solved
    site, from the frame's left/right edge midpoints and centre."""
    k = cam.k0
    # geodetic (ellipsoid normal), not geocentric, up
    phi = math.radians(sol["slat"])
    ra = site.zenith_ra(k) + math.radians(sol["dlst"])
    upd = np.array([math.cos(phi) * math.cos(ra), math.cos(phi) * math.sin(ra),
                    math.sin(phi)]) @ site.P
    up = upd / np.linalg.norm(upd)
    north = site.pole - (site.pole @ up) * up
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    W, H = cam.width, cam.height
    px = np.array([[0, H / 2], [W / 2, H / 2], [W - 1, H / 2]], float)
    v = cam.unproject(px, k)
    az = np.degrees(np.arctan2(v @ east, v @ north)) % 360
    return float(az[0]), float(az[1]), float(az[2])


def site_map(sol: dict, wedge, layers: dict, *, px: int = 1400,
             ) -> np.ndarray:
    lat0, lon0 = sol["lat"], sol["lon"]

    def panel(half_lat, zoom, title):
        kx = math.cos(math.radians(lat0))
        half_lon = half_lat / kx * 1.3
        Wm, Hm = int(px * 1.3), px
        img = np.full((Hm, Wm, 3), (236, 232, 226), np.uint8)

        def P(la, lo):
            dl = (lo - lon0 + 180) % 360 - 180
            return (int((dl + half_lon) / (2 * half_lon) * Wm),
                    int((lat0 + half_lat - la) / (2 * half_lat) * Hm))

        step = 0.25 if zoom else (1 if half_lat < 5 else 5)
        for la in np.arange(math.floor((lat0 - half_lat) / step) * step,
                            lat0 + half_lat + step, step):
            cv2.line(img, P(la, lon0 - half_lon), P(la, lon0 + half_lon),
                     (215, 210, 204), 1)
            cv2.putText(img, f"{abs(la):.2f}{'N' if la >= 0 else 'S'}",
                        (6, P(la, lon0)[1] - 4), FONT, 0.5, (150, 145, 140), 1,
                        cv2.LINE_AA)
        for lo in np.arange(math.floor((lon0 - half_lon) / step) * step,
                            lon0 + half_lon + step, step):
            cv2.line(img, P(lat0 - half_lat, lo), P(lat0 + half_lat, lo),
                     (215, 210, 204), 1)
            cv2.putText(img, f"{abs(lo):.2f}{'E' if lo >= 0 else 'W'}",
                        (P(lat0, lo)[0] + 4, Hm - 8), FONT, 0.5,
                        (150, 145, 140), 1, cv2.LINE_AA)

        def lines(fc, colour_, width):
            for f in fc["features"]:
                g = f["geometry"]
                if g is None:
                    continue
                parts = [g["coordinates"]] if g["type"] == "LineString" \
                    else g["coordinates"]
                for ln in parts:
                    a = np.asarray(ln, float)
                    if not ((abs(a[:, 1] - lat0) < 2.5 * half_lat).any() and
                            (abs((a[:, 0] - lon0 + 180) % 360 - 180)
                             < 2.5 * half_lon).any()):
                        continue
                    pts = np.array([P(la, lo) for lo, la in a], np.int32)
                    cv2.polylines(img, [pts], False, colour_, width,
                                  cv2.LINE_AA)

        lines(layers["borders"], (170, 160, 180), 1)
        lines(layers["countries"], (140, 130, 150), 2)
        lines(layers["coast"], (170, 120, 70), 2)
        towns = []
        for f in layers["places"]["features"]:
            p = f["properties"]
            lo, la = f["geometry"]["coordinates"]
            x, y = P(la, lo)
            if 0 <= x < Wm and 0 <= y < Hm:
                towns.append((p.get("scalerank", 10), p.get("name", ""), x, y))
        towns.sort()
        for _, name, x, y in towns[:14]:
            cv2.circle(img, (x, y), 5, (90, 90, 90), -1, cv2.LINE_AA)
            cv2.putText(img, name, (x + 8, y + 5), FONT, 0.6, (80, 80, 80), 1,
                        cv2.LINE_AA)
        x0, y0 = P(lat0, lon0)
        Lw = int((0.12 if zoom else 0.07) * Wm)
        left, _, right = wedge
        span = (right - left) % 360
        for az in (left, right):
            a = math.radians(az)
            cv2.line(img, (x0, y0), (int(x0 + Lw * math.sin(a)),
                                     int(y0 - Lw * math.cos(a))),
                     (60, 140, 230), 2, cv2.LINE_AA)
        cv2.ellipse(img, (x0, y0), (Lw, Lw), -90, left, left + span,
                    (60, 140, 230), 2, cv2.LINE_AA)
        ax = max(3, int(2 * sol["sig_lon"] / (2 * half_lon) * Wm))
        ay = max(3, int(2 * sol["sig_lat"] / (2 * half_lat) * Hm))
        cv2.ellipse(img, (x0, y0), (ax, ay), 0, 0, 360, (40, 40, 200), 2,
                    cv2.LINE_AA)
        cv2.circle(img, (x0, y0), 7, (30, 30, 210), -1, cv2.LINE_AA)
        ns, ew = ("N" if lat0 >= 0 else "S"), ("E" if lon0 >= 0 else "W")
        cv2.putText(img, f"camera {abs(lat0):.3f}{ns} {abs(lon0):.3f}{ew}",
                    (x0 - 150, y0 + 44), FONT, 0.75, (30, 30, 180), 2,
                    cv2.LINE_AA)
        cv2.putText(img, title, (16, 34), FONT, 0.9, (60, 60, 60), 2,
                    cv2.LINE_AA)
        return img

    a = panel(3.2, False, "Blind site solve from satellite trails "
                          "(2-sigma ellipse, view wedge)")
    b = panel(0.35, True, "Zoom: +-0.35 deg")
    return np.hstack([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b])


def write(wd, *, map: bool = True, log=print) -> list[Path]:
    """All deliverables into the work folder; returns their paths."""
    from remove_satellites.core import video

    from .site import Site
    sol = wd.read("solution.json")
    info = wd.info
    cam = wd.camera()
    stem = wd.clip.stem
    out = []
    k = int(np.median([d["k"] for d in sol["ids"]])) if sol["ids"] else cam.k0
    bg = wd.cleaned_frame(k)
    dated = dating(wd, k)
    img = trails_image(bg, sol, f"Identified satellite trails - {stem}", dated)
    p = wd.path(f"{stem}_satellites.jpg")
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out.append(p)
    box = busiest(sol, info["width"], info["height"])
    p = wd.path(f"{stem}_satellites_closeup.jpg")
    cv2.imwrite(str(p), closeup(bg, sol, box), [cv2.IMWRITE_JPEG_QUALITY, 90])
    out.append(p)
    p = wd.path(f"{stem}_satellites.json")
    p.write_text(json.dumps(dict(
        site=dict(lat=sol["lat"], lon=sol["lon"],
                  height_km=sol.get("height_km"), sig_lat=sol["sig_lat"],
                  sig_lon=sol["sig_lon"]),
        frame0_utc=sol["frame0_utc"], exposure_s=sol["E"],
        interval_s=sol["spf"], null_rate=sol["null_rate"],
        dated_by=[dict(name=o["name"], kind=o["kind"]) for o in dated],
        elapsed_s=sol.get("elapsed_s"), timing=sol.get("timing"),
        objects=sol["objects"]), indent=1))
    out.append(p)
    layers = _layers(log) if map else None
    if layers:
        hz = wd.read("horizon.json")
        t_mid = sol["frame0_tt"] + (float(cam.s(info["n_frames"] - 1)) + 1)             * sol["spf"] / 2 / 86400
        site = Site(cam, hz["up_cam"], t_mid, info["n_frames"])
        m = site_map(sol, view_wedge(cam, sol, site), layers)
        p = wd.path(f"{stem}_site_map.png")
        cv2.imwrite(str(p), m)
        out.append(p)
    return out
