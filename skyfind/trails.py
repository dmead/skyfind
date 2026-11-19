"""Trail segments from what the cleaner removed, linked across frames.

Per frame: the residual (original - cleaned) above the sky, foreground
masked, thresholded; elongated connected components become segments
(principal-axis end points). Aircraft carry coloured, blinking nav lights:
`chroma` (channel spread over mean) and `blink` (brightness variation along
the segment) separate them from steady, neutral satellites.

An end within a few pixels of the frame edge or the skyline was cut short
there, not by the exposure: `clip_a` / `clip_b` flag it so the solver
doesn't read an exposure boundary into it.

Across frames: an object's consecutive exposures give collinear segments
whose ends nearly meet (the gap between exposures is short), so a segment in
frame k+1 joins the track whose last segment it continues.
"""

from __future__ import annotations

import cv2
import numpy as np


class Extractor:
    def __init__(self, fg_mask: np.ndarray | None, shape, *, thresh=18.0,
                 min_area=25, min_len=25.0, min_aspect=5.0, edge_px=6,
                 skyline_px=14):
        h, w = shape
        fg = np.zeros((h, w), np.uint8) if fg_mask is None else \
            (fg_mask > 0).astype(np.uint8)
        self.sky = cv2.dilate(fg, np.ones((15, 15), np.uint8)) == 0
        near = cv2.dilate(fg, np.ones((2 * skyline_px + 15,) * 2, np.uint8))
        near[:edge_px], near[-edge_px:] = 1, 1
        near[:, :edge_px], near[:, -edge_px:] = 1, 1
        self.cut = near > 0
        self.thresh, self.min_area = thresh, min_area
        self.min_len, self.min_aspect = min_len, min_aspect

    def __call__(self, original: np.ndarray, cleaned: np.ndarray) -> list[dict]:
        rgb = original.astype(np.int16) - cleaned.astype(np.int16)
        res = rgb.max(2).astype(np.float32) if rgb.ndim == 3 else \
            rgb.astype(np.float32)
        if rgb.ndim == 2:
            rgb = rgb[..., None].repeat(3, 2)
        return self.segments(res, rgb)

    def segments(self, res: np.ndarray, rgb: np.ndarray) -> list[dict]:
        m = ((res > self.thresh) & self.sky).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        h, w = res.shape
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < self.min_area:
                continue
            x0, y0 = st[i, 0], st[i, 1]
            ys, xs = np.nonzero(lab[y0:y0 + st[i, 3], x0:x0 + st[i, 2]] == i)
            xs, ys = xs + x0, ys + y0
            pts = np.column_stack([xs, ys]).astype(np.float64)
            c = pts.mean(0)
            _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
            d = vt[0]
            t = (pts - c) @ d
            length = t.max() - t.min()
            width = 4 * s[1] / np.sqrt(len(pts))
            if length < self.min_len or \
                    length < self.min_aspect * max(width, 1.0):
                continue
            t0, t1 = _half_max_ends(res, c, d, t.min(), t.max(), width)
            a, b = c + t0 * d, c + t1 * d
            wgt = res[ys, xs]
            ch = np.clip(rgb[ys, xs].astype(np.float64), 0, None).mean(0)
            chroma = float((ch.max() - ch.min()) / max(ch.mean(), 1e-6))
            nb = max(4, int(length // 20))
            bins = np.clip(((t - t.min()) / max(length, 1e-6) * nb).astype(int),
                           0, nb - 1)
            prof = np.bincount(bins, weights=wgt, minlength=nb) / \
                np.maximum(np.bincount(bins, minlength=nb), 1)
            blink = float(prof.std() / max(prof.mean(), 1e-6))

            def cut(p):
                x, y = int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1))
                return bool(self.cut[y, x])

            out.append(dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]),
                            y2=float(b[1]), len=float(length),
                            width=float(width), flux=float(wgt.sum()),
                            peak=float(np.percentile(wgt, 90)),
                            b=float(ch[0]), g=float(ch[1]), r=float(ch[2]),
                            chroma=chroma, blink=blink,
                            clip_a=cut(a), clip_b=cut(b)))
        return out


def join_pieces(segs: list[dict], land: np.ndarray, *,
                collinear: float = 0.995, max_off: float = 4.0,
                over_land: float = 0.5) -> list[dict]:
    """One frame's segments with the pieces of a trail rejoined: a trail
    crossing behind a tree (a dolly clip's) comes out as pieces either side of
    it, each cut short where the land hides it. Two pieces join when
    they're collinear and in line, their facing ends are both cut, and
    the gap between those lies mostly on land; the joined segment keeps
    the outer ends and their flags, so a trail that went behind the tree
    still shows where its exposure began and ended."""
    near = cv2.dilate(land.astype(np.uint8), np.ones((31, 31), np.uint8))
    h, w = land.shape
    segs = [dict(s) for s in segs]

    def ends(s):
        return (np.array([s["x1"], s["y1"]]), np.array([s["x2"], s["y2"]]))

    def mergeable(s, t):
        a1, b1 = ends(s)
        a2, b2 = ends(t)
        d = (b1 - a1) / max(np.linalg.norm(b1 - a1), 1e-9)
        e = (b2 - a2) / max(np.linalg.norm(b2 - a2), 1e-9)
        if abs(d @ e) < collinear:
            return None
        nrm = np.array([-d[1], d[0]])
        if max(abs((a2 - a1) @ nrm), abs((b2 - a1) @ nrm)) > \
                max_off + (s["width"] + t["width"]) / 2:
            return None
        # in line along d: s's span, then a gap, then t's (or the reverse)
        ts = sorted([(0.0, 0, "a"), ((b1 - a1) @ d, 0, "b"),
                     ((a2 - a1) @ d, 1, "a"), ((b2 - a1) @ d, 1, "b")])
        if ts[0][1] != ts[1][1]:
            return None                           # overlapping pieces
        (_, i_out1, e_out1), (_, i_in1, e_in1), (_, i_in2, e_in2), \
            (_, i_out2, e_out2) = ts
        pieces = (s, t)
        if not (pieces[i_in1]["clip_" + e_in1]
                and pieces[i_in2]["clip_" + e_in2]):
            return None
        p = np.array([pieces[i_in1]["x" + ("1" if e_in1 == "a" else "2")],
                      pieces[i_in1]["y" + ("1" if e_in1 == "a" else "2")]])
        q = np.array([pieces[i_in2]["x" + ("1" if e_in2 == "a" else "2")],
                      pieces[i_in2]["y" + ("1" if e_in2 == "a" else "2")]])
        u = np.linspace(0, 1, max(3, int(np.linalg.norm(q - p) // 4)))
        xy = p[None] + u[:, None] * (q - p)[None]
        xs = np.clip(xy[:, 0].astype(int), 0, w - 1)
        ys = np.clip(xy[:, 1].astype(int), 0, h - 1)
        if near[ys, xs].mean() < over_land:
            return None
        return (pieces[i_out1], e_out1), (pieces[i_out2], e_out2)

    def end_of(s, e):
        n = "1" if e == "a" else "2"
        return s["x" + n], s["y" + n], s["clip_" + e]

    merged = True
    while merged:
        merged = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                m = mergeable(segs[i], segs[j])
                if m is None:
                    continue
                s, t = segs[i], segs[j]
                (x1, y1, c1), (x2, y2, c2) = end_of(*m[0]), end_of(*m[1])
                fl = s["flux"] + t["flux"]
                ch = np.array([(s[c] * s["flux"] + t[c] * t["flux"])
                               / max(fl, 1e-9) for c in "bgr"])
                ln = s["len"] + t["len"]
                segs[i] = dict(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    len=float(np.hypot(x2 - x1, y2 - y1)),
                    width=(s["width"] * s["len"] + t["width"] * t["len"]) / ln,
                    flux=fl, peak=max(s["peak"], t["peak"]),
                    b=float(ch[0]), g=float(ch[1]), r=float(ch[2]),
                    chroma=float((ch.max() - ch.min()) / max(ch.mean(), 1e-6)),
                    blink=(s["blink"] * s["len"] + t["blink"] * t["len"]) / ln,
                    clip_a=bool(c1), clip_b=bool(c2),
                    pieces=s.get("pieces", 1) + t.get("pieces", 1))
                del segs[j]
                merged = True
                break
            if merged:
                break
    return segs


def _half_max_ends(res, c, d, t_lo, t_hi, width, pad=4.0):
    """Trail ends where the brightness along the trail falls to half its
    plateau. The thresholded blob runs past the true ends by however far
    the blurred edge stays above the threshold (more for bright trails);
    half maximum is where a blurred step's edge really is."""
    ts = np.arange(t_lo - pad, t_hi + pad + 0.25, 0.5)
    n = np.array([-d[1], d[0]])
    offs = np.linspace(-max(width, 1.0), max(width, 1.0), 5)
    pts = c + ts[:, None, None] * d + offs[None, :, None] * n   # (T, O, 2)
    prof = cv2.remap(res.astype(np.float32),
                     pts[..., 0].astype(np.float32),
                     pts[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=0.0).mean(1)
    inner = (ts > t_lo + 0.2 * (t_hi - t_lo)) & (ts < t_hi - 0.2 * (t_hi - t_lo))
    if inner.sum() < 3:
        return t_lo, t_hi
    half = 0.5 * float(np.median(prof[inner]))
    above = prof >= half
    first = int(np.argmax(above))
    last = len(prof) - 1 - int(np.argmax(above[::-1]))
    if first == 0 or last == len(prof) - 1 or half <= 0:
        return t_lo, t_hi                   # runs off the sampled stretch

    def cross(i, j):                        # linear crossing between samples
        p, q = prof[i], prof[j]
        return ts[i] + (ts[j] - ts[i]) * (half - p) / (q - p) if q != p \
            else ts[j]

    return float(cross(first - 1, first)), float(cross(last + 1, last))


def link(segs: dict[int, list[dict]], *, max_gap=2, collinear=0.97,
         max_offset=15.0) -> list[dict]:
    """Chain segments of consecutive frames into tracks.

    `segs`: frame -> segments. A segment continues a track whose last
    segment (at most `max_gap` frames back) is collinear with it, within
    `max_offset` px of its line, with nearest ends less than half a segment
    length + 40 px apart. Returns [{frames: [...], segs: [...]}]."""
    # frames in order; a clip re-timed by repeating frames keeps only one
    # frame per capture (`captures`), and the gaps count in captures
    frames = sorted(int(k) for k in segs)
    by_frame = {int(k): v for k, v in segs.items()}
    tracks, open_tracks = [], []
    for r, k in enumerate(frames):
        tracks += [t for t in open_tracks if r - t[-1][0] > max_gap]
        open_tracks = [t for t in open_tracks if r - t[-1][0] <= max_gap]
        claimed = set()
        for tr in open_tracks:
            kk, s = tr[-1]
            if kk >= r:
                continue
            p1, p2 = np.array([s["x1"], s["y1"]]), np.array([s["x2"], s["y2"]])
            d = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-9)
            best = None
            for j, q in enumerate(by_frame[k]):
                if j in claimed:
                    continue
                q1, q2 = np.array([q["x1"], q["y1"]]), np.array([q["x2"], q["y2"]])
                e = (q2 - q1) / max(np.linalg.norm(q2 - q1), 1e-9)
                if abs(d @ e) < collinear:
                    continue
                mid = (q1 + q2) / 2
                off = abs(d[0] * (mid - p1)[1] - d[1] * (mid - p1)[0])
                gap = min(np.linalg.norm(q1 - p2), np.linalg.norm(q2 - p1),
                          np.linalg.norm(q1 - p1), np.linalg.norm(q2 - p2))
                if off < max_offset and gap < 0.5 * s["len"] + 40:
                    if best is None or gap < best[0]:
                        best = (gap, j)
            if best:
                claimed.add(best[1])
                tr.append((r, by_frame[k][best[1]]))
        for j, q in enumerate(by_frame[k]):
            if j not in claimed:
                open_tracks.append([(r, q)])
    tracks += open_tracks
    out = [dict(frames=[frames[r] for r, _ in tr], segs=[s for _, s in tr])
           for tr in tracks]
    out.sort(key=lambda t: t["frames"][0])
    return out


def captures(clock):
    """A clip re-timed by repeating frames (a 25 fps timelapse put out at
    60 fps): the tracker's clock stands still over a repeat and moves by a
    fraction over a blended frame. -> (the first frame of each capture, a
    clock in whole captures); (None, clock) when every frame is a capture
    of its own.

    A whole step starts a capture. A blended frame is an extra frame
    between captures: the steps into and out of it can read a half and a
    whole, so fractions only count once several in a row make a whole (a
    run of true 50/50 blends)."""
    c = np.asarray(clock, float)
    steps = np.diff(c)
    if not len(steps) or (steps >= 1 - 1e-6).all():
        return None, clock
    keep, out, n, frac = [0], [0.0], 0, 0.0
    for k, st in enumerate(steps, 1):
        if st >= 0.75:
            n += int(round(st))
            frac = 0.0
            keep.append(k)
        elif st >= 0.25:
            frac += st
            if frac >= 0.75:
                n += 1
                frac = 0.0
                keep.append(k)
        out.append(float(n))
    return keep, out


# ---------------------------------------------------------------- clutter
CLUTTER_CELL = 24                 # px
CLUTTER_RUN = 4                   # consecutive frames lighting one spot


def clutter(segs: dict, shape, *, cell: int = CLUTTER_CELL,
            run: int = CLUTTER_RUN, grow: int = 2) -> np.ndarray:
    """Spots lit frame after frame: hardware (a telescope tracking or
    slewing, a dome turning), a lamp, a skyline or trees flickering. A
    satellite never lights the same spot in consecutive frames — each
    exposure starts where the last one ended, and the next satellite of a
    train comes by minutes later — so a spot the residual lights in `run`
    consecutive frames isn't sky traffic. -> bool grid of `cell`-px spots,
    grown by `grow` to take in the fringes."""
    h, w = shape
    cur = np.zeros((h // cell + 1, w // cell + 1), np.int32)
    lit = np.zeros(cur.shape, bool)
    by_frame = {int(k): v for k, v in segs.items()}
    for k in sorted(by_frame):               # one frame per capture, in order
        hit = np.zeros(cur.shape, bool)
        for s in by_frame.get(k, ()):
            gy, gx = _cells(s, cell, h, w)
            hit[gy, gx] = True
        cur = np.where(hit, cur + 1, 0)
        lit |= cur >= run
    return cv2.dilate(lit.astype(np.uint8),
                      np.ones((2 * grow + 1,) * 2, np.uint8)) > 0


def _cells(s: dict, cell: int, h: int, w: int):
    n = int(max(abs(s["x2"] - s["x1"]), abs(s["y2"] - s["y1"])) / (cell / 2)) + 2
    xs = np.clip(np.linspace(s["x1"], s["x2"], n), 0, w - 1)
    ys = np.clip(np.linspace(s["y1"], s["y2"], n), 0, h - 1)
    return (ys // cell).astype(int), (xs // cell).astype(int)


def in_clutter(s: dict, grid: np.ndarray, *, cell: int = CLUTTER_CELL,
               frac: float = 0.3) -> bool:
    """Whether at least `frac` of a segment lies in cluttered patches."""
    h, w = grid.shape[0] * cell, grid.shape[1] * cell
    gy, gx = _cells(s, cell, h, w)
    return bool(grid[gy, gx].mean() >= frac)
