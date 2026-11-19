"""Which way is up: the skyline gives the camera's up vector, and with the
sky's rotation axis that is the latitude and each frame's sidereal time.

Skyline = the sky-to-land edge. The foreground mask's top edge sits a little
above the real ridge (closing + blur), so each column looks for the lowest
strong bright-to-dark drop just below the mask top in a smooth (cleaned)
frame. Trees and peaks poke above the far horizon, so a plane through the
camera centre is fitted robustly to the skyline rays, keeping the *lowest*
consistent line (the distant ridge dominates the lower envelope).

The result is only a start: a ridge sits a degree or two above the true
horizon, so latitude and sidereal time from here can be off by that much;
the site/time solve treats them as rough and searches around them.
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera


def skyline(fg_mask: np.ndarray, gray: np.ndarray, *, step: int = 8,
            below_frac: float = 0.075, min_drop: float = 8.0):
    """(x, y) pixels of the sky-to-land edge, one per `step` columns."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    h, w = g.shape
    below = max(20, int(below_frac * h))
    cols, ys_out = [], []
    for c in range(0, w, step):
        col = fg_mask[:, c] > 0
        if not col.any() or col[0]:
            continue
        y0 = int(np.argmax(col))
        ys = np.arange(max(y0 - 10, 6), min(y0 + below, h - 6))
        if not ys.size:
            continue
        drop = g[ys - 5, c] - g[ys + 5, c]
        if drop.max() <= min_drop:
            continue
        strong = np.flatnonzero(drop > 0.5 * drop.max())
        cols.append(c)
        ys_out.append(ys[strong.max()])
    return np.array(cols, float), np.array(ys_out, float)


def up_vector(cam: Camera, xs: np.ndarray, ys: np.ndarray, *, iters=4000,
              inlier_deg=0.25, below_deg=0.5, seed=0):
    """Camera-frame up vector from skyline pixels -> (up, inlier mask)."""
    R = cam.px_to_ray(np.column_stack([xs, ys]))
    rng = np.random.default_rng(seed)
    best = None
    tin, tbelow = np.radians(inlier_deg), np.radians(below_deg)
    for _ in range(iters):
        i, j = rng.choice(len(R), 2, replace=False)
        u = np.cross(R[i], R[j])
        n = np.linalg.norm(u)
        if n < 1e-6:
            continue
        u /= n
        if u[1] > 0:                        # up points towards image -y
            u = -u
        d = R @ u                           # > 0 above the plane
        inl = np.abs(d) < tin
        score = inl.sum() - 5 * (d < -tbelow).sum()
        if best is None or score > best[0]:
            best = (score, u, inl)
    _, u, inl = best
    _, _, vt = np.linalg.svd(R[inl])
    u = vt[-1]
    if u[1] > 0:
        u = -u
    return u, inl


def latitude(cam: Camera, up: np.ndarray) -> float:
    """90 deg minus the angle between up and the sky's rotation axis."""
    pole = cam.M @ cam.axis
    return float(90 - np.degrees(np.arccos(np.clip(up @ pole, -1, 1))))


def line_segments(gray: np.ndarray, min_len: float) -> np.ndarray | None:
    """(N, 4) straight edges: OpenCV's line segment detector, or where a
    build lacks it (the phone's OpenCV 4.5.1, left out over its licence)
    Canny edges and a probabilistic Hough transform."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    try:
        segs = cv2.createLineSegmentDetector(0).detect(blur)[0]
    except cv2.error:
        edges = cv2.Canny(blur.astype(np.uint8), 40, 120)
        segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=30,
                               minLineLength=max(int(min_len), 10),
                               maxLineGap=3)
    return None if segs is None else segs.reshape(-1, 4).astype(np.float32)


def verticals(cam: Camera, gray: np.ndarray, fg_mask: np.ndarray, *,
              max_tilt_deg: float = 25.0, min_len_frac: float = 0.04,
              iters: int = 3000, inlier_deg: float = 0.3, seed: int = 0):
    """Up from vertical structures: towers, poles, rockets, walls. Their
    edges are parallel in the world, so in the camera they all lie on
    planes through one direction — the zenith (their vanishing point).
    -> (up, number of agreeing edges) or (None, 0)."""
    h, w = gray.shape
    segs = line_segments(gray, min_len_frac * h)
    if segs is None:
        return None, 0
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    tilt = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    land = cv2.dilate(fg_mask, np.ones((31, 31), np.uint8)) > 0
    mx = np.clip(((segs[:, 0] + segs[:, 2]) / 2).astype(int), 0, w - 1)
    my = np.clip(((segs[:, 1] + segs[:, 3]) / 2).astype(int), 0, h - 1)
    keep = (length > min_len_frac * h) & (tilt < max_tilt_deg) & land[my, mx]
    segs = segs[keep]
    if len(segs) < 6:
        return None, 0
    r1, r2 = cam.px_to_ray(segs[:, :2]), cam.px_to_ray(segs[:, 2:])
    n = np.cross(r1, r2)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    tin = np.sin(np.radians(inlier_deg))
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(n), 2, replace=False)
        v = np.cross(n[i], n[j])
        if np.linalg.norm(v) < 1e-6:
            continue
        v /= np.linalg.norm(v)
        inl = np.abs(n @ v) < tin
        if best is None or inl.sum() > best[1].sum():
            best = (v, inl)
    v, inl = best
    _, _, vt = np.linalg.svd(n[inl])
    v = vt[-1]
    if v[1] > 0:                             # up points towards image -y
        v = -v
    return v, int(inl.sum())


def sky_rays(cam: Camera, sky: np.ndarray | None = None, n: int = 9
             ) -> np.ndarray:
    """Camera rays on an n x n grid over the frame where `sky` (mask, 1 =
    sky; default everywhere) is sky: whatever the site, these were above
    the horizon. Only points clearly above the land in their own column
    count (a margin of a twentieth of the frame): the land is below the
    skyline, so a patch the foreground mask missed under it — a dark ridge,
    a telescope's shadowed side — is ground, and may be below the horizon."""
    xs, ys = np.meshgrid(np.linspace(0, cam.width - 1, n),
                         np.linspace(0, cam.height - 1, n))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    if sky is not None:
        sy, sx = sky.shape[:2]
        land = sky == 0
        top = np.where(land.any(0), land.argmax(0), sy)   # first land row
        cx = (xy[:, 0] * (sx - 1) / (cam.width - 1)).astype(int)
        cy = (xy[:, 1] * (sy - 1) / (cam.height - 1)).astype(int)
        keep = cy < top[cx] - sy // 20
        xy = xy[keep] if keep.any() else xy
    return cam.px_to_ray(xy)


def solve(cam: Camera, fg_mask: np.ndarray, gray: np.ndarray, log=print
          ) -> dict:
    """Up from the skyline, or — when the skyline is a jumble (trees,
    buildings) and there are vertical structures — from their vanishing
    point. With neither (pure sky) `up_cam` is None: the solve then finds
    the site from the satellites alone (`solve.free_passes`)."""
    xs, ys = skyline(fg_mask, gray)
    sky_up = inl = None
    if len(xs) >= 20:
        sky_up, inl = up_vector(cam, xs, ys)
    vert_up, n_vert = verticals(cam, gray, fg_mask)
    # a few columns of foreground speckle make no skyline
    weak_skyline = sky_up is None or inl.sum() < max(20, 0.3 * len(xs))
    if vert_up is not None and n_vert >= 8 and weak_skyline:
        up, source = vert_up, f"vertical structures ({n_vert} edges)"
    elif not weak_skyline:
        up, source = sky_up, (f"skyline ({int(inl.sum())} of {len(xs)} "
                              "columns on the line)")
    else:
        if log:
            log("  no skyline or vertical structures: the site comes from "
                "the satellites alone")
        return dict(up_cam=None, lat=None, roll_deg=None, source="none",
                    n_columns=int(len(xs)),
                    n_inliers=int(inl.sum()) if inl is not None else 0,
                    n_verticals=n_vert, skyline=[])
    lat = latitude(cam, up)
    roll = float(np.degrees(np.arctan2(up[0], -up[1])))
    if log:
        log(f"  up from the {source}; roll {roll:+.2f} deg; latitude "
            f"{lat:.2f} deg (rough)")
    return dict(up_cam=up.tolist(), lat=lat, roll_deg=roll, source=source,
                n_columns=int(len(xs)),
                n_inliers=int(inl.sum()) if inl is not None else 0,
                n_verticals=n_vert,
                skyline=np.column_stack([xs, ys]).round(1).tolist())
