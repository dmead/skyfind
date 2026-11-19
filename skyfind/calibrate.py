"""One camera model for the whole clip, from the stars.

The tripod is fixed, so every frame shares one lens (f, principal point,
radial distortion) and one orientation, and the sky turns about the pole of
date at a constant rate per frame (`camera.Camera`). Two plate solves some
frames apart give the start: focal length from their scale, orientation from
their matched stars, the rate from the rotation between them. Then iterate:
predict catalogue stars in sampled frames, match to detections, least-squares
refit, tighten the tolerance, free more parameters.

Stars alone can't separate latitude from camera tilt (only the net rotation
is observable); that split comes from the horizon (`horizon.py`).

A camera on a motion-control rig turns as the clip runs: `follow` then
fits one lens and an orientation per sampled frame instead, and
`motion.py` splits each orientation into the Earth's turn and the
camera's (the land, kilometres away, holds still).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .camera import Camera, kabsch, radec_to_vec, rotate, unit
from .catalog import Stars
from .platesolve import PlateSolution, pinhole_project, pinhole_rays


def detect(frame: np.ndarray, sky_mask: np.ndarray | None = None, *,
           max_stars: int = 2500) -> np.ndarray:
    """(N, 3) [x, y, flux] star detections off the foreground. The land is
    blanked to the sky's median first, so lights or snow on it can't take
    the brightest-N places from stars."""
    from remove_satellites.core import stars
    if sky_mask is not None and (sky_mask == 0).any():
        frame = frame.copy()
        land = sky_mask == 0
        frame[land] = np.median(frame[~land], axis=0).astype(frame.dtype)
    d = stars.detect(frame, max_stars=max_stars, max_area=3000)
    if sky_mask is not None and len(d):
        d = d[sky_mask[d[:, 1].astype(int), d[:, 0].astype(int)] > 0]
    return d


def initial_camera(a: PlateSolution, b: PlateSolution, width: int,
                   height: int, k0: int, clock=None) -> Camera:
    """Camera from two plate solves at frames a.frame < b.frame (`clock`:
    capture index per frame, when a re-timed clip dropped frames)."""
    s = Camera(np.zeros(12), k0, width, height, clock=clock).s
    f0 = float(np.sqrt(a.scale * b.scale))
    c = np.array([width / 2, height / 2])

    def orient(sol):
        ray = unit(np.column_stack([(sol.xy - c) / f0, np.ones(len(sol.xy))]))
        return kabsch(sol.vec, ray)

    Ma, Mb = orient(a), orient(b)
    rv = Rotation.from_matrix(Ma.T @ Mb).as_rotvec()
    omega = -np.degrees(rv[2]) / (s(b.frame) - s(a.frame))
    z = np.array([0, 0, 1.0])
    # M at k0 = Ma R(z, -omega (k0 - ka)), applied to row vectors
    Rk = np.stack([rotate(e, z, -np.radians(omega) * (s(k0) - s(a.frame)))
                   for e in np.eye(3)], 1)
    M = Ma @ Rk
    p = np.r_[Rotation.from_matrix(M).as_rotvec(), omega, f0, c, 0, 0, 0, 0]
    return Camera(p, k0, width, height, clock=clock)


class _Matcher:
    def __init__(self, stars: Stars, dets: dict[int, np.ndarray],
                 sky_mask: np.ndarray | None, maglim: float):
        keep = stars.mag <= maglim
        self.v = radec_to_vec(stars.ra[keep], stars.dec[keep])
        self.mag = stars.mag[keep]
        self.dets = {k: d[:, :2] for k, d in dets.items() if len(d)}
        self.trees = {k: cKDTree(d) for k, d in self.dets.items()}
        self.sky = sky_mask

    def gather(self, cam: Camera, frames, tol: float):
        ks, vs, obs = [], [], []
        for k in frames:
            if k not in self.trees:
                continue
            pred, z = cam.project(self.v, k)
            ok = cam.in_frame(pred, z, 5) & (z > 0.2)
            idx = np.flatnonzero(ok)
            if self.sky is not None and idx.size:
                idx = idx[self.sky[pred[idx, 1].astype(int),
                                   pred[idx, 0].astype(int)] > 0]
            if not idx.size:
                continue
            dist, j = self.trees[k].query(pred[idx])
            good = dist < tol
            order = np.argsort(self.mag[idx[good]])     # brightest claims first
            used = set()
            for i_, j_ in zip(idx[good][order], j[good][order]):
                if j_ in used:
                    continue
                used.add(j_)
                ks.append(k)
                vs.append(self.v[i_])
                obs.append(self.dets[k][j_])
        return np.array(ks), np.array(vs).reshape(-1, 3), \
            np.array(obs).reshape(-1, 2)


def _fit(cam: Camera, ks, vs, obs, free) -> Camera:
    free = list(free)

    def res(q):
        p = cam.p.copy()
        p[free] = q
        return (cam.with_params(p).project(vs, ks)[0] - obs).ravel()

    r = least_squares(res, cam.p[free], loss="soft_l1", f_scale=2.0,
                      x_scale="jac")
    p = cam.p.copy()
    p[free] = r.x
    return cam.with_params(p)


ORIENT, RATE, LENS, AXIS, REFR = [0, 1, 2], [3], [4, 5, 6, 7, 8], [9, 10], [11]


def _match_near(Q, f, c, V, mag, det_xy, tree, width, height, tols):
    """Catalogue stars predicted with orientation Q (pinhole f, c) matched
    to detections at tightening tolerances, Q refitted at each ->
    (Q, catalogue idx, detection idx, stars predicted)."""
    ci = dj = np.zeros(0, int)
    n_pred = 0
    for tol in tols:
        pred, z = pinhole_project(Q, f, c, V)
        ok = (z > 0.2) & (pred[:, 0] > 0) & (pred[:, 0] < width) & \
            (pred[:, 1] > 0) & (pred[:, 1] < height)
        idx = np.flatnonzero(ok)
        if idx.size < 6:
            break
        d, j = tree.query(pred[idx])
        m = d < tol
        order = np.argsort(mag[idx[m]])          # brightest claims first
        cand_c, cand_d = idx[m][order], j[m][order]
        _, first = np.unique(cand_d, return_index=True)
        ci, dj = cand_c[np.sort(first)], cand_d[np.sort(first)]
        n_pred = int(idx.size)
        if len(ci) < 6:
            break
        Q = kabsch(V[ci], pinhole_rays(det_xy[dj], f, c))
    return Q, ci, dj, n_pred


def track(seed: PlateSolution, dets: dict[int, np.ndarray], stars: Stars,
          width: int, height: int, *, maglim: float = 5.5,
          verify_dets: int = 600, blind=None, predict=None, log=print
          ) -> dict[int, PlateSolution]:
    """Plate solutions for every sampled frame, walking out from `seed`: a
    moving camera's next orientation is the last one carried on by the
    turn so far, then matched to that frame's stars at tightening
    tolerances; the first step from the seed takes the turn `predict(k)`
    (a fitted camera's orientation at frame k) says, when given. A match
    must beat chance as a blind solve's must (6 sigma plus 8 over what the
    detection density gives). Where tracking loses
    the stars, `blind(k)` (a blind solve) may pick them up again; two
    misses in a row end the walk (twilight, cloud)."""
    keep = stars.mag <= maglim
    V = radec_to_vec(stars.ra[keep], stars.dec[keep])
    mag = stars.mag[keep]
    f = seed.scale
    c = np.array([width / 2, height / 2])
    Q0 = seed.M if seed.M is not None else \
        kabsch(seed.vec, pinhole_rays(seed.xy, f, c))
    out = {seed.frame: seed}
    ks = sorted(k for k in dets if len(dets[k]))
    i0 = ks.index(seed.frame)
    for step in (1, -1):
        hist = [(seed.frame, Q0)]
        misses = 0
        i = i0 + step
        while 0 <= i < len(ks) and misses < 2:
            k = ks[i]
            i += step
            (k1, Q1) = hist[-1]
            Qp = Q1
            if len(hist) > 1:                   # carry the turn on
                k2, Q2 = hist[-2]
                rv = Rotation.from_matrix(Q1 @ Q2.T).as_rotvec()
                Qp = Rotation.from_rotvec(rv * (k - k1) / (k1 - k2)
                                          ).as_matrix() @ Q1
            elif predict is not None:
                Qp = predict(k) @ predict(k1).T @ Q1
            det_xy = dets[k][:verify_dets, :2].astype(float)
            tree = cKDTree(det_xy)
            Q, ci, dj, n_pred = _match_near(Qp, f, c, V, mag, det_xy, tree,
                                            width, height, (40, 15, 6, 4))
            p_hit = min(1.0, len(det_xy) * np.pi * 4.0 ** 2 /
                        (width * height))
            chance = n_pred * p_hit
            if len(ci) >= max(20, chance + 6 * np.sqrt(chance) + 8):
                out[k] = PlateSolution(k, det_xy[dj], V[ci], mag[ci], [], f,
                                       int(len(ci)), Q)
                hist.append((k, Q))
                misses = 0
                continue
            s = blind(k) if blind else None
            if s is not None:
                out[k] = s
                hist.append((k, s.M if s.M is not None else
                             kabsch(s.vec, pinhole_rays(s.xy, f, c))))
                misses = 0
            else:
                misses += 1
    if log:
        log(f"  stars followed through {len(out)} of {len(ks)} sampled "
            f"frames ({min(out)}-{max(out)})")
    return dict(sorted(out.items()))


def fixed_misfit(sols: dict[int, PlateSolution], width: int, height: int,
                 clock=None) -> dict[int, float]:
    """How far each frame's own orientation (followed stars, pinhole) is
    from the best fixed camera — one pose, the sky turning at one rate
    about the celestial pole: {frame: deg}. Under half a degree for a
    tripod (the pinhole ignores the lens's distortion; the pole of date is
    J2000's to 0.3 deg); degrees for a camera on a motion-control rig.
    The axis is pinned: a steady pan and the Earth's turn together look
    like a turn about a tilted axis, which a free axis would take up
    (a dolly clip: 0.6 deg free, 6.8 pinned)."""
    ks = sorted(sols)
    s_of = Camera(np.zeros(12), ks[0], width, height, clock=clock).s
    sk = np.array([float(s_of(k)) for k in ks])
    c = np.array([width / 2, height / 2])
    Q = np.array([sols[k].M if sols[k].M is not None else
                  kabsch(sols[k].vec, pinhole_rays(sols[k].xy,
                                                   sols[k].scale, c))
                  for k in ks])
    k0 = len(ks) // 2
    z = np.array([0, 0, 1.0])

    def res(q):
        M = Rotation.from_rotvec(q[:3]).as_matrix()
        R = Rotation.from_rotvec(np.outer(q[3] * (sk - sk[k0]), z)).as_matrix()
        return Rotation.from_matrix(
            np.einsum("nji,jk,nkl->nil", Q, M, R)).as_rotvec().ravel()

    rv0 = Rotation.from_matrix(Q[0].T @ Q[-1]).as_rotvec()
    w0 = rv0[2] / (sk[-1] - sk[0]) if sk[-1] > sk[0] else 0.0
    best = None
    for w in (w0, -w0):
        r = least_squares(res, np.r_[Rotation.from_matrix(Q[k0]).as_rotvec(),
                                     w], loss="soft_l1",
                          f_scale=np.radians(0.05))
        if best is None or r.cost < best.cost:
            best = r
    e = np.degrees(np.linalg.norm(res(best.x).reshape(-1, 3), axis=1))
    return {k: float(v) for k, v in zip(ks, e)}


# where the frame-by-frame fit starts: from the tracker's clean pairs,
# and from a wide first match at a few focal lengths (x the plate solves')
FOLLOW_STARTS = [("pairs", 1.0)] + [(8, m) for m in (0.97, 1.0, 1.03)]


def follow(sols: dict[int, PlateSolution], dets: dict[int, np.ndarray],
           stars: Stars, width: int, height: int, *, sky_mask=None,
           maglim: float = 5.5, log=print):
    """A moving camera, frame by frame: one lens (f, principal point, k1,
    k2) and each plate-solved sampled frame's own orientation, fitted to
    the stars together. `sols`: plate solutions per frame. Returns
    ({frame: Q (GCRS -> camera)}, lens camera (a Camera whose `ray_to_px`
    / `px_to_ray` are the lens), stats).

    The fit has more than one basin: a wide lens's plate-solve scale is
    off by its distortion, and started there it can settle on the wrong
    focal length with the distortion bent to suit (a dolly clip: 817 px and k1
    +0.011 from 825, against 811 and -0.003 from 814 — 0.9 deg apart at
    the frame's edge). So it's started more than one way — from the
    tracker's own matched pairs (clean, but central), and from a wide
    first match at focal lengths about the plate solves' — and the one
    that puts the most catalogue stars on detections wins (the same clip: 809 px,
    k1 -0.019, 3371 stars; the others 2850-2917)."""
    f_plate = float(np.median([s.scale for s in sols.values()]))
    best = None
    for first, m in FOLLOW_STARTS:
        Q, lc, stats = _follow_from(sols, dets, stars, width, height,
                                    f_plate * m, sky_mask, maglim,
                                    first=first)
        hits = _hits(Q, lc, dets, stars, maglim)
        if best is None or hits > best[0]:
            best = (hits, Q, lc, stats)
    hits, Q, lc, stats = best
    if log:
        log(f"  frame by frame: {len(Q)} frames, {stats['n_matches']} "
            f"matches, median {stats['median_resid_px']:.2f}px, p90 "
            f"{stats['p90_resid_px']:.2f}px; f {lc.f:.1f} px/rad "
            f"({hits} catalogue stars on detections)")
    return Q, lc, dict(stats, hits=int(hits))


def _hits(Q, lc, dets, stars, maglim, tol=3.0) -> int:
    """Catalogue stars predicted within `tol` px of a detection, summed
    over the frames: how well a lens and orientations fit."""
    keep = stars.mag <= maglim
    V = radec_to_vec(stars.ra[keep], stars.dec[keep])
    n = 0
    for k, q in Q.items():
        pred, z = lc.ray_to_px(V @ q.T)
        ok = lc.in_frame(pred, z, 5) & (z > 0.2)
        if ok.any() and len(dets.get(k, ())):
            d, _ = cKDTree(dets[k][:, :2]).query(pred[ok])
            n += int((d < tol).sum())
    return n


def _follow_from(sols, dets, stars, width, height, f0, sky_mask, maglim,
                 log=None, first="pairs"):
    """`follow` from one starting focal length, the first round fitted to
    the tracker's pairs (`first` "pairs") or to a match within `first`
    px."""
    ks = sorted(sols)
    c = np.array([width / 2, height / 2])
    Q = {k: kabsch(s.vec, pinhole_rays(s.xy, f0, c)) for k, s in sols.items()}
    lens = np.array([f0, c[0], c[1], 0.0, 0.0])
    keep = stars.mag <= maglim
    V = radec_to_vec(stars.ra[keep], stars.dec[keep])
    mag = stars.mag[keep]
    trees = {k: cKDTree(dets[k][:, :2]) for k in ks if len(dets.get(k, ()))}
    ks = [k for k in ks if k in trees]

    def lens_cam(q):
        p = np.zeros(12)
        p[4:9] = q
        return Camera(p, ks[0], width, height)

    def gather(tol):
        fi, vs, obs = [], [], []
        lc = lens_cam(lens)
        for i, k in enumerate(ks):
            pred, z = lc.ray_to_px(V @ Q[k].T)
            ok = lc.in_frame(pred, z, 5) & (z > 0.2)
            idx = np.flatnonzero(ok)
            if sky_mask is not None and idx.size:
                idx = idx[sky_mask[pred[idx, 1].astype(int),
                                   pred[idx, 0].astype(int)] > 0]
            if not idx.size:
                continue
            d, j = trees[k].query(pred[idx])
            good = d < tol
            used = set()
            for i_, j_ in zip(idx[good][np.argsort(mag[idx[good]])],
                              j[good][np.argsort(mag[idx[good]])]):
                if j_ not in used:
                    used.add(j_)
                    fi.append(i)
                    vs.append(V[i_])
                    obs.append(dets[k][j_, :2])
        return np.array(fi), np.array(vs).reshape(-1, 3), \
            np.array(obs, float).reshape(-1, 2)

    def pairs():
        """The stars the tracker (or a plate solve) matched: clean, bright
        pairs, so the lens starts right before re-matching against
        detections that are mostly noise (a first match at 8 px against
        those settled on different lenses from different starts)."""
        fi, vs, obs = [], [], []
        for i, k in enumerate(ks):
            s = sols[k]
            fi += [i] * len(s.xy)
            vs.append(np.asarray(s.vec).reshape(-1, 3))
            obs.append(np.asarray(s.xy, float).reshape(-1, 2))
        return np.array(fi), np.vstack(vs), np.vstack(obs)

    from scipy.sparse import lil_matrix
    # then wider re-matching; the tight rounds repeat while they still find
    # more stars (the edges come in as the distortion settles)
    last = -1
    for tol in (first, 5, 4) + (3,) * 8:
        fi, vs, obs = pairs() if tol == "pairs" else gather(tol)
        if tol == 3:
            if len(fi) <= last:
                break
            last = len(fi)
        rv0 = np.array([Rotation.from_matrix(Q[k]).as_rotvec() for k in ks])

        def res(q):
            R = Rotation.from_rotvec(q[5:].reshape(-1, 3)).as_matrix()
            r = np.einsum("nij,nj->ni", R[fi], vs)
            return (lens_cam(q[:5]).ray_to_px(r)[0] - obs).ravel()

        sp = lil_matrix((2 * len(fi), 5 + 3 * len(ks)), dtype=int)
        sp[:, :5] = 1
        for n, i in enumerate(fi):
            sp[2 * n:2 * n + 2, 5 + 3 * i:8 + 3 * i] = 1
        r = least_squares(res, np.r_[lens, rv0.ravel()], loss="soft_l1",
                          f_scale=2.0, x_scale="jac", jac_sparsity=sp)
        lens = r.x[:5]
        R = Rotation.from_rotvec(r.x[5:].reshape(-1, 3)).as_matrix()
        Q = {k: R[i] for i, k in enumerate(ks)}
        e = np.hypot(*res(r.x).reshape(-1, 2).T)
        if log:
            log(f"  frame by frame, tol {tol}, {len(ks)} frames: "
                f"{len(fi):5d} matches, median {np.median(e):.2f}px, p90 "
                f"{np.percentile(e, 90):.2f}px")
    stats = dict(n_matches=int(len(fi)), median_resid_px=float(np.median(e)),
                 p90_resid_px=float(np.percentile(e, 90)),
                 frames=[int(k) for k in ks], axis_fitted=False, moving=True)
    return Q, lens_cam(lens), stats


def refine_turn(cam: Camera, frames, stars: Stars, *, sky_mask=None,
                maglim: float = 5.5, min_stars: int = 20,
                log=print) -> tuple[Camera, dict]:
    """A turning camera's turn, frame by frame: each frame's own stars fix
    its orientation (a small rotation on the turn the sampled frames gave
    it), so jitter between the sampled frames doesn't reach the satellites
    (a dolly clip: 0.25 deg at frame 167, between samples 9 frames apart). The
    lens stays. `frames`: (k, image) pairs; frames with fewer than
    `min_stars` matched keep the interpolated turn. -> (camera, {frame:
    correction deg})."""
    keep = stars.mag <= maglim
    V = radec_to_vec(stars.ra[keep], stars.dec[keep])
    turn = cam.turn.copy()
    fixed = {}
    for k, img in frames:
        det = detect(img, sky_mask)[:, :2]
        if len(det) < min_stars:
            continue
        tree = cKDTree(det)
        rays = cam.rays(V, k)
        R = np.eye(3)
        n = 0
        for tol in (4.0, 3.0, 2.5):
            p, z = cam.ray_to_px(rays @ R.T)
            ok = cam.in_frame(p, z, 5) & (z > 0.2)
            if sky_mask is not None and ok.any():
                ok[ok] = sky_mask[p[ok, 1].astype(int),
                                  p[ok, 0].astype(int)] > 0
            d, j = tree.query(p[ok])
            g = d < tol
            n = int(g.sum())
            if n < min_stars:
                break
            R = kabsch(rays[ok][g], cam.px_to_ray(det[j[g]]))
        if n < min_stars:
            continue
        G = Rotation.from_rotvec(turn[k]).as_matrix()
        turn[k] = Rotation.from_matrix(R @ G).as_rotvec()
        fixed[int(k)] = float(np.degrees(Rotation.from_matrix(R).magnitude()))
    out = Camera(cam.p, cam.k0, cam.width, cam.height, dict(cam.meta),
                 cam.up, cam.clock, turn)
    if log and fixed:
        v = np.array(list(fixed.values()))
        log(f"  each frame's own stars: {len(fixed)} frames turned by "
            f"median {np.median(v):.3f}, most {v.max():.3f} deg from the "
            "sampled frames' turn")
    return out, fixed


def add_refraction(cam: Camera, dets: dict[int, np.ndarray], stars: Stars,
                   up: np.ndarray, *, sky_mask=None, maglim: float = 5.5,
                   refr0: float = 0.85, lens_and_rate: bool = True,
                   log=print):
    """Refit a fitted camera with refraction about `up` (camera frame, from
    the skyline), the atmosphere's density free. Returns (camera, stats)."""
    cam = cam.with_up(up, cam.p[11] or refr0)
    m = _Matcher(stars, dets, sky_mask, maglim)
    frames = sorted(m.dets)
    ax = AXIS if cam.meta.get("stats", {}).get("axis_fitted") else []
    for tol in (5, 3, 3):
        ks, vs, obs = m.gather(cam, frames, tol)
        # a turning camera's lens and rate came with its turn (the
        # interval the land keeps still): only its pose and the air here
        free = ORIENT + RATE + LENS + ax + REFR if lens_and_rate else \
            ORIENT + REFR
        cam = _fit(cam, ks, vs, obs, free)
        r = np.hypot(*(cam.project(vs, ks)[0] - obs).T)
        if log:
            log(f"  fit with refraction, tol {tol}px: {len(ks):5d} matches, "
                f"median {np.median(r):.2f}px, p90 {np.percentile(r, 90):.2f}"
                f"px, density {cam.p[11]:.2f}")
    stats = dict(cam.meta.get("stats", {}), n_matches=int(len(ks)),
                 median_resid_px=float(np.median(r)),
                 p90_resid_px=float(np.percentile(r, 90)), refraction=True)
    cam.meta["stats"] = stats
    return cam, stats


def refine(cam: Camera, dets: dict[int, np.ndarray], stars: Stars, *,
           sky_mask=None, maglim: float = 5.5, near: list[int] | None = None,
           log=print):
    """Iterative match-and-fit over the sampled frames in `dets`.
    Returns (camera, stats)."""
    m = _Matcher(stars, dets, sky_mask, maglim)
    frames = sorted(m.dets)
    near = near or [min(frames, key=lambda k: abs(k - cam.k0))]
    span = abs(cam.omega) * float(cam.s(frames[-1]) - cam.s(frames[0]))
    # the axis (pole of date) needs the sky to have turned a fair way
    ax = AXIS if span > 20 else []
    mid = [k for k in frames if abs(k - cam.k0) <= (frames[-1] - frames[0]) / 4]
    schedule = [(25, near, ORIENT + [4]),
                (25, near + mid, ORIENT + RATE + [4]),
                (15, frames, ORIENT + RATE + [4, 7]),
                (8, frames, ORIENT + RATE + LENS),
                (5, frames, ORIENT + RATE + LENS + ax),
                (3, frames, ORIENT + RATE + LENS + ax),
                (3, frames, ORIENT + RATE + LENS + ax)]
    for tol, fr, free in schedule:
        ks, vs, obs = m.gather(cam, fr, tol)
        if len(ks) < 12:
            raise RuntimeError(f"camera fit lost the stars ({len(ks)} "
                               f"matches at {tol} px)")
        cam = _fit(cam, ks, vs, obs, free)
        r = np.hypot(*(cam.project(vs, ks)[0] - obs).T)
        if log:
            log(f"  fit tol {tol:2d}px, {len(fr):3d} frames: {len(ks):5d} "
                f"matches, median {np.median(r):.2f}px, p90 "
                f"{np.percentile(r, 90):.2f}px")
    stats = dict(n_matches=int(len(ks)), median_resid_px=float(np.median(r)),
                 p90_resid_px=float(np.percentile(r, 90)),
                 frames=[int(k) for k in frames], axis_fitted=bool(ax))
    return cam, stats


def match_fraction(cam: Camera, dets: dict[int, np.ndarray], stars: Stars, *,
                   sky_mask=None, maglim: float = 4.5, tol: float = 3.0
                   ) -> dict[int, float]:
    """Per sampled frame: fraction of predicted bright stars detected. Drops
    where dawn, dusk or cloud wash the sky out."""
    m = _Matcher(stars, dets, sky_mask, maglim)
    out = {}
    for k in sorted(dets):
        pred, z = cam.project(m.v, k)
        ok = cam.in_frame(pred, z, 5) & (z > 0.2)
        idx = np.flatnonzero(ok)
        if sky_mask is not None and idx.size:
            idx = idx[sky_mask[pred[idx, 1].astype(int),
                               pred[idx, 0].astype(int)] > 0]
        if not idx.size or k not in m.trees:
            out[k] = 0.0
            continue
        d, _ = m.trees[k].query(pred[idx])
        out[k] = float((d < tol).mean())
    return out


def usable_range(fraction: dict[int, float], rel: float = 0.5
                 ) -> tuple[int, int]:
    """Longest run of sampled frames whose match fraction is at least `rel`
    of the median -> (first, last) frame. One bad sample between good ones
    (a glitched frame) doesn't break the run; twilight and cloud last
    longer."""
    ks = sorted(fraction)
    good = np.array([fraction[k] for k in ks]) >= rel * np.median(
        list(fraction.values()))
    good[1:-1] |= good[:-2] & good[2:]
    best, cur = (0, -1), None
    for i, g in enumerate(good):
        if g and cur is None:
            cur = i
        if (not g or i == len(good) - 1) and cur is not None:
            end = i if g else i - 1
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    return ks[best[0]], ks[best[1]]
