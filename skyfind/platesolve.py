"""Blind plate solve of one frame against a bright-star catalogue.

Quad hashing (astrometry.net style, simplified): each star and 3 of its
nearest neighbours form a quad; its shape (the 5 smallest of the 6 pairwise
distances over the largest) is scale and rotation invariant. Catalogue quads
are built in the tangent plane at their first star.

Wide-angle frames are the norm here, and a gnomonic lens stretches an
off-axis quad (radially by 1/cos^2, tangentially by 1/cos): a 50-deg-off
quad no longer hashes like its catalogue twin. So image quads are hashed in
the tangent plane at their own first star too, which needs the focal length:
a ladder of guesses spanning the plausible fields of view is tried (a guess
within ~30% is enough; the match itself then measures f).

A hit proposes a rotation (quad stars vs catalogue directions) and focal
length. Verification grows outwards from the quad: predict catalogue stars
within a radius, match detections, refit rotation + f on the matches, widen
— so lens distortion, unknown yet, can't sink a correct hypothesis.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .camera import kabsch, radec_to_vec, unit
from .catalog import Stars


@dataclass
class PlateSolution:
    frame: int
    xy: np.ndarray            # (n, 2) matched image stars
    vec: np.ndarray           # (n, 3) their catalogue directions
    mag: np.ndarray
    names: list
    scale: float              # focal length, px/rad (pinhole, centre)
    score: int
    M: np.ndarray | None = None   # camera orientation (pinhole)


def tangent(v: np.ndarray, c: np.ndarray):
    """Gnomonic projection of unit vectors v about unit vector c ->
    ((xi, eta), cos distance). Axes: east/north of c for the z pole."""
    e = np.cross([0, 0, 1.0], c)
    if np.linalg.norm(e) < 1e-9:                     # c is the pole
        e = np.array([0, 1.0, 0])
    e /= np.linalg.norm(e)
    n = np.cross(c, e)
    d = v @ c
    return np.stack([(v @ e) / d, (v @ n) / d], -1), d


def quad_hash(pts: np.ndarray):
    """pts (4, 2) -> (hash (5,), canonical order of the 4 points)."""
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    dd = d[np.triu_indices(4, 1)]
    order = np.argsort(d.sum(1))
    return np.sort(dd / dd.max())[:5], order


def _quads(vec: np.ndarray, nn_idx: np.ndarray, max_sep_deg: float):
    hs, qs = [], []
    cmax = np.cos(np.radians(max_sep_deg))
    for i, nn in enumerate(nn_idx):
        for c in itertools.combinations(nn, 3):
            q = np.array([i, *c])
            tp, d = tangent(vec[q], vec[i])
            if (d < cmax).any():
                continue
            h, o = quad_hash(tp)
            hs.append(h)
            qs.append(q[o])
    return np.array(hs).reshape(-1, 5), np.array(qs).reshape(-1, 4)


def catalog_quads(vec: np.ndarray, k: int = 6, max_sep_deg: float = 25.0):
    _, nn = cKDTree(vec).query(vec, k + 1)
    return _quads(vec, nn[:, 1:], max_sep_deg)


def image_quads(rays: np.ndarray, xy: np.ndarray, k: int = 6):
    """Quads of image stars (neighbours in pixels), hashed on rays."""
    _, nn = cKDTree(xy).query(xy, min(k + 1, len(xy)))
    return _quads(rays, nn[:, 1:], 60.0)


def pinhole_rays(xy: np.ndarray, f: float, c: np.ndarray) -> np.ndarray:
    return unit(np.column_stack([(xy - c) / f, np.ones(len(xy))]))


def pinhole_project(M: np.ndarray, f: float, c: np.ndarray, v: np.ndarray):
    r = v @ M.T
    z = r[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return c + f * r[:, :2] / z[:, None], z


def _refit(M, f, c, vec, xy):
    """Rotation + focal length to matched pairs (pinhole, soft L1)."""
    def res(q):
        p, _ = pinhole_project(Rotation.from_rotvec(q[:3]).as_matrix(), q[3],
                               c, vec)
        return (p - xy).ravel()
    r = least_squares(res, np.r_[Rotation.from_matrix(M).as_rotvec(), f],
                      loss="soft_l1", f_scale=3.0, x_scale="jac")
    return Rotation.from_rotvec(r.x[:3]).as_matrix(), float(r.x[3])


def _verify(M, f, c, anchor, cvec, det_tree, det_xy, width, height,
            match_px, radii):
    """Grow a hypothesis outwards from `anchor` (a catalogue direction):
    -> (pairs (cat index, det index), M, f, stars predicted in frame)."""
    pairs = np.zeros((0, 2), int)
    n_pred = 0
    for rad in radii:
        near = np.flatnonzero(cvec @ anchor > np.cos(np.radians(rad)))
        pred, z = pinhole_project(M, f, c, cvec[near])
        ok = (z > 0.2) & (pred[:, 0] > 0) & (pred[:, 0] < width) & \
            (pred[:, 1] > 0) & (pred[:, 1] < height)
        if ok.sum() < 4:
            continue
        d, j = det_tree.query(pred[ok])
        m = d < match_px
        # one detection per catalogue star and vice versa
        ci, dj = near[ok][m], j[m]
        _, keep = np.unique(dj, return_index=True)
        ci, dj = ci[keep], dj[keep]
        if len(ci) < 6:
            break
        pairs, n_pred = np.column_stack([ci, dj]), int(ok.sum())
        M, f = _refit(M, f, c, cvec[ci], det_xy[dj])
    return pairs, M, f, n_pred


def solve(det: np.ndarray, stars: Stars, width: int, height: int, *,
          frame: int = 0, fov_deg=(8.0, 140.0),
          schedule=((40, 3.5), (60, 4.0), (80, 4.5), (120, 5.0),
                    (160, 5.5), (220, 6.0), (300, 6.5)),
          accept: int = 25, match_px: float = 6.0, hash_tol: float = 0.012,
          verify_mag: float = 5.0, verify_dets: int = 600,
          log=print) -> PlateSolution | None:
    """`det`: (N, >=2) detections (x, y, ...), brightest first, foreground
    already removed. Tries more image stars against deeper catalogues, and a
    ladder of focal lengths, until a hypothesis verifies.

    Verifying means matching predicted catalogue stars to the brightest
    `verify_dets` detections, and beating chance clearly: with many noisy
    detections (a compressed video) a 6-px circle catches *something* a
    fair share of the time, so a solution needs at least `accept` matches
    and 6 sigma plus 8 more than the detection density alone would give."""
    all_xy = np.asarray(det)[:, :2].astype(float)
    if len(all_xy) < 10:
        return None
    det_xy = all_xy[:verify_dets]
    det_tree = cKDTree(det_xy)
    p_hit = min(1.0, len(det_xy) * np.pi * match_px ** 2 / (width * height))

    def excess(score, n_pred):
        chance = n_pred * p_hit
        return score - chance - 6 * np.sqrt(chance) - 8
    c = np.array([width / 2, height / 2])
    half = np.hypot(width, height) / 2
    # focal ladder: fields of view from wide to narrow, ~25% apart
    f_lo = width / 2 / np.tan(np.radians(fov_deg[1]) / 2)
    f_hi = width / 2 / np.tan(np.radians(fov_deg[0]) / 2)
    ladder = np.geomspace(f_lo, f_hi, int(np.ceil(np.log(f_hi / f_lo)
                                                   / np.log(1.3))) + 1)
    vcache: dict = {}

    def verify_cat(vmag):
        # a narrow field has few bright stars: verify as deep as it hashes
        if vmag not in vcache:
            keep = stars.mag <= vmag
            vcache[vmag] = (radec_to_vec(stars.ra[keep], stars.dec[keep]),
                            np.flatnonzero(keep))
        return vcache[vmag]
    cat_cache = {}
    best = None
    tried = set()
    for n_img, maglim in schedule:
        keep = stars.mag <= maglim
        if keep.sum() < 10:
            continue
        vec = radec_to_vec(stars.ra[keep], stars.dec[keep])
        if maglim not in cat_cache:
            ch, cq = catalog_quads(vec)
            cat_cache[maglim] = (ch, cq, cKDTree(ch))
        ch, cq, ctree = cat_cache[maglim]
        cvec, cidx = verify_cat(max(verify_mag, maglim + 0.5))
        xy = all_xy[:n_img]
        nhits = 0
        for f0 in ladder:
            rays = pinhole_rays(xy, f0, c)
            ih, iq = image_quads(rays, xy)
            if not len(ih):
                continue
            hits = ctree.query_ball_point(ih, r=hash_tol)
            for ii, cand in enumerate(hits):
                for cj in cand:
                    qi, qc = iq[ii], cq[cj]
                    key = (tuple(qi), tuple(qc))
                    if key in tried:
                        continue
                    tried.add(key)
                    nhits += 1
                    ti, _ = tangent(rays[qi], rays[qi[0]])
                    tc, _ = tangent(vec[qc], vec[qc[0]])
                    # scale of catalogue angles -> image (f0) angles
                    s = np.sqrt(((ti - ti.mean(0)) ** 2).sum()
                                / ((tc - tc.mean(0)) ** 2).sum())
                    if not 0.7 < s < 1.45:
                        continue            # another rung of the ladder
                    f = f0 * s
                    r4 = pinhole_rays(xy[qi], f, c)
                    M = kabsch(vec[qc], r4)
                    p4, _ = pinhole_project(M, f, c, vec[qc])
                    if np.abs(p4 - xy[qi]).max() > max(6.0, 0.01 * f):
                        continue
                    anchor = vec[qc[0]]
                    fov_r = np.degrees(np.arctan(half / f))
                    radii = [r for r in (8, 15, 25, 40, 60, 90) if r < fov_r]
                    radii.append(fov_r + 5)
                    pairs, M2, f2, n_pred = _verify(
                        M, f, c, anchor, cvec, det_tree, det_xy, width,
                        height, match_px, radii)
                    x = excess(len(pairs), n_pred)
                    if best is None or x > best[0]:
                        best = (x, pairs, M2, f2, cvec, cidx)
                if best and best[0] >= accept:
                    break
            if best and best[0] >= accept:
                break
        if log:
            log(f"  plate solve frame {frame}: {n_img} image stars vs mag < "
                f"{maglim}: {len(ch)} catalogue quads, {nhits} hits tried, "
                f"best {len(best[1]) if best else 0} stars "
                f"({best[0] if best else 0:+.0f} over chance)")
        if best and best[0] > 0 and len(best[1]) >= accept:
            break
    if not best or best[0] <= 0 or len(best[1]) < accept:
        return None
    _, pairs, M, f, cvec, cidx = best
    score = len(pairs)
    ci, dj = pairs[:, 0], pairs[:, 1]
    return PlateSolution(frame, det_xy[dj], cvec[ci], stars.mag[cidx[ci]],
                         [stars.name(int(i)) for i in cidx[ci]], float(f),
                         int(score), M)
