"""The solve's heavy lifting on the GPU (CuPy), when there is one.

The pass search sweeps every catalogued object at every row of the orbit
table (20 s apart over the clock window) for each track: ~10^8 object-rows
per track, done on the CPU a row at a time in worker processes. Here the
orbit table is copied to the graphics card once and the rows go through in
large blocks. Then every candidate's fit (`fit_all`) and every segment's
identification (`identify`) run there too. Same tests, in the same order,
as the CPU code: single precision only where the CPU's is, or with margins
and a double-precision recheck.
"""

from __future__ import annotations

import math

import numpy as np

from .site import A_WGS84, F_WGS84, WE

_E2 = F_WGS84 * (2 - F_WGS84)

# One thread per (table row, object), testing it against every track: the
# free-site tests before refinement (`solve.free_passes`'s first two
# stages) in single precision — GeForce cards do double precision at 1/64
# the rate — each loosened by a margin, so nothing that passes in double
# precision fails here; the survivors are checked again in double precision
# (`_EXACT`), as on the CPU. Each thread reads its object's position once
# for all the tracks. The line of sight's distance from the Earth's centre
# comes from a cross product (|p x d|), not |p|^2 - (p.d)^2, which cancels
# in single precision.
_STAGE1 = r"""
extern "C" __global__ void stage1(
    const float* r, const float* v, const long long nobj,
    const long long row_lo, const long long nrows, const int ntr,
    const int* t_lo, const int* t_hi, const float* d0, const float* sky,
    const int nsky, const float* sun, const float dark, const float Rs,
    const float A, const float e2, const float h0, const float WE,
    const float* pole, const float* u0, const float* rate_lo,
    const float* rate_hi, const float cosd, const float margin,
    int* out_row, int* out_j, int* out_t, int* count, const int max_out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows * nobj) return;
    long long row = row_lo + idx / nobj;
    long long j = idx % nobj;
    long long o = (row * nobj + j) * 3;
    float px = r[o], py = r[o + 1], pz = r[o + 2];
    if (isnan(px)) return;
    float vx = 0, vy = 0, vz = 0;
    bool have_v = false;
    for (int t = 0; t < ntr; t++) {
        if (row < t_lo[t] || row > t_hi[t]) continue;
        const float* d = d0 + 3 * t;
        float pd = px * d[0] + py * d[1] + pz * d[2];
        float cx = py * d[2] - pz * d[1], cy = pz * d[0] - px * d[2],
              cz = px * d[1] - py * d[0];
        float q = cx * cx + cy * cy + cz * cz;          // |p x d|^2
        float disc = Rs * Rs - q;
        if (!(disc > -2 * Rs * margin * 10)) continue;  // ~10 km slack
        float rho = pd - sqrtf(fmaxf(disc, 0.f));
        if (!(rho > -10.f)) continue;
        float x0 = px - rho * d[0], x1 = py - rho * d[1],
              x2 = pz - rho * d[2];
        float xn = sqrtf(x0 * x0 + x1 * x1 + x2 * x2);
        float ux = x0 / xn, uy = x1 / xn, uz = x2 / xn;
        const float* sn = sun + 3 * t;
        if (!(ux * sn[0] + uy * sn[1] + uz * sn[2] < dark + margin)) continue;
        float Rw = A * sqrtf((1 - e2) / (1 - e2 * (1 - uz * uz))) + h0;
        float disc2 = Rw * Rw - q;                       // on the ellipsoid
        float rho2 = pd - sqrtf(fmaxf(disc2, 0.f));
        float y0 = px - rho2 * d[0], y1 = py - rho2 * d[1],
              y2 = pz - rho2 * d[2];
        if (!have_v) { vx = v[o]; vy = v[o + 1]; vz = v[o + 2]; have_v = true; }
        float wx = vx - WE * (pole[1] * y2 - pole[2] * y1);
        float wy = vy - WE * (pole[2] * y0 - pole[0] * y2);
        float wz = vz - WE * (pole[0] * y1 - pole[1] * y0);
        float wd = wx * d[0] + wy * d[1] + wz * d[2];
        wx -= wd * d[0]; wy -= wd * d[1]; wz -= wd * d[2];
        float wn = sqrtf(wx * wx + wy * wy + wz * wz);
        float r_app = wn / fmaxf(rho2, 1e-3f) * 57.29578f;
        if (!(r_app < rate_hi[t] * 1.05f && r_app > rate_lo[t] / 1.05f))
            continue;
        float wm = fmaxf(wn, 1e-12f);
        const float* u = u0 + 3 * t;
        if (!((wx * u[0] + wy * u[1] + wz * u[2]) / wm > cosd - 0.01f))
            continue;
        // the frame's sky above the ground point: the dearest test, last
        // (all must pass, so the order changes nothing else)
        const float* sk = sky + (long long)t * nsky * 3;
        bool up = true;
        for (int k = 0; k < nsky; k++)
            if (!(ux * sk[3 * k] + uy * sk[3 * k + 1] + uz * sk[3 * k + 2]
                  > -margin)) { up = false; break; }
        if (!up) continue;
        int k = atomicAdd(count, 1);
        if (k < max_out) {
            out_row[k] = (int)row; out_j[k] = (int)j; out_t[k] = t;
        }
    }
}
"""

# Shared by the double-precision kernels below: a table position (cubic
# Hermite, as `DeviceOrbits.at_rows`) and where a line of sight meets the
# ellipsoid (`_ground_wgs84`).
_FREE_COMMON = r"""
__device__ void at_row(const float* R, const float* V, long long nobj,
                       long long nt, double dt, long long j, double x,
                       double* out)
{
    long long i = (long long)floor(x);
    if (i < 0) i = 0;
    if (i > nt - 2) i = nt - 2;
    double u = x - i, u2 = u * u, u3 = u2 * u;
    double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u,
           h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    long long a0 = (i * nobj + j) * 3, a1 = ((i + 1) * nobj + j) * 3;
    for (int k = 0; k < 3; k++)
        out[k] = h00 * (double)R[a0 + k] + h10 * ((double)V[a0 + k] * dt)
               + h01 * (double)R[a1 + k] + h11 * ((double)V[a1 + k] * dt);
}

__device__ bool ground84(const double* p, const double* d, double h0,
                         double A, double e2, double* x, double* range)
{
    double pd = p[0] * d[0] + p[1] * d[1] + p[2] * d[2];
    double pp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    double R1 = 6371.0 + h0;
    double disc = R1 * R1 - (pp - pd * pd);
    double rho = pd - sqrt(fmax(disc, 0.0));
    double s0 = p[0] - rho * d[0], s1 = p[1] - rho * d[1],
           s2 = p[2] - rho * d[2];
    double sn = sqrt(s0 * s0 + s1 * s1 + s2 * s2);
    double z = s2 / sn;
    double R = A * sqrt((1 - e2) / (1 - e2 * (1 - z * z))) + h0;
    disc = R * R - (pp - pd * pd);
    rho = pd - sqrt(fmax(disc, 0.0));
    for (int k = 0; k < 3; k++) x[k] = p[k] - rho * d[k];
    *range = rho;
    return disc > 0 && rho > 0;
}
"""

# One thread per survivor: `_exact`'s tests in double precision.
_EXACT = _FREE_COMMON + r"""
extern "C" __global__ void exact(
    const float* r, const float* v, const long long nobj, const int n,
    const int* srow, const int* sj, const int* st, const double* d0,
    const double* sky, const int nsky, const double* sun, const double dark,
    const double A, const double e2, const double h0, const double WE,
    const double* pole, const double* u0, const double* rate_lo,
    const double* rate_hi, const double cosd, bool* ok, double* xout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ok[i] = false;
    int t = st[i];
    long long o = ((long long)srow[i] * nobj + sj[i]) * 3;
    double p[3] = {r[o], r[o + 1], r[o + 2]};
    const double* d = d0 + 3 * t;
    // on a sphere first: the sky above, the Sun down
    double pd = p[0] * d[0] + p[1] * d[1] + p[2] * d[2];
    double pp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    double R1 = 6371.0 + h0;
    double disc = R1 * R1 - (pp - pd * pd);
    double rho = pd - sqrt(fmax(disc, 0.0));
    if (!(disc > 0 && rho > 0)) return;
    double x0 = p[0] - rho * d[0], x1 = p[1] - rho * d[1],
           x2 = p[2] - rho * d[2];
    double xn = sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    double ux = x0 / xn, uy = x1 / xn, uz = x2 / xn;
    const double* sk = sky + (long long)t * nsky * 3;
    double mn = 3.0;
    for (int k = 0; k < nsky; k++) {
        double c = ux * sk[3 * k] + uy * sk[3 * k + 1] + uz * sk[3 * k + 2];
        if (c < mn) mn = c;
    }
    if (!(mn > 0)) return;
    const double* sn = sun + 3 * t;
    if (!(ux * sn[0] + uy * sn[1] + uz * sn[2] < dark)) return;
    // on the ellipsoid: the apparent rate and direction
    double y[3], rho2;
    ground84(p, d, h0, A, e2, y, &rho2);
    double w0 = v[o] - WE * (pole[1] * y[2] - pole[2] * y[1]);
    double w1 = v[o + 1] - WE * (pole[2] * y[0] - pole[0] * y[2]);
    double w2 = v[o + 2] - WE * (pole[0] * y[1] - pole[1] * y[0]);
    double wd = w0 * d[0] + w1 * d[1] + w2 * d[2];
    w0 -= wd * d[0]; w1 -= wd * d[1]; w2 -= wd * d[2];
    double wn = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
    double r_app = wn / rho2 * 57.29577951308232;
    if (!(r_app < rate_hi[t] && r_app > rate_lo[t])) return;
    double wm = fmax(wn, 1e-12);
    const double* u = u0 + 3 * t;
    if (!((w0 / wm) * u[0] + (w1 / wm) * u[1] + (w2 / wm) * u[2] > cosd))
        return;
    ok[i] = true;
    xout[3 * i] = y[0]; xout[3 * i + 1] = y[1]; xout[3 * i + 2] = y[2];
}
"""

# One thread per survivor: `free_passes`' refinement — the ground point
# every `coarse` s across its table row, checked against the track's two
# other segments (turned with the Earth), then every `fine` s around the
# best. Minimum and argmax as CuPy's (NaN wins).
_REFINE = _FREE_COMMON + r"""
extern "C" __global__ void refine(
    const float* R, const float* V, const long long nobj, const long long nt,
    const double dt, const int n, const int* srow, const int* sj,
    const int* st, const double* x0, const double* d0, const double* chkM,
    const double* chkd, const double* chkdt, const double A,
    const double e2, const double h0, const double* cgrid, const int nc,
    const double* fgrid, const int nf, double* top_out, double* dt_out,
    double* x_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = st[i];
    long long j = sj[i];
    double row = srow[i];
    const double* d = d0 + 3 * t;
    double top = -2.0, tdt = 0.0;
    double bx[3] = {x0[3 * i], x0[3 * i + 1], x0[3 * i + 2]};
    for (int pass = 0; pass < 2; pass++) {
        int nk = pass == 0 ? nc : nf;
        double bw = 0, bdl = 0, bxd[3] = {0, 0, 0};
        int bi = -1;
        bool bnan = false;
        for (int k = 0; k < nk; k++) {
            double dl = pass == 0 ? cgrid[k] : tdt + fgrid[k];
            double xr = row + dl / dt;
            double p[3], xd[3];
            at_row(R, V, nobj, nt, dt, j, xr, p);
            double rg;
            bool hit = ground84(p, d, h0, A, e2, xd, &rg);
            double worst = hit ? 2.0 : -2.0;
            for (int c = 0; c < 2; c++) {
                const double* M = chkM + (t * 2 + c) * 9;
                const double* dc = chkd + (t * 2 + c) * 3;
                double q[3];
                at_row(R, V, nobj, nt, dt, j, xr + chkdt[t * 2 + c] / dt, q);
                for (int m = 0; m < 3; m++)
                    q[m] -= xd[0] * M[m] + xd[1] * M[3 + m] + xd[2] * M[6 + m];
                double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                double cv = (q[0] / qn) * dc[0] + (q[1] / qn) * dc[1]
                          + (q[2] / qn) * dc[2];
                worst = (isnan(cv) || isnan(worst) || cv < worst) ?
                    (isnan(worst) ? worst : cv) : worst;
            }
            if (isnan(worst)) {
                if (!bnan) {
                    bnan = true; bi = k; bw = worst; bdl = dl;
                    bxd[0] = xd[0]; bxd[1] = xd[1]; bxd[2] = xd[2];
                }
            } else if (!bnan && (bi < 0 || worst > bw)) {
                bi = k; bw = worst; bdl = dl;
                bxd[0] = xd[0]; bxd[1] = xd[1]; bxd[2] = xd[2];
            }
        }
        if (bw > top) {
            top = bw; tdt = bdl;
            bx[0] = bxd[0]; bx[1] = bxd[1]; bx[2] = bxd[2];
        }
    }
    top_out[i] = top; dt_out[i] = tdt;
    x_out[3 * i] = bx[0]; x_out[3 * i + 1] = bx[1]; x_out[3 * i + 2] = bx[2];
}
"""


def available() -> bool:
    """A CUDA device CuPy can compile for."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        (cp.arange(2) + 1).sum().item()          # compiles a kernel: NVRTC
        return True
    except Exception:                          # noqa: BLE001 - no GPU stack
        return False


class DeviceOrbits:
    """The orbit table (positions, velocities; float32) on the GPU: copied
    whole, or already there (`Upload`)."""

    def __init__(self, orbits, r=None, v=None):
        import cupy as cp
        self.cp = cp
        self.r = cp.asarray(np.ascontiguousarray(orbits.r)) if r is None \
            else r
        self.v = cp.asarray(np.ascontiguousarray(orbits.v)) if v is None \
            else v
        self.t0 = orbits.t0
        self.dt = orbits.dt
        self.n_t = orbits.r.shape[0]

    def at_rows(self, x, j):
        """Positions (float64) of objects `j` at fractional table rows `x`
        (cubic Hermite, as `Orbits.at_many`)."""
        cp = self.cp
        i = cp.clip(cp.floor(x).astype(cp.int64), 0, self.n_t - 2)
        u = (x - i)[..., None]
        p0 = self.r[i, j].astype(cp.float64)
        p1 = self.r[i + 1, j].astype(cp.float64)
        m0 = self.v[i, j].astype(cp.float64) * self.dt
        m1 = self.v[i + 1, j].astype(cp.float64) * self.dt
        u2, u3 = u * u, u * u * u
        return ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0
                + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1)

    def free(self):
        self.r = self.v = None
        self.cp.get_default_memory_pool().free_all_blocks()


class Upload:
    """The orbit table sent to the GPU block by block as the propagation
    finishes them (`Orbits.propagate(on_block=...)`), so the copy overlaps
    the rest of the work; `done` hands over the DeviceOrbits (None when
    the GPU hasn't room for the table and `reserve` bytes more)."""

    def __init__(self, reserve: int = 1 << 30):
        import cupy as cp
        self.cp = cp
        self.reserve = reserve
        self.r = self.v = None
        self.full = False

    def __call__(self, r, v, rows, cols) -> None:
        cp = self.cp
        if self.full:
            return
        if self.r is None:
            free, _ = cp.cuda.runtime.memGetInfo()
            if free < r.nbytes + v.nbytes + self.reserve:
                self.full = True
                return
            try:
                self.r = cp.empty(r.shape, cp.float32)
                self.v = cp.empty(v.shape, cp.float32)
            except cp.cuda.memory.OutOfMemoryError:
                self.r = self.v = None
                self.full = True
                cp.get_default_memory_pool().free_all_blocks()
                return
        self.r[rows, cols] = cp.asarray(np.ascontiguousarray(r[rows, cols]))
        self.v[rows, cols] = cp.asarray(np.ascontiguousarray(v[rows, cols]))

    def done(self, orbits, log=None) -> DeviceOrbits | None:
        if self.r is None:
            if log:
                free, _ = self.cp.cuda.runtime.memGetInfo()
                need = orbits.r.nbytes + orbits.v.nbytes
                log(f"  GPU: {free / 2**30:.1f} GB free, the orbit table "
                    f"needs {need / 2**30:.1f} GB: searching on the CPU")
            return None
        self.cp.cuda.Device().synchronize()
        return DeviceOrbits(orbits, self.r, self.v)


_free_kernels: dict = {}


def _free_kernel(cp, name):
    if name not in _free_kernels:
        src = {"stage1": _STAGE1, "exact": _EXACT, "refine": _REFINE}[name]
        _free_kernels[name] = cp.RawKernel(src, name)
    return _free_kernels[name]


def free_passes(P, dev: DeviceOrbits, segs, **kw):
    """`solve.free_passes` on the GPU for one track (`free_passes_all`)."""
    return free_passes_all(P, dev, [segs], **kw)[0]


def free_passes_all(P, dev: DeviceOrbits, tracks, **kw) -> list[list]:
    """`solve.free_passes` for every track at once on the GPU
    (`freesearch.run` with these kernels). -> each track's candidates."""
    from . import freesearch
    return freesearch.run(P, tracks, _GpuFree(dev), n_t=dev.n_t,
                          step=dev.dt, t0=dev.t0, **kw)


class _GpuFree:
    """The free search's kernels on the GPU, for `freesearch.run`."""

    def __init__(self, dev: DeviceOrbits, max_out: int = 1 << 23):
        self.dev = dev
        self.cp = self.xp = dev.cp
        self.max_out = max_out

    def get(self, x):
        return self.cp.asnumpy(x)

    def _a32(self, x, t=np.float32):
        return self.cp.asarray(np.ascontiguousarray(x, t))

    def _a64(self, x):
        return self.cp.asarray(np.ascontiguousarray(x, np.float64))

    def stage1(self, a):
        cp, dev = self.cp, self.dev
        nobj = dev.r.shape[1]
        a32 = self._a32
        max_out = self.max_out
        while True:
            s_row = cp.empty(max_out, cp.int32)
            s_j = cp.empty(max_out, cp.int32)
            s_t = cp.empty(max_out, cp.int32)
            count = cp.zeros(1, cp.int32)
            total = a["nrows"] * nobj
            _free_kernel(cp, "stage1")(((total + 255) // 256,), (256,), (
                dev.r, dev.v, np.int64(nobj), np.int64(a["row_lo"]),
                np.int64(a["nrows"]), np.int32(len(a["t_lo"])),
                a32(a["t_lo"], np.int32), a32(a["t_hi"], np.int32),
                a32(a["d0s"]), a32(a["skys"]), np.int32(a["skys"].shape[1]),
                a32(a["suns"]), np.float32(a["dark"]),
                np.float32(6371.0 + a["h0"]), np.float32(a["A"]),
                np.float32(a["e2"]), np.float32(a["h0"]), np.float32(a["WE"]),
                a32(a["pole"]), a32(a["u0s"]), a32(a["r_lo"]),
                a32(a["r_hi"]), np.float32(a["cosd"]), np.float32(1e-4),
                s_row, s_j, s_t, count, np.int32(max_out)))
            n = int(count.get()[0])
            if n <= max_out:
                break
            max_out = n + 1024                   # too many: again, with room
        return s_row[:n], s_j[:n], s_t[:n]

    def exact(self, a, s_row, s_j, s_t):
        cp, dev, a64 = self.cp, self.dev, self._a64
        n = len(s_row)
        ok = cp.empty(n, cp.bool_)
        x = cp.empty((n, 3))
        _free_kernel(cp, "exact")(((n + 255) // 256,), (256,), (
            dev.r, dev.v, np.int64(dev.r.shape[1]), np.int32(n),
            cp.ascontiguousarray(s_row), cp.ascontiguousarray(s_j),
            cp.ascontiguousarray(s_t), a64(a["d0s"]), a64(a["skys"]),
            np.int32(a["skys"].shape[1]), a64(a["suns"]),
            np.float64(a["dark"]), np.float64(a["A"]), np.float64(a["e2"]),
            np.float64(a["h0"]), np.float64(a["WE"]), a64(a["pole"]),
            a64(a["u0s"]), a64(a["r_lo"]), a64(a["r_hi"]),
            np.float64(a["cosd"]), ok, x))
        return ok, x

    def refine(self, a, s_row, s_j, s_t, x, coarse, fine, M, dc, dts):
        cp, dev, a64 = self.cp, self.dev, self._a64
        n = len(s_row)
        top, top_dt, xr = cp.empty(n), cp.empty(n), cp.empty((n, 3))
        _free_kernel(cp, "refine")(((n + 255) // 256,), (256,), (
            dev.r, dev.v, np.int64(dev.r.shape[1]), np.int64(dev.n_t),
            np.float64(dev.dt), np.int32(n), s_row, s_j, s_t, x,
            a64(a["d0s"]), a64(M), a64(dc), a64(dts), np.float64(a["A"]),
            np.float64(a["e2"]), np.float64(a["h0"]), a64(coarse),
            np.int32(len(coarse)), a64(fine), np.int32(len(fine)), top,
            top_dt, xr))
        return top, top_dt, xr


# ================================================================ fits
# One thread per (candidate, segment end): `solve.Problem.dirs` for that end
# at the candidate's (lat, dlst, T, E), minus where the end was seen, in
# degrees (0 where the end isn't free). Double precision throughout, like
# the CPU's.
_RESID = r"""
extern "C" __global__ void resid(
    const double* p, const long long* idx, const int C, const int M,
    const long long* jobj, const double* row0, const double* zra,
    const double* sgn, const double* obs, const bool* free,
    const float* R, const float* V, const long long nobj, const long long nt,
    const double dt, const double* Pm, const double* pole,
    const double* axis, const double w, const double A, const double e2,
    const double h, const double WE, double* out)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)C * M) return;
    int c = t / M, m = t % M;
    long long ci = idx[c];
    long long e = ci * M + m;
    double* o = out + ((long long)c * M + m) * 3;
    if (!free[e]) { o[0] = 0; o[1] = 0; o[2] = 0; return; }
    const double DEG = 57.29577951308232, RAD = 0.017453292519943295;
    double lat = p[c * 4], dlst = p[c * 4 + 1], T = p[c * 4 + 2],
           E = p[c * 4 + 3];
    double offs = sgn[e] * E / 2;
    double sp = sin(lat * RAD), cph = cos(lat * RAD);
    double N = A / sqrt(1 - e2 * sp * sp);
    double ra = zra[e] + dlst * RAD;
    double rho = (N + h) * cph;
    double v0 = rho * cos(ra), v1 = rho * sin(ra), v2 = (N * (1 - e2) + h) * sp;
    double g[3];
    for (int k = 0; k < 3; k++)
        g[k] = v0 * Pm[k] + v1 * Pm[3 + k] + v2 * Pm[6 + k];   // v @ Pm
    double vs0 = WE * (pole[1] * g[2] - pole[2] * g[1]);
    double vs1 = WE * (pole[2] * g[0] - pole[0] * g[2]);
    double vs2 = WE * (pole[0] * g[1] - pole[1] * g[0]);
    double x = row0[e] + (T + offs) / dt;
    long long i = (long long)floor(x);
    if (i < 0) i = 0;
    if (i > nt - 2) i = nt - 2;
    double u = x - i, u2 = u * u, u3 = u2 * u;
    double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u,
           h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    long long j = jobj[ci];
    long long a0 = (i * nobj + j) * 3, a1 = ((i + 1) * nobj + j) * 3;
    double d[3];
    double vsv[3] = {vs0, vs1, vs2};
    for (int k = 0; k < 3; k++) {
        double pos = h00 * R[a0 + k] + h10 * V[a0 + k] * dt
                   + h01 * R[a1 + k] + h11 * V[a1 + k] * dt;
        d[k] = pos - g[k] - vsv[k] * offs;
    }
    double n = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    d[0] /= n; d[1] /= n; d[2] /= n;
    double ang = w * offs, c_ = cos(ang), s_ = sin(ang);
    double ad = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
    double cx = axis[1] * d[2] - axis[2] * d[1];
    double cy = axis[2] * d[0] - axis[0] * d[2];
    double cz = axis[0] * d[1] - axis[1] * d[0];
    double rr[3] = {d[0] * c_ + cx * s_ + axis[0] * ad * (1 - c_),
                    d[1] * c_ + cy * s_ + axis[1] * ad * (1 - c_),
                    d[2] * c_ + cz * s_ + axis[2] * ad * (1 - c_)};
    for (int k = 0; k < 3; k++)
        o[k] = (rr[k] - obs[e * 3 + k]) * DEG;
}
"""


# ================================================================ fits
class FitBatch:
    """Every track's candidates at once on the GPU, for `batchfit.fit`:
    each candidate carries its own track's segment ends (padded to the
    longest track, the padding never free). The residuals are
    `solve.Problem.dirs` done with the orbit table on the device."""

    def __init__(self, P, dev: DeviceOrbits, j, row0, zra, sign, obs, free):
        import cupy as cp
        self.xp = cp
        self.P, self.dev = P, dev
        f64 = lambda a: cp.ascontiguousarray(cp.asarray(a, cp.float64))  # noqa
        self.j = cp.ascontiguousarray(cp.asarray(j, cp.int64))
        self.row0 = f64(row0)                    # (C, M) table row at T=0
        self.zra = f64(zra)                      # (C, M) zenith RA (rad)
        self.sign = f64(sign)                    # (C, M) -1 first end, +1
        self.obs = f64(obs)                      # (C, M, 3)
        self.free = cp.ascontiguousarray(cp.asarray(free, cp.bool_))
        site, cam = P.site, P.camera
        self.Pm = f64(site.P)
        self.pole = f64(site.pole)
        ax, w = P.exposure_turn()      # rad of sky (and rig) per second
        self.axis = f64(ax)
        self.w = w

    _kernel = None

    def residuals(self, p, idx=None):
        """One kernel launch (`_RESID`); `residuals_ops` is the same in
        CuPy array operations (kept for checking)."""
        cp = self.xp
        if FitBatch._kernel is None:
            FitBatch._kernel = cp.RawKernel(_RESID, "resid")
        if idx is None:
            idx = cp.arange(len(self.j), dtype=cp.int64)
        idx = cp.ascontiguousarray(idx, dtype=cp.int64)
        C, M = len(idx), self.row0.shape[1]
        out = cp.empty((C, M, 3))
        if not C:
            return out
        n = C * M
        dev = self.dev
        FitBatch._kernel(((n + 255) // 256,), (256,), (
            cp.ascontiguousarray(p, dtype=cp.float64), idx, np.int32(C),
            np.int32(M), self.j, self.row0, self.zra, self.sign, self.obs,
            self.free, dev.r, dev.v, np.int64(dev.r.shape[1]),
            np.int64(dev.n_t), np.float64(dev.dt), self.Pm, self.pole,
            self.axis, np.float64(self.w), np.float64(A_WGS84),
            np.float64(_E2), np.float64(self.P.h0), np.float64(WE), out))
        return out

    def residuals_ops(self, p, idx=None):
        cp = self.xp
        sl = slice(None) if idx is None else idx
        j, row0, zra = self.j[sl], self.row0[sl], self.zra[sl]
        obs, free, sign = self.obs[sl], self.free[sl], self.sign[sl]
        lat, dlst, T, E = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
        offs = sign * E / 2                                   # (C, M)
        # the site at each end's frame (site.Site.gcrs)
        phi = cp.radians(lat)
        sp, cph = cp.sin(phi), cp.cos(phi)
        N = A_WGS84 / cp.sqrt(1 - _E2 * sp * sp)
        h = self.P.h0
        ra = zra + cp.radians(dlst)
        rho = (N + h) * cph
        zz = cp.broadcast_to((N * (1 - _E2) + h) * sp, ra.shape)
        v = cp.stack([rho * cp.cos(ra), rho * cp.sin(ra), zz], -1) @ self.Pm
        vs = WE * cp.cross(self.pole, v)
        # the object at the end's time
        x = row0 + (T + offs) / self.dev.dt
        pos = self.dev.at_rows(x, cp.broadcast_to(j[:, None], x.shape))
        d = pos - v - vs * offs[..., None]
        d /= cp.linalg.norm(d, axis=-1, keepdims=True)
        # as the mid-exposure camera model sees it (camera.rotate)
        ang = (self.w * offs)[..., None]
        c, s = cp.cos(ang), cp.sin(ang)
        ax = self.axis
        d = d * c + cp.cross(ax, d) * s + ax * (d @ ax)[..., None] * (1 - c)
        r = cp.degrees(d - obs)
        r[~free] = 0.0
        return r


def fit_all(P, dev: DeviceOrbits, tracks, cands: dict, *,
            ends: int = 3_000_000, log=None) -> dict:
    """`batchfit.fit_candidates` for every track's candidates, on the GPU:
    shortest tracks first, as many candidates at a time as pad to about
    `ends` segment ends (tracks mixed: each fit is its own, so how they're
    grouped never changes one). -> {track: [fit dict or None per
    candidate]}."""
    import cupy as cp
    from skyfield.api import wgs84
    from skyfield.positionlib import Geocentric

    from . import data
    from .batchfit import _observed, fit
    from .site import AU_KM
    from .solve import GOOD_FIT
    order = [(i, c) for i in sorted(cands, key=lambda i: len(tracks[i]))
             for c in range(len(cands[i]))]
    parts, a = [], 0
    while a < len(order):
        # the candidates are in order of track length: the last one taken
        # sets the padding
        b = a + 1
        step = 1 << 16
        while b < len(order):
            nb = min(b + step, len(order))
            if (nb - a) * 2 * len(tracks[order[nb - 1][0]]) > ends:
                if step == 1:
                    break
                step //= 2
                continue
            b = nb
        parts.append(order[a:b])
        a = b
    out = {i: [None] * len(cands[i]) for i in cands}
    t_ref0 = (P.t_ref - dev.t0) * 86400
    per_track = {}
    for i in cands:
        segs = tracks[i]
        ks = np.repeat([float(s.k) for s in segs], 2)
        per_track[i] = dict(
            row0=(t_ref0 + np.asarray(P.s(ks), float) * P.spf) / dev.dt,
            zra=np.asarray(P.site.zenith_ra(ks), float),
            sign=np.tile([-1.0, 1.0], len(segs)), k1=float(segs[0].k))
    done = 0
    for part in parts:
        done += len(part)
        M = max(2 * len(tracks[i]) for i, _ in part)
        C = len(part)
        j = np.zeros(C, np.int64)
        row0 = np.zeros((C, M))
        zra = np.zeros((C, M))
        sign = np.ones((C, M))
        obs = np.zeros((C, M, 3))
        free = np.zeros((C, M), bool)
        p0 = np.zeros((C, 4))
        k1 = np.zeros(C)
        n = 0
        # track by track within the chunk (the ends are the track's)
        by_track: dict = {}
        for r_, (i, c) in enumerate(part):
            by_track.setdefault(i, []).append((r_, c))
        for i, rows in by_track.items():
            t = per_track[i]
            m = len(t["row0"])
            rr = np.array([r_ for r_, _ in rows])
            cs = [cands[i][c] for _, c in rows]
            af = np.array([c_[3] for c_ in cs], bool)
            o, fr = _observed(tracks[i], af)
            j[rr] = [c_[0] for c_ in cs]
            row0[rr] = t["row0"][-1]                 # padding: harmless
            row0[rr, :m] = t["row0"]
            zra[rr] = t["zra"][-1]
            zra[rr, :m] = t["zra"]
            sign[rr, :m] = t["sign"]
            obs[rr, :m] = o
            free[rr, :m] = fr
            p0[rr] = [[c_[4], c_[5], c_[1], min(max(c_[2], 0.5), P.spf)]
                      for c_ in cs]
            k1[rr] = t["k1"]
            n += len(rows)
        use = free.any(1)
        lo = np.column_stack([np.full(C, -89.9), p0[:, 1] - 12,
                              np.full(C, -P.twin - 120), np.full(C, 0.2)])
        hi = np.column_stack([np.full(C, 89.9), p0[:, 1] + 12,
                              np.full(C, P.twin + 120), np.full(C, P.spf)])
        keep = np.flatnonzero(use)
        if not keep.size:
            continue
        batch = FitBatch(P, dev, j[keep], row0[keep], zra[keep], sign[keep],
                         obs[keep], free[keep])
        p, r, ok = fit(batch, cp.asarray(p0[keep]), cp.asarray(lo[keep]),
                       cp.asarray(hi[keep]))
        errs = cp.linalg.norm(r, axis=2)
        errs[~batch.free] = cp.nan
        med = cp.asnumpy(cp.nanmedian(errs, axis=1))
        mx = cp.asnumpy(cp.nanmax(errs, axis=1))
        p, ok = cp.asnumpy(p), cp.asnumpy(ok)
        del batch, r, errs
        # only what `solve.track_result` keeps is needed: the good fits,
        # and each track's best when none is good
        good = ok & np.isfinite(med)
        want = good & (med < GOOD_FIT)
        tr_of = np.array([part[r_][0] for r_ in keep])
        for i in np.unique(tr_of[good]):
            q = np.flatnonzero(good & (tr_of == i))
            if not want[q].any():
                want[q[np.argmin(med[q])]] = True
        # where each site is, at its track's first frame
        sel = np.flatnonzero(want)
        kk = k1[keep][sel]
        rs = P.site.gcrs(kk, p[sel, 0], p[sel, 1], P.h0)
        tt = P.t_ref + np.asarray(P.s(kk), float) * P.spf / 86400 + \
            p[sel, 2] / 86400
        g = wgs84.geographic_position_of(Geocentric(
            rs.T / AU_KM, t=data.times(tt)))
        glat = np.full(len(keep), np.nan)
        glon = np.full(len(keep), np.nan)
        glat[sel] = np.atleast_1d(g.latitude.degrees)
        glon[sel] = np.atleast_1d(g.longitude.degrees)
        for q in sel:
            r_ = keep[q]
            i, c = part[r_]
            jj = int(j[r_])
            out[i][c] = dict(obj=P.names[jj], norad=int(P.norad[jj]), j=jj,
                             lat=float(glat[q]), lon=float(glon[q]),
                             slat=float(p[q, 0]), dlst=float(p[q, 1]),
                             T=float(p[q, 2]), E=float(p[q, 3]),
                             med_err=float(med[q]), max_err=float(mx[q]))
        if log:
            log(f"  fitted {done} of {len(order)} candidates on the GPU")
    return out


# ================================================================ identify
# One thread per (segment at a solution, object): is the object within the
# prefilter cone of the segment's middle at any of the coarse times
# (`solve.identify`'s first step)? Single precision, the cone loosened a
# little so nothing the CPU would keep is lost.
_NEAR = r"""
extern "C" __global__ void near_(
    const float* R, const float* V, const long long nobj, const long long nt,
    const double dt, const int Q, const double* x0, const double step,
    const int nstep, const double* rs, const double* mid, const float cosp,
    int* out_q, int* out_j, int* count, const int max_out)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)Q * nobj) return;
    int q = t / nobj;
    long long j = t % nobj;
    float sx = rs[3 * q], sy = rs[3 * q + 1], sz = rs[3 * q + 2];
    float mx = mid[3 * q], my = mid[3 * q + 1], mz = mid[3 * q + 2];
    float fdt = (float)dt;
    for (int s = 0; s < nstep; s++) {
        double x = x0[q] + s * step;
        long long i = (long long)floor(x);
        if (i < 0) i = 0;
        if (i > nt - 2) i = nt - 2;
        float u = (float)(x - i), u2 = u * u, u3 = u2 * u;
        float h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u,
              h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
        long long a0 = (i * nobj + j) * 3, a1 = ((i + 1) * nobj + j) * 3;
        float d0 = h00 * R[a0] + h10 * V[a0] * fdt + h01 * R[a1]
                 + h11 * V[a1] * fdt - sx;
        float d1 = h00 * R[a0 + 1] + h10 * V[a0 + 1] * fdt + h01 * R[a1 + 1]
                 + h11 * V[a1 + 1] * fdt - sy;
        float d2 = h00 * R[a0 + 2] + h10 * V[a0 + 2] * fdt + h01 * R[a1 + 2]
                 + h11 * V[a1 + 2] * fdt - sz;
        float n = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        if (d0 * mx + d1 * my + d2 * mz > cosp * n) {
            int k = atomicAdd(count, 1);
            if (k < max_out) { out_q[k] = q; out_j[k] = (int)j; }
            return;
        }
    }
}
"""

# One thread per (segment at a solution, nearby object): `solve._best_pass`
# for that object — its directions over the exposure (`Problem.dirs`,
# double precision) and the times it passes closest to each end.
_PASS = r"""
extern "C" __global__ void pass_(
    const float* R, const float* V, const long long nobj, const long long nt,
    const double dt, const int npair, const int* pq, const int* pj,
    const double* xq, const double* rs, const double* vs, const double* A,
    const double* B, const double* axis, const double w, const double s0,
    const double ds, const int ns, double* ca_out, double* cb_out,
    int* ia_out, int* ib_out, bool* fin_out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npair) return;
    int q = pq[t];
    long long j = pj[t];
    const double* r = rs + 3 * q;
    const double* v = vs + 3 * q;
    const double* a = A + 3 * q;
    const double* b = B + 3 * q;
    double bca = 0, bcb = 0;
    int ia = 0, ib = 0;
    bool fin = true;
    for (int s = 0; s < ns; s++) {
        double off = s0 + s * ds;
        double x = xq[q] + off / dt;
        long long i = (long long)floor(x);
        if (i < 0) i = 0;
        if (i > nt - 2) i = nt - 2;
        double u = x - i, u2 = u * u, u3 = u2 * u;
        double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u,
               h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
        long long a0 = (i * nobj + j) * 3, a1 = ((i + 1) * nobj + j) * 3;
        double d[3];
        for (int k = 0; k < 3; k++)
            d[k] = h00 * R[a0 + k] + h10 * V[a0 + k] * dt + h01 * R[a1 + k]
                 + h11 * V[a1 + k] * dt - r[k] - v[k] * off;
        double n = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        d[0] /= n; d[1] /= n; d[2] /= n;
        double ang = w * off, c_ = cos(ang), s_ = sin(ang);
        double ad = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
        double cx = axis[1] * d[2] - axis[2] * d[1];
        double cy = axis[2] * d[0] - axis[0] * d[2];
        double cz = axis[0] * d[1] - axis[1] * d[0];
        double e0 = d[0] * c_ + cx * s_ + axis[0] * ad * (1 - c_);
        double e1 = d[1] * c_ + cy * s_ + axis[1] * ad * (1 - c_);
        double e2 = d[2] * c_ + cz * s_ + axis[2] * ad * (1 - c_);
        if (!(isfinite(e0) && isfinite(e1) && isfinite(e2))) fin = false;
        double ca = e0 * a[0] + e1 * a[1] + e2 * a[2];
        double cb = e0 * b[0] + e1 * b[1] + e2 * b[2];
        if (isnan(ca)) ca = -2.0;
        if (isnan(cb)) cb = -2.0;
        if (s == 0 || ca > bca) { bca = ca; ia = s; }
        if (s == 0 || cb > bcb) { bcb = cb; ib = s; }
    }
    ca_out[t] = bca; cb_out[t] = bcb; ia_out[t] = ia; ib_out[t] = ib;
    fin_out[t] = fin;
}
"""

_id_kernels: dict = {}


def _id_kernel(cp, name):
    if name not in _id_kernels:
        src = {"near_": _NEAR, "pass_": _PASS}[name]
        _id_kernels[name] = cp.RawKernel(src, name)
    return _id_kernels[name]


def identify(P, dev: DeviceOrbits, segs, sols, *, tol=0.25, centre_s=8.0,
             prefilter_deg=8.0, max_out=1 << 22) -> list[list[dict]]:
    """`solve.identify_all` on the GPU: every segment at every solution at
    once — the nearby objects in one kernel over the whole catalogue, then
    `solve._best_pass` for each (segment, object) in another; the same
    tests in the same order. -> one list of IDs per solution."""
    from . import data
    from .camera import unit
    from .solve import _ID_COARSE, _ID_FINE
    cp = dev.cp
    nS, nH = len(segs), len(sols)
    if not nS or not nH:
        return [[] for _ in sols]
    ks = np.array([s.k for s in segs], float)
    a = np.array([s.a for s in segs], float)
    b = np.array([s.b for s in segs], float)
    mid = unit(a + b)
    clip = np.array([bool(s.clip_a or s.clip_b) for s in segs])
    # every (solution, segment) pair: q = h * nS + i
    base = (P.t_ref - dev.t0) * 86400 + np.asarray(P.s(ks), float) * P.spf
    T = np.array([float(s["T"]) for s in sols])
    E = np.array([float(s["E"]) for s in sols])
    rs = np.concatenate([P.site.gcrs(ks, s["slat"], s["dlst"],
                                     s.get("height_km", P.h0))
                         for s in sols]).reshape(-1, 3)
    vs = P.site.velocity(rs)
    xq = ((base[None, :] + T[:, None]) / dev.dt).ravel()   # row, mid-exp.
    Q = nH * nS
    nobj = dev.r.shape[1]

    def g(x, t=cp.float64):
        return cp.ascontiguousarray(cp.asarray(x, t))
    rs_g, xq_g = g(rs), g(xq)
    # the nearby objects (single precision, a slightly wider cone)
    c0 = float(_ID_COARSE[0])
    dc = float(_ID_COARSE[1] - _ID_COARSE[0])
    cosp = math.cos(math.radians(prefilter_deg)) - 1e-4
    while True:
        out_q = cp.empty(max_out, cp.int32)
        out_j = cp.empty(max_out, cp.int32)
        count = cp.zeros(1, cp.int32)
        n = Q * nobj
        _id_kernel(cp, "near_")(((n + 255) // 256,), (256,), (
            dev.r, dev.v, np.int64(nobj), np.int64(dev.n_t),
            np.float64(dev.dt), np.int32(Q), g(xq + c0 / dev.dt),
            np.float64(dc / dev.dt), np.int32(len(_ID_COARSE)), rs_g,
            g(np.tile(mid, (nH, 1))), np.float32(cosp), out_q, out_j,
            count, np.int32(max_out)))
        m = int(count.get()[0])
        if m <= max_out:
            break
        max_out = m + 1024                     # too many: again, with room
    if not m:
        return [[] for _ in sols]
    # in (pair, object) order, as the CPU takes them
    key = out_q[:m].astype(cp.int64) * nobj + out_j[:m]
    key.sort()
    pq = (key // nobj).astype(cp.int32)
    pj = (key % nobj).astype(cp.int32)
    del out_q, out_j, key
    # each one's closest passes to the ends over the exposure (double)
    ns = len(_ID_FINE)
    ca, cb = cp.empty(m), cp.empty(m)
    ia, ib = cp.empty(m, cp.int32), cp.empty(m, cp.int32)
    fin = cp.empty(m, cp.bool_)
    axis, w = P.exposure_turn()
    axis = np.asarray(axis, float)
    _id_kernel(cp, "pass_")(((m + 255) // 256,), (256,), (
        dev.r, dev.v, np.int64(nobj), np.int64(dev.n_t), np.float64(dev.dt),
        np.int32(m), pq, pj, xq_g, rs_g, g(vs), g(np.tile(a, (nH, 1))),
        g(np.tile(b, (nH, 1))), g(axis), np.float64(w),
        np.float64(_ID_FINE[0]), np.float64(_ID_FINE[1] - _ID_FINE[0]),
        np.int32(ns), ca, cb, ia, ib, fin))
    # `_best_pass`'s tests
    ss = cp.asarray(_ID_FINE)
    sa, sb = ss[ia], ss[ib]
    err = cp.maximum(cp.degrees(cp.arccos(cp.clip(ca, -1, 1))),
                     cp.degrees(cp.arccos(cp.clip(cb, -1, 1))))
    tm, span = (sa + sb) / 2, cp.abs(sb - sa)
    Eq = cp.asarray(np.repeat(E, nS))[pq]
    clipq = cp.asarray(np.tile(clip, nH))[pq]
    inside = cp.where(clipq,
                      cp.maximum(cp.abs(sa), cp.abs(sb)) <= Eq / 2 + 3,
                      cp.abs(tm) <= centre_s)
    ok = fin & (ia > 0) & (ia < ns - 1) & (ib > 0) & (ib < ns - 1) & \
        (err < tol) & inside & (span <= Eq + 3)
    sel = cp.flatnonzero(ok)
    q_, j_ = cp.asnumpy(pq[sel]), cp.asnumpy(pj[sel])
    err_, tm_, span_ = (cp.asnumpy(x[sel]) for x in (err, tm, span))
    # the best-fitting object for each (solution, segment); ties go to the
    # first in catalogue order, as argmin's do
    o = np.lexsort((err_, q_))
    first = o[np.unique(q_[o], return_index=True)[1]]
    out: list[list[dict]] = [[] for _ in sols]
    ts = data.timescale()
    for h in range(nH):
        pick = first[q_[first] // nS == h]
        if not pick.size:
            continue
        pick = pick[np.argsort(q_[pick])]
        i = q_[pick] % nS
        utc = ts.tt_jd(np.asarray(P.tk(ks[i]), float) + T[h] / 86400
                       ).utc_strftime("%Y-%m-%d %H:%M:%S")
        for r, (k, ii) in enumerate(zip(pick, i)):
            jj, s = int(j_[k]), segs[ii]
            out[h].append(dict(seg=int(ii), k=s.k, name=P.names[jj],
                               norad=int(P.norad[jj]),
                               err_deg=round(float(err_[k]), 4),
                               dt=float(tm_[k]), span=float(span_[k]),
                               utc=utc[r][11:], clipped=bool(clip[ii])))
    return out


# ================================================================ skyline
# The pass search with a skyline (the site known roughly): `solve.
# _near_objects`, one thread per (table row, object), single precision like
# the CPU's; then `solve.check_passes`, one thread per (object, clock, grid
# site) cell, double precision.
_SWEEP = r"""
extern "C" __global__ void sweep_(
    const float* R, const long long nobj, const long long row_lo,
    const long long npair, const float* rc, const float* ea,
    const float* eb, const float cosn, const int motion, const float rate,
    const float* dir, const float rate_factor, const float cosd,
    const float step, int* out_p, int* out_j, int* count, const int max_out)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npair * nobj) return;
    long long p = t / nobj, j = t % nobj;
    long long o = ((row_lo + p) * nobj + j) * 3;
    float d0 = R[o] - rc[0], d1 = R[o + 1] - rc[1], d2 = R[o + 2] - rc[2];
    float n = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
    d0 /= n; d1 /= n; d2 /= n;
    float m = fmaxf(d0 * ea[0] + d1 * ea[1] + d2 * ea[2],
                    d0 * eb[0] + d1 * eb[1] + d2 * eb[2]);
    if (!(m > cosn)) return;
    if (motion) {
        long long o2 = o + nobj * 3;
        float f0 = R[o2] - rc[0], f1 = R[o2 + 1] - rc[1],
              f2 = R[o2 + 2] - rc[2];
        float n2 = sqrtf(f0 * f0 + f1 * f1 + f2 * f2);
        float v0 = f0 / n2 - d0, v1 = f1 / n2 - d1, v2 = f2 / n2 - d2;
        float r = sqrtf(v0 * v0 + v1 * v1 + v2 * v2) * 57.29578f / step;
        float vd = v0 * d0 + v1 * d1 + v2 * d2;
        float u0 = v0 - vd * d0, u1 = v1 - vd * d1, u2 = v2 - vd * d2;
        float un = fmaxf(sqrtf(u0 * u0 + u1 * u1 + u2 * u2), 1e-12f);
        if (!(r < rate * rate_factor && r > rate / rate_factor &&
              (u0 * dir[0] + u1 * dir[1] + u2 * dir[2]) / un > cosd))
            return;
    }
    int k = atomicAdd(count, 1);
    if (k < max_out) { out_p[k] = (int)p; out_j[k] = (int)j; }
}
"""

_CHECK = r"""
__device__ void look(const float* R, const float* V, long long nobj,
                     long long nt, double dt, long long j, double x,
                     const double* rs, const double* vs, double off,
                     double* d)
{
    long long i = (long long)floor(x);
    if (i < 0) i = 0;
    if (i > nt - 2) i = nt - 2;
    double u = x - i, u2 = u * u, u3 = u2 * u;
    double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u,
           h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    long long a0 = (i * nobj + j) * 3, a1 = ((i + 1) * nobj + j) * 3;
    for (int k = 0; k < 3; k++)
        d[k] = h00 * R[a0 + k] + h10 * V[a0 + k] * dt + h01 * R[a1 + k]
             + h11 * V[a1 + k] * dt - rs[k] - vs[k] * off;
    // one reciprocal square root, not a root and three divisions: double
    // precision division is slow on GeForce cards
    double rn = rsqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    d[0] *= rn; d[1] *= rn; d[2] *= rn;
}

__device__ double deg(double c)
{
    return acos(fmin(fmax(c, -1.0), 1.0)) * 57.29577951308232;
}

extern "C" __global__ void check_(
    const float* R, const float* V, const long long nobj, const long long nt,
    const double dt, const long long ncell, const int ng, const int* hj,
    const double* hT, const long long* list, const int use_list,
    const int ns, const int mid, const double* xb, const double* RS,
    const double* VS, const double* A, const double* B, const double tol,
    const double s0, const double cstep, const int nc, const double fstep,
    const int nf, bool* alive, double* e_out, double* tc_out,
    double* span_out, bool* first_out, const int write)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncell) return;
    long long c = use_list ? list[t] : t;
    long long h = c / ng;
    int g = (int)(c % ng);
    long long j = hj[h];
    double T = hT[h];
    double d[3], pv[3], st[64];
    alive[t] = false;
    // a conservative first look at the middle segment (`_coarse_ok`)
    {
        const double* rs = RS + (mid * ng + g) * 3;
        const double* vs = VS + (mid * ng + g) * 3;
        const double* a = A + mid * 3;
        const double* b = B + mid * 3;
        double bca = -3, bcb = -3;
        int ia = 0, ib = 0;
        for (int i = 0; i < nc; i++) {
            double off = s0 + i * cstep;
            look(R, V, nobj, nt, dt, j, xb[mid] + (T + off) / dt, rs, vs,
                 off, d);
            if (i > 0)          // the step's cosine (its angle when needed)
                st[i - 1] = d[0] * pv[0] + d[1] * pv[1] + d[2] * pv[2];
            double ca = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
            double cb = d[0] * b[0] + d[1] * b[1] + d[2] * b[2];
            if (isnan(ca)) ca = -2.0;
            if (isnan(cb)) cb = -2.0;
            if (ca > bca) { bca = ca; ia = i; }
            if (cb > bcb) { bcb = cb; ib = i; }
            pv[0] = d[0]; pv[1] = d[1]; pv[2] = d[2];
        }
        double ra = deg(st[ia < nc - 2 ? ia : nc - 2]) / cstep;
        double rb = deg(st[ib < nc - 2 ? ib : nc - 2]) / cstep;
        if (!(deg(bca) <= tol + 1.5 * ra * cstep / 2 + 0.01)) return;
        if (!(deg(bcb) <= tol + 1.5 * rb * cstep / 2 + 0.01)) return;
    }
    // every segment, the middle one first (`check_passes`)
    double tlo = 1e300, thi = -1e300;
    for (int q = 0; q < ns; q++) {
        int s = q == 0 ? mid : (q <= mid ? q - 1 : q);
        const double* rs = RS + (s * ng + g) * 3;
        const double* vs = VS + (s * ng + g) * 3;
        const double* a = A + s * 3;
        const double* b = B + s * 3;
        double bca = -3, bcb = -3;
        int ia = 0, ib = 0;
        for (int i = 0; i < nf; i++) {
            double off = s0 + i * fstep;
            look(R, V, nobj, nt, dt, j, xb[s] + (T + off) / dt, rs, vs,
                 off, d);
            double ca = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
            double cb = d[0] * b[0] + d[1] * b[1] + d[2] * b[2];
            if (isnan(ca)) ca = -2.0;
            if (isnan(cb)) cb = -2.0;
            if (ca > bca) { bca = ca; ia = i; }
            if (cb > bcb) { bcb = cb; ib = i; }
        }
        double e = fmax(deg(bca), deg(bcb));
        if (!(e <= tol && ia > 0 && ia < nf - 1 && ib > 0 && ib < nf - 1))
            return;
        double sa = s0 + ia * fstep, sb = s0 + ib * fstep;
        double tc = T + (sa + sb) / 2;
        tlo = fmin(tlo, tc);
        thi = fmax(thi, tc);
        if (write) {
            e_out[t * ns + s] = e;
            tc_out[t * ns + s] = tc;
            span_out[t * ns + s] = fabs(sb - sa);
            first_out[t * ns + s] = ia < ib;
        }
    }
    alive[t] = thi - tlo < 40;
}
"""

_sky_kernels: dict = {}


def _sky_kernel(cp, name):
    if name not in _sky_kernels:
        src = {"sweep_": _SWEEP, "check_": _CHECK}[name]
        _sky_kernels[name] = cp.RawKernel(src, name)
    return _sky_kernels[name]


def near_objects(P, dev: DeviceOrbits, segs, *, near_deg=25.0, quantum=40.0,
                 rate_factor=1.5, dir_deg=40.0, max_out=1 << 22):
    """`solve.near_objects` on the GPU, the whole clock window at once
    -> (objects, clocks), sorted by object then clock."""
    from .solve import track_motion
    cp = dev.cp
    seg = segs[len(segs) // 2]
    rate, u = track_motion(P, segs)
    tk = float(P.tk(seg.k))
    rc = P.site.gcrs(seg.k, P.lat0, 0.0, P.h0)
    step = dev.dt
    lo = max(0, int(np.ceil(((tk - P.twin / 86400) - dev.t0) * 86400
                            / step)))
    hi = min(dev.n_t - 1, int(np.floor(((tk + P.twin / 86400) - dev.t0)
                                       * 86400 / step)) + 1)
    rows = np.arange(lo, hi + 1)
    Ts = (dev.t0 + rows * step / 86400 - tk) * 86400      # as the CPU's
    npair = hi - lo
    nobj = dev.r.shape[1]
    if npair <= 0:
        return np.zeros(0, np.int64), np.zeros(0)
    f32 = lambda a: cp.asarray(np.ascontiguousarray(a, np.float32))  # noqa
    while True:
        out_p = cp.empty(max_out, cp.int32)
        out_j = cp.empty(max_out, cp.int32)
        count = cp.zeros(1, cp.int32)
        n = npair * nobj
        _sky_kernel(cp, "sweep_")(((n + 255) // 256,), (256,), (
            dev.r, np.int64(nobj), np.int64(lo), np.int64(npair), f32(rc),
            f32(seg.a), f32(seg.b),
            np.float32(math.cos(math.radians(near_deg))), np.int32(1),
            np.float32(rate), f32(u), np.float32(rate_factor),
            np.float32(math.cos(math.radians(dir_deg))), np.float32(step),
            out_p, out_j, count, np.int32(max_out)))
        m = int(count.get()[0])
        if m <= max_out:
            break
        max_out = m + 1024
    p, j = cp.asnumpy(out_p[:m]), cp.asnumpy(out_j[:m]).astype(np.int64)
    # (object, clock quantum) once each, as one integer key
    qi = np.rint(Ts / quantum).astype(np.int64)
    q0, nq = int(qi.min()), int(qi.max() - qi.min()) + 1
    key = np.unique(j * nq + (qi[p] - q0))
    return key // nq, ((key % nq) + q0) * float(quantum)


def check_passes(P, dev: DeviceOrbits, segs, J, Ts, *, tol=3.0, step=1.0,
                 merge_s=60.0, coarse=4.0, chunk=1 << 23):
    """`solve.check_passes` on the GPU for (object, clock) hypotheses `J`,
    `Ts` (sorted by object, then clock): every cell (hypothesis x grid
    site) in one kernel, the survivors' per-segment results in a second
    pass over them alone, then the CPU's choice of cells."""
    from .solve import SITE_GRID
    cp = dev.cp
    if not len(J):
        return []
    grid = np.array(SITE_GRID, float) + [P.lat0, 0.0]
    ng, ns = len(grid), len(segs)
    mid = ns // 2
    ks = np.array([s.k for s in segs], float)
    xb = ((P.t_ref - dev.t0) * 86400 + np.asarray(P.s(ks), float) * P.spf) \
        / dev.dt
    RS = np.stack([P.site.gcrs(s.k, grid[:, 0], grid[:, 1], P.h0)
                   for s in segs])                          # (ns, ng, 3)
    VS = P.site.velocity(RS.reshape(-1, 3)).reshape(RS.shape)
    ss = np.arange(-60, 60 + step / 2, step)
    sc = np.arange(-60, 60 + coarse / 2, coarse)
    assert len(sc) <= 65 and ss[0] == sc[0]

    def g(x, t=cp.float64):
        return cp.ascontiguousarray(cp.asarray(x, t))
    common = (g(xb), g(RS), g(VS), g(np.array([s.a for s in segs])),
              g(np.array([s.b for s in segs])), np.float64(tol),
              np.float64(ss[0]), np.float64(coarse), np.int32(len(sc)),
              np.float64(step), np.int32(len(ss)))
    hj, hT = g(J, cp.int32), g(Ts)
    kern = _sky_kernel(cp, "check_")
    nobj = dev.r.shape[1]
    head = (dev.r, dev.v, np.int64(nobj), np.int64(dev.n_t),
            np.float64(dev.dt))
    one = cp.zeros(1)
    oneb = cp.zeros(1, cp.bool_)
    ncell = len(J) * ng
    live = []
    for a in range(0, ncell, chunk):
        n = min(chunk, ncell - a)
        lst = cp.arange(a, a + n, dtype=cp.int64)
        alive = cp.empty(n, cp.bool_)
        kern(((n + 255) // 256,), (256,), head + (
            np.int64(n), np.int32(ng), hj, hT, lst, np.int32(1),
            np.int32(ns), np.int32(mid)) + common + (
            alive, one, one, one, oneb, np.int32(0)))
        live.append(lst[alive])
    live = cp.concatenate(live)
    n = len(live)
    if not n:
        return []
    alive = cp.empty(n, cp.bool_)
    e, tc, span = cp.empty((n, ns)), cp.empty((n, ns)), cp.empty((n, ns))
    first = cp.empty((n, ns), cp.bool_)
    kern(((n + 255) // 256,), (256,), head + (
        np.int64(n), np.int32(ng), hj, hT, live, np.int32(1), np.int32(ns),
        np.int32(mid)) + common + (alive, e, tc, span, first, np.int32(1)))
    c = cp.asnumpy(live)
    e, tc, span, first = (cp.asnumpy(x) for x in (e, tc, span, first))
    h, gi = c // ng, c % ng
    med = np.median(e, axis=1)
    # per hypothesis, its best grid site (the first of equals)
    o = np.lexsort((gi, med, h))
    best = o[np.unique(h[o], return_index=True)[1]]
    # then object by object, as the CPU goes
    jb = np.asarray(J)[h[best]]
    o = np.argsort(jb, kind="stable")
    best, jb = best[o], jb[o]
    mtc = np.median(tc[best], axis=1)
    msp = np.median(span[best], axis=1)
    cut = np.flatnonzero(np.r_[True, jb[1:] != jb[:-1], True])
    out = []
    for a, b in zip(cut[:-1], cut[1:]):
        found = sorted((float(med[best[q]]), float(mtc[q]), int(h[best[q]]),
                        int(gi[best[q]]), q) for q in range(a, b))
        kept: list = []
        for f in found:
            if all(abs(f[1] - k[1]) > merge_s for k in kept):
                kept.append(f)
        for _, t_c, _, gg, q in kept:
            out.append((int(jb[a]), t_c, float(msp[q]),
                        [bool(x) for x in first[best[q]]],
                        float(grid[gg, 0]), float(grid[gg, 1])))
    return out


def skyline_passes(P, dev: DeviceOrbits, segs):
    """`solve.candidates` with a skyline, on the GPU."""
    J, Ts = near_objects(P, dev, segs)
    return check_passes(P, dev, segs, J, Ts)


# ================================================================ join
def join_tracks(P, dev: DeviceOrbits, tracks, orders, starts,
                join_err: float = 0.15, span: float = 60.0,
                chunk: int = 100000) -> list[dict]:
    """`solve.join_tracks` for several joint-fit starts at once: each
    start's other tracks, each track's fitted objects tried in order at
    the start's solution, the first with a median end error under
    `join_err` joining. The end orders as `batchfit.end_orders` (closest
    passes over +-`span` s: the `_PASS` kernel), the errors as
    `batchfit.Batch`'s (`FitBatch`). -> one {track: (object, end order,
    median)} per start."""
    from .batchfit import _observed
    cp = dev.cp
    out: list[dict] = [{} for _ in starts]
    groups = [(si, ti) for si, (sets, _) in enumerate(starts)
              for ti, J in enumerate(orders) if J and ti not in sets]
    if not groups:
        return out
    pp = np.array([p[:4] for _, p in starts], float)          # (S, 4)
    base0 = (P.t_ref - dev.t0) * 86400
    axis, w = P.exposure_turn()
    axis = np.asarray(axis, float)
    ss = np.arange(-span, span + 0.25, 0.5)
    nobj = dev.r.shape[1]

    def g(x, t=cp.float64):
        return cp.ascontiguousarray(cp.asarray(x, t))

    # the tracks' segment ends, once each
    seg_k = {ti: np.array([float(s.k) for s in tracks[ti]])
             for ti in {t for _, t in groups}}
    # through the groups in chunks of about `chunk` objects
    nxt = 0
    while nxt < len(groups):
        take, size = [], 0
        while nxt < len(groups) and (
                not take or size + len(orders[groups[nxt][1]]) <= chunk):
            take.append(nxt)
            size += len(orders[groups[nxt][1]])
            nxt += 1
        # ---- end orders: one q per (group, segment); pairs (q, object)
        q_k, q_a, q_b, q_s, q0 = [], [], [], [], []
        for gi in take:
            si, ti = groups[gi]
            q0.append(len(q_k))
            q_k += list(seg_k[ti])
            q_a += [s.a for s in tracks[ti]]
            q_b += [s.b for s in tracks[ti]]
            q_s += [si] * len(tracks[ti])
        q_k, q_s = np.array(q_k), np.array(q_s)
        rs = P.site.gcrs(q_k, pp[q_s, 0], pp[q_s, 1], P.h0)
        vs = P.site.velocity(rs)
        xq = (base0 + np.asarray(P.s(q_k), float) * P.spf + pp[q_s, 2]) \
            / dev.dt
        pq, pj = [], []
        for n_, gi in enumerate(take):
            ti = groups[gi][1]
            J = np.asarray(orders[ti], np.int64)
            n = len(tracks[ti])
            pq.append(np.tile(q0[n_] + np.arange(n), len(J)))
            pj.append(np.repeat(J, n))
        pq = np.concatenate(pq).astype(np.int32)
        pj = np.concatenate(pj).astype(np.int32)
        m = len(pq)
        ca, cb = cp.empty(m), cp.empty(m)
        ia, ib = cp.empty(m, cp.int32), cp.empty(m, cp.int32)
        fin = cp.empty(m, cp.bool_)
        _id_kernel(cp, "pass_")(((m + 255) // 256,), (256,), (
            dev.r, dev.v, np.int64(nobj), np.int64(dev.n_t),
            np.float64(dev.dt), np.int32(m), g(pq, cp.int32),
            g(pj, cp.int32), g(xq), g(rs), g(vs), g(np.array(q_a)),
            g(np.array(q_b)), g(axis), np.float64(w), np.float64(ss[0]),
            np.float64(ss[1] - ss[0]), np.int32(len(ss)), ca, cb, ia, ib,
            fin))
        af_all = cp.asnumpy(ia < ib)
        del ca, cb, ia, ib, fin
        # ---- median end errors at each start's solution (`Batch`)
        n_max = max(len(tracks[groups[gi][1]]) for gi in take)
        M = 2 * n_max
        R = size
        j = np.zeros(R, np.int64)
        row0 = np.zeros((R, M))
        zra = np.zeros((R, M))
        sign = np.ones((R, M))
        obs = np.zeros((R, M, 3))
        free = np.zeros((R, M), bool)
        prow = np.zeros((R, 4))
        spans = []
        r0 = a0 = 0
        for gi in take:
            si, ti = groups[gi]
            J = np.asarray(orders[ti], np.int64)
            n = len(tracks[ti])
            af = af_all[a0:a0 + len(J) * n].reshape(len(J), n)
            a0 += len(J) * n
            o, fr = _observed(tracks[ti], af)
            ks = np.repeat(seg_k[ti], 2)
            sl = slice(r0, r0 + len(J))
            j[sl] = J
            row0[sl] = (base0 + np.asarray(P.s(ks[-1]), float) * P.spf) \
                / dev.dt                                   # padding
            row0[sl, :2 * n] = (base0 + np.asarray(P.s(ks), float) * P.spf) \
                / dev.dt
            zra[sl] = float(P.site.zenith_ra(ks[-1]))
            zra[sl, :2 * n] = np.asarray(P.site.zenith_ra(ks), float)
            sign[sl, :2 * n] = np.tile([-1.0, 1.0], n)
            obs[sl, :2 * n] = o
            free[sl, :2 * n] = fr
            prow[sl] = pp[si]
            spans.append((si, ti, r0, J, af))
            r0 += len(J)
        batch = FitBatch(P, dev, j, row0, zra, sign, obs, free)
        with np.errstate(invalid="ignore"):
            e = cp.linalg.norm(batch.residuals(g(prow)), axis=2)
            e[~batch.free] = cp.nan
            ok = batch.free.any(1)
            med = cp.full(R, cp.inf)
            if bool(ok.any()):
                med[ok] = cp.nanmedian(e[ok], axis=1)
        med = cp.asnumpy(med)
        del batch, e
        good = med < join_err
        for si, ti, r0, J, af in spans:
            k = np.flatnonzero(good[r0:r0 + len(J)])
            if k.size:
                k = int(k[0])
                out[si][ti] = (int(J[k]), [bool(x) for x in af[k]],
                               float(med[r0 + k]))
    return out
