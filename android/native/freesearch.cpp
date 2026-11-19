// The free-site pass search's kernels in C++ threads, for a machine
// without CUDA (the phone): gpu.py's _STAGE1, _EXACT and _REFINE, the same
// tests in the same order, called through ctypes by skyfind/native.py
// (freesearch.run does the rest).
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

int n_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int) n : 4;
}

template <class F>
void parallel(long long n, F fn) {
    int k = n_threads();
    if (n < 4096) k = 1;
    std::vector<std::thread> th;
    for (int t = 0; t < k; t++)
        th.emplace_back([=] {
            for (long long i = n * t / k; i < n * (t + 1) / k; i++) fn(i);
        });
    for (auto &x : th) x.join();
}

void at_row(const float *R, const float *V, long long nobj, long long nt,
            double dt, long long j, double x, double *out) {
    long long i = (long long) floor(x);
    if (i < 0) i = 0;
    if (i > nt - 2) i = nt - 2;
    double u = x - i, u2 = u * u, u3 = u2 * u;
    double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u,
           h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    long long a0 = (i * nobj + j) * 3, a1 = ((i + 1) * nobj + j) * 3;
    for (int k = 0; k < 3; k++)
        out[k] = h00 * (double) R[a0 + k] + h10 * ((double) V[a0 + k] * dt)
               + h01 * (double) R[a1 + k] + h11 * ((double) V[a1 + k] * dt);
}

bool ground84(const double *p, const double *d, double h0, double A,
              double e2, double *x, double *range) {
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

}  // namespace

extern "C" {

// Every (row, object) against every track in single precision, with
// margins; survivors (row, object, track) into out_*, up to max_out.
// Returns how many there were (more than max_out: call again with room).
long long fs_stage1(
    const float *r, const float *v, long long nobj, long long row_lo,
    long long nrows, int ntr, const int *t_lo, const int *t_hi,
    const float *d0, const float *sky, int nsky, const float *sun,
    float dark, float Rs, float A, float e2, float h0, float WE,
    const float *pole, const float *u0, const float *rate_lo,
    const float *rate_hi, float cosd, float margin, int *out_row,
    int *out_j, int *out_t, long long max_out) {
    std::atomic<long long> count(0);
    parallel(nrows * nobj, [&](long long idx) {
        long long row = row_lo + idx / nobj;
        long long j = idx % nobj;
        long long o = (row * nobj + j) * 3;
        float px = r[o], py = r[o + 1], pz = r[o + 2];
        if (std::isnan(px)) return;
        float vx = 0, vy = 0, vz = 0;
        bool have_v = false;
        for (int t = 0; t < ntr; t++) {
            if (row < t_lo[t] || row > t_hi[t]) continue;
            const float *d = d0 + 3 * t;
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
            const float *sn = sun + 3 * t;
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
            const float *u = u0 + 3 * t;
            if (!((wx * u[0] + wy * u[1] + wz * u[2]) / wm > cosd - 0.01f))
                continue;
            // the frame's sky above the ground point: the dearest test, last
            const float *sk = sky + (long long) t * nsky * 3;
            bool up = true;
            for (int k = 0; k < nsky; k++)
                if (!(ux * sk[3 * k] + uy * sk[3 * k + 1] + uz * sk[3 * k + 2]
                      > -margin)) { up = false; break; }
            if (!up) continue;
            long long k = count.fetch_add(1);
            if (k < max_out) {
                out_row[k] = (int) row; out_j[k] = (int) j; out_t[k] = t;
            }
        }
    });
    return count.load();
}

// The survivors' tests again in double precision (_EXACT).
void fs_exact(
    const float *r, const float *v, long long nobj, int n, const int *srow,
    const int *sj, const int *st, const double *d0, const double *sky,
    int nsky, const double *sun, double dark, double A, double e2,
    double h0, double WE, const double *pole, const double *u0,
    const double *rate_lo, const double *rate_hi, double cosd, uint8_t *ok,
    double *xout) {
    parallel(n, [&](long long i) {
        ok[i] = 0;
        int t = st[i];
        long long o = ((long long) srow[i] * nobj + sj[i]) * 3;
        double p[3] = {r[o], r[o + 1], r[o + 2]};
        const double *d = d0 + 3 * t;
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
        const double *sk = sky + (long long) t * nsky * 3;
        double mn = 3.0;
        for (int k = 0; k < nsky; k++) {
            double c = ux * sk[3 * k] + uy * sk[3 * k + 1] + uz * sk[3 * k + 2];
            if (c < mn) mn = c;
        }
        if (!(mn > 0)) return;
        const double *sn = sun + 3 * t;
        if (!(ux * sn[0] + uy * sn[1] + uz * sn[2] < dark)) return;
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
        const double *u = u0 + 3 * t;
        if (!((w0 / wm) * u[0] + (w1 / wm) * u[1] + (w2 / wm) * u[2] > cosd))
            return;
        ok[i] = 1;
        xout[3 * i] = y[0]; xout[3 * i + 1] = y[1]; xout[3 * i + 2] = y[2];
    });
}

// Each survivor refined across its table row, then finely around the best
// (_REFINE); NaN wins its comparisons as CuPy's argmax lets it.
void fs_refine(
    const float *R, const float *V, long long nobj, long long nt, double dt,
    int n, const int *srow, const int *sj, const int *st, const double *x0,
    const double *d0, const double *chkM, const double *chkd,
    const double *chkdt, double A, double e2, double h0,
    const double *cgrid, int nc, const double *fgrid, int nf,
    double *top_out, double *dt_out, double *x_out) {
    parallel(n, [&](long long i) {
        int t = st[i];
        long long j = sj[i];
        double row = srow[i];
        const double *d = d0 + 3 * t;
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
                    const double *M = chkM + (t * 2 + c) * 9;
                    const double *dc = chkd + (t * 2 + c) * 3;
                    double q[3];
                    at_row(R, V, nobj, nt, dt, j, xr + chkdt[t * 2 + c] / dt, q);
                    for (int m = 0; m < 3; m++)
                        q[m] -= xd[0] * M[m] + xd[1] * M[3 + m] + xd[2] * M[6 + m];
                    double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                    double cv = (q[0] / qn) * dc[0] + (q[1] / qn) * dc[1]
                              + (q[2] / qn) * dc[2];
                    worst = (std::isnan(cv) || std::isnan(worst) || cv < worst) ?
                        (std::isnan(worst) ? worst : cv) : worst;
                }
                if (std::isnan(worst)) {
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
    });
}

}  // extern "C"
