"""The free-site pass search's kernels compiled for the CPU (the phone has
no CUDA and its Python threads share one GIL): gpu.py's kernels in C++
threads (`android/native/freesearch.cpp`), driven by `freesearch.run` as
the GPU's are. The library is named by SKYFIND_NATIVE (the app sets it);
without it the search stays in Python (`solve.free_passes`)."""

from __future__ import annotations

import ctypes
import os

import numpy as np

_lib = None


def available() -> bool:
    return _load() is not None


def _load():
    global _lib
    if _lib is None:
        path = os.environ.get("SKYFIND_NATIVE")
        if not path or not os.path.exists(path):
            return None
        lib = ctypes.CDLL(path)
        p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        f, d = ctypes.c_float, ctypes.c_double
        lib.fs_stage1.restype = ll
        lib.fs_stage1.argtypes = [p, p, ll, ll, ll, i, p, p, p, p, i, p, f, f,
                                  f, f, f, f, p, p, p, p, f, f, p, p, p, ll]
        lib.fs_exact.restype = None
        lib.fs_exact.argtypes = [p, p, ll, i, p, p, p, p, p, i, p, d, d, d, d,
                                 d, p, p, p, p, d, p, p]
        lib.fs_refine.restype = None
        lib.fs_refine.argtypes = [p, p, ll, ll, d, i, p, p, p, p, p, p, p, p,
                                  d, d, d, p, i, p, i, p, p, p]
        _lib = lib
    return _lib


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def free_passes_all(P, tracks, **kw) -> list[list]:
    """`solve.free_passes` for every track at once, in native threads."""
    from . import freesearch
    o = P.orbits
    return freesearch.run(P, tracks, _NativeFree(o), n_t=len(o.r),
                          step=o.dt, t0=o.t0, **kw)


class _NativeFree:
    """The free search's kernels in C++ threads, for `freesearch.run`."""

    xp = np

    def __init__(self, orbits, max_out: int = 1 << 22):
        self.lib = _load()
        self.r = np.ascontiguousarray(orbits.r, np.float32)
        self.v = np.ascontiguousarray(orbits.v, np.float32)
        self.dt = float(orbits.dt)
        self.max_out = max_out
        self.keep = []                       # arrays the calls point into

    @staticmethod
    def get(x):
        return np.asarray(x)

    def _c(self, x, t):
        a = np.ascontiguousarray(x, t)
        self.keep.append(a)
        return _p(a)

    def stage1(self, a):
        f32, i32 = np.float32, np.int32
        nobj = self.r.shape[1]
        max_out = self.max_out
        while True:
            s_row = np.empty(max_out, i32)
            s_j = np.empty(max_out, i32)
            s_t = np.empty(max_out, i32)
            n = self.lib.fs_stage1(
                _p(self.r), _p(self.v), nobj, a["row_lo"], a["nrows"],
                len(a["t_lo"]), self._c(a["t_lo"], i32),
                self._c(a["t_hi"], i32), self._c(a["d0s"], f32),
                self._c(a["skys"], f32), a["skys"].shape[1],
                self._c(a["suns"], f32), a["dark"], 6371.0 + a["h0"], a["A"],
                a["e2"], a["h0"], a["WE"], self._c(a["pole"], f32),
                self._c(a["u0s"], f32), self._c(a["r_lo"], f32),
                self._c(a["r_hi"], f32), a["cosd"], 1e-4, _p(s_row),
                _p(s_j), _p(s_t), max_out)
            if n <= max_out:
                break
            max_out = n + 1024                   # too many: again, with room
        return s_row[:n].copy(), s_j[:n].copy(), s_t[:n].copy()

    def exact(self, a, s_row, s_j, s_t):
        f64, i32 = np.float64, np.int32
        n = len(s_row)
        ok = np.zeros(n, np.uint8)
        x = np.empty((n, 3))
        self.lib.fs_exact(
            _p(self.r), _p(self.v), self.r.shape[1], n,
            self._c(s_row, i32), self._c(s_j, i32), self._c(s_t, i32),
            self._c(a["d0s"], f64), self._c(a["skys"], f64),
            a["skys"].shape[1], self._c(a["suns"], f64), a["dark"], a["A"],
            a["e2"], a["h0"], a["WE"], self._c(a["pole"], f64),
            self._c(a["u0s"], f64), self._c(a["r_lo"], f64),
            self._c(a["r_hi"], f64), a["cosd"], _p(ok), _p(x))
        return ok.astype(bool), x

    def refine(self, a, s_row, s_j, s_t, x, coarse, fine, M, dc, dts):
        f64, i32 = np.float64, np.int32
        n = len(s_row)
        top, top_dt, xr = np.empty(n), np.empty(n), np.empty((n, 3))
        self.lib.fs_refine(
            _p(self.r), _p(self.v), self.r.shape[1], len(self.r), self.dt, n,
            self._c(s_row, i32), self._c(s_j, i32), self._c(s_t, i32),
            self._c(x, f64), self._c(a["d0s"], f64), self._c(M, f64),
            self._c(dc, f64), self._c(dts, f64), a["A"], a["e2"], a["h0"],
            self._c(coarse, f64), len(coarse), self._c(fine, f64), len(fine),
            _p(top), _p(top_dt), _p(xr))
        return top, top_dt, xr
