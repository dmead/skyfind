"""SGP4 for the orbit table (`orbits`): python-sgp4's C++ extension where
it is installed (the desktop), else the same Vallado code built for Android
(`android/native`: libsgp4c.so, through ctypes; the app sets SKYFIND_SGP4C
to it), else python-sgp4's pure Python (right, but far too slow for a
catalogue). The two used here, `Satrec.twoline2rv(line1, line2)` and
`SatrecArray(sats).sgp4(jd, fr) -> (e, r, v)`, behave the same either way:
the Android library does what the extension's wrapper does."""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from sgp4.api import Satrec as _Satrec, SatrecArray as _SatrecArray
    from sgp4.api import accelerated as _accelerated
except ImportError:                                     # pragma: no cover
    _Satrec = _SatrecArray = None
    _accelerated = False


def _load():
    path = os.environ.get("SKYFIND_SGP4C")
    if not path or not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    p, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    dp = ctypes.POINTER(ctypes.c_double)
    lib.sgp4c_size.restype = i
    lib.sgp4c_twoline2rv.argtypes = [p, i, ctypes.c_char_p, ctypes.c_char_p]
    lib.sgp4c_twoline2rv.restype = i
    lib.sgp4c_epoch.argtypes = [p, i, dp, dp]
    lib.sgp4c_array.argtypes = [p, i, i, p, p, i, p, p, p]
    lib.sgp4c_array.restype = None
    return lib


_lib = None if _accelerated else _load()
NATIVE = _accelerated or _lib is not None


if _accelerated or _lib is None:
    Satrec, SatrecArray = _Satrec, _SatrecArray
else:
    _SIZE = _lib.sgp4c_size()
    _THREADS = os.cpu_count() or 4

    class Satrec:
        """One element set (from its two lines), held as the library's
        struct."""

        __slots__ = ("buf", "error", "jdsatepoch", "jdsatepochF")

        @classmethod
        def twoline2rv(cls, line1: str, line2: str) -> "Satrec":
            s = cls()
            s.buf = ctypes.create_string_buffer(_SIZE)
            s.error = _lib.sgp4c_twoline2rv(s.buf, 0, line1.encode("ascii"),
                                            line2.encode("ascii"))
            jd, fr = ctypes.c_double(), ctypes.c_double()
            _lib.sgp4c_epoch(s.buf, 0, ctypes.byref(jd), ctypes.byref(fr))
            s.jdsatepoch, s.jdsatepochF = jd.value, fr.value
            return s

    class SatrecArray:
        """Element sets side by side, propagated together (the objects
        split across threads: the library runs without the GIL)."""

        def __init__(self, sats):
            self.n = len(sats)
            self.buf = ctypes.create_string_buffer(max(_SIZE * self.n, 1))
            base = ctypes.addressof(self.buf)
            for i, s in enumerate(sats):
                ctypes.memmove(base + i * _SIZE, s.buf, _SIZE)

        def sgp4(self, jd, fr):
            jd = np.ascontiguousarray(jd, np.float64)
            fr = np.ascontiguousarray(fr, np.float64)
            nt = len(jd)
            e = np.zeros((self.n, nt), np.uint8)
            r = np.empty((self.n, nt, 3), np.float64)
            v = np.empty((self.n, nt, 3), np.float64)
            ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)   # noqa: E731

            def part(lo_hi):
                lo, hi = lo_hi
                _lib.sgp4c_array(self.buf, lo, hi, ptr(jd), ptr(fr), nt,
                                 ptr(e), ptr(r), ptr(v))

            k = max(1, min(_THREADS, self.n // 64))
            bounds = np.linspace(0, self.n, k + 1).astype(int)
            spans = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])
                     if b > a]
            if not spans:                       # no objects: empty arrays
                pass
            elif len(spans) == 1:
                part(spans[0])
            else:
                with ThreadPoolExecutor(len(spans)) as ex:
                    list(ex.map(part, spans))
            return e, r, v
