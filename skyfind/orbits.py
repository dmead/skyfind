"""Where every catalogued object is, fast, over a whole night (and the
window of clock hypotheses around it).

SGP4 for every object on a regular time grid (default 20 s), rotated TEME ->
GCRS once, stored as float32 positions and velocities; any time in between
is a cubic Hermite interpolation (position + velocity at both grid ends), good
to ~0.05 km in LEO. The table is the solver's hot data (~3 GB for today's
catalogue over +-12 h). Worker processes share one copy: in shared memory
(`shared=True`: never touches the disk), or as memory-mapped .npy files in a
folder. Several processes propagate it at once, each filling its own block
of (times x objects), so every block is written by one process in rows.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np

from . import data, par

CHUNK = 3000                    # objects per SGP4 batch (bounds memory)
ROWS = 256                      # table rows (times) per propagation job


def sgp4_times(t):
    """(jd, fraction) in UTC for SGP4, as skyfield does it (TLE epochs are
    UTC; UT1 would put LEO objects up to ~7 km off)."""
    jd = np.atleast_1d(t.whole)
    fr = np.atleast_1d(t.tai_fraction - t._leap_seconds() / 86400)
    return jd, fr


def _grid(t_start_tt: float, rows, grid_dt: float):
    """SGP4's times and the GCRS -> TEME rotations at table `rows`."""
    from skyfield.sgp4lib import TEME
    times = data.timescale().tt_jd(t_start_tt + np.asarray(rows) * grid_dt
                                   / 86400)
    jd, fr = sgp4_times(times)
    return jd, fr, TEME.rotation_at(times)          # (3,3,n_t) gcrs -> teme


def _rotate(R, x):
    """TEME (sats, times, 3) -> GCRS (times, sats, 3), R (3, 3, times):
    written out term by term so that any split of the table into blocks
    gives the same bits (the parallel table equals the serial one)."""
    out = x[:, :, 0, None] * R[0].T[None] + x[:, :, 1, None] * R[1].T[None] \
        + x[:, :, 2, None] * R[2].T[None]
    return out.transpose(1, 0, 2)


def _fill(r, v, sats, c0, rows, jd, fr, R) -> None:
    """Propagate `sats` into table rows `rows` (a slice), columns c0.. of
    r, v (GCRS; nan where SGP4 fails, e.g. decayed)."""
    from .sgp4lib import SatrecArray
    for c in range(0, len(sats), CHUNK):
        part = sats[c:c + CHUNK]
        e, rr, vv = SatrecArray(part).sgp4(jd, fr)
        sl = slice(c0 + c, c0 + c + len(part))
        block = _rotate(R, rr)
        block[(e != 0).T] = np.nan
        r[rows, sl] = block
        v[rows, sl] = _rotate(R, vv)


def _fill_block(args) -> int:
    """Worker: propagate TLE `lines` (objects c0..) over table rows
    lo..hi into the shared table."""
    where, lines, c0, t_start, lo, hi, grid_dt = args
    from .sgp4lib import Satrec
    tab = _Table.attach(where)
    try:
        sats = [Satrec.twoline2rv(a, b) for a, b in lines]
        _fill(tab.r, tab.v, sats, c0, slice(lo, hi),
              *_grid(t_start, np.arange(lo, hi), grid_dt))
        tab.flush()
    finally:
        tab.close()
    return len(lines)


class NoRoom(MemoryError):
    """Not enough memory to share the table: put it in files instead."""


def _attach_shm(name: str):
    """A block the table's creator made, attached without telling the
    resource tracker (Linux, macOS): the workers share the creator's
    tracker, and whatever one registers or unregisters there is the
    creator's registration (an unregister after use cancelled it, and the
    creator's own unlink then failed in the tracker). The creator alone
    registers and frees it. Windows has no tracker for these."""
    from multiprocessing import resource_tracker, shared_memory
    if os.name == "nt":
        return shared_memory.SharedMemory(name=name)
    try:                                   # Python 3.13+
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        pass
    with _ATTACH:                          # threads attaching at once
        register = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = register


_ATTACH = threading.Lock()


class _Table:
    """The r and v arrays with how another process finds them: shared
    memory blocks by name, or .npy files in a folder."""

    def __init__(self, r, v, where: dict, shm=()):
        self.r, self.v, self.where, self.shm = r, v, where, list(shm)

    @classmethod
    def create(cls, shape, *, shared: bool, folder=None) -> "_Table":
        if shared:
            from multiprocessing import shared_memory
            nbytes = int(np.prod(shape)) * 4
            shm = []
            try:
                for _ in range(2):
                    shm.append(shared_memory.SharedMemory(create=True,
                                                          size=nbytes))
            except OSError as e:
                for x in shm:
                    x.close()
                    x.unlink()
                raise NoRoom(f"{2 * nbytes / 2**30:.1f} GB: {e}") from e
            r, v = (np.ndarray(shape, np.float32, buffer=s.buf) for s in shm)
            return cls(r, v, dict(shm=[s.name for s in shm],
                                  shape=list(shape)), shm)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("orbits_r.npy", "orbits_v.npy"):
            np.lib.format.open_memmap(folder / name, mode="w+",
                                      dtype=np.float32, shape=shape).flush()
        return cls.attach(dict(dir=str(folder.resolve()), shape=list(shape)),
                          mode="r+")

    @classmethod
    def attach(cls, where: dict, mode: str = "r+") -> "_Table":
        shape = tuple(where["shape"])
        if "shm" in where:
            shm = [_attach_shm(n) for n in where["shm"]]
            r, v = (np.ndarray(shape, np.float32, buffer=s.buf) for s in shm)
            return cls(r, v, where, shm)
        d = Path(where["dir"])
        return cls(np.load(d / "orbits_r.npy", mmap_mode=mode),
                   np.load(d / "orbits_v.npy", mmap_mode=mode), where)

    def flush(self) -> None:
        for a in (self.r, self.v):
            if isinstance(a, np.memmap):
                a.flush()

    def close(self, unlink: bool = False) -> None:
        self.r = self.v = None
        for s in self.shm:
            try:
                s.close()
            except BufferError:          # a view still out there: the OS
                pass                     # frees it with the last process
            if unlink:
                try:
                    s.unlink()
                except (FileNotFoundError, OSError):
                    pass
        self.shm = []


class Orbits:
    """Positions of catalogued objects (GCRS, km) between TT Julian dates."""

    def __init__(self, r: np.ndarray, v: np.ndarray, t0_tt: float,
                 grid_dt: float, table: _Table | None = None,
                 owner: bool = False):
        self.r, self.v = r, v            # (n_t, n_sat, 3) float32
        self.t0 = float(t0_tt)
        self.dt = float(grid_dt)
        self._table = table
        self._owner = owner

    @property
    def shared(self) -> bool:
        return bool(self._table and self._table.shm)

    @classmethod
    def propagate(cls, sats, t_start_tt: float, t_end_tt: float, *,
                  grid_dt: float = 20.0, folder=None, workers: int = 1,
                  pool=None, shared: bool = False, on_block=None,
                  log=print) -> "Orbits":
        """`sats`: TLE (line1, line2) pairs, or sgp4 Satrec objects (in
        process only). With `shared` (shared memory) or a `folder` (.npy
        files), the table can be handed to other processes, and `workers`
        processes — or `pool`'s (a `solve.Workers`) — fill it in blocks;
        `on_block(r, v, rows, cols)` is called as each is done (slices)."""
        from .sgp4lib import Satrec
        n_t = int(np.ceil((t_end_tt - t_start_tt) * 86400 / grid_dt)) + 2
        n = len(sats)
        tic = time.time()
        if not shared and folder is None:
            r = np.empty((n_t, n, 3), np.float32)
            v = np.empty((n_t, n, 3), np.float32)
            objs = [s if isinstance(s, Satrec) else Satrec.twoline2rv(*s)
                    for s in sats]
            _fill(r, v, objs, 0, slice(None),
                  *_grid(t_start_tt, np.arange(n_t), grid_dt))
            orb = cls(r, v, t_start_tt, grid_dt)
        else:
            tab = _Table.create((n_t, n, 3), shared=shared, folder=folder)
            nw = pool.n if pool is not None else workers
            lines = [tuple(s) for s in sats]
            jobs = _blocks(tab.where, lines, t_start_tt, n_t, grid_dt, nw)
            def done(job):
                if on_block is not None:
                    _, part, c0, _, lo, hi, _ = job
                    on_block(tab.r, tab.v, slice(lo, hi),
                             slice(c0, c0 + len(part)))

            def run(ex):
                futs = {ex.submit(_fill_block, job): job for job in jobs}
                for f in as_completed(futs):
                    f.result()
                    done(futs[f])

            try:
                if pool is not None:
                    run(pool.ex)
                elif nw <= 1 or n < 2 * nw:
                    for job in jobs:
                        _fill_block(job)
                        done(job)
                else:
                    with par.executor(nw) as ex:
                        run(ex)
            except BaseException:
                tab.close(unlink=True)
                raise
            tab.flush()
            orb = cls(tab.r, tab.v, t_start_tt, grid_dt, tab, owner=True)
            if folder is not None:
                orb.save(folder)
        if log:
            log(f"propagated {n} objects x {n_t} times "
                f"({(t_end_tt - t_start_tt) * 24:.1f} h) in "
                f"{time.time() - tic:.0f} s"
                + (f" ({pool.n if pool is not None else workers} processes)"
                   if (shared or folder) and (pool is not None or workers > 1)
                   else ""))
        return orb

    # ---------------------------------------------------------- io
    def save(self, folder, *, copy: bool = False) -> None:
        """Make the table loadable from `folder` (`load`): a note of where
        it is when it's shared or already there, else (or with `copy`) the
        arrays themselves as .npy files."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta = dict(t0_tt=self.t0, grid_dt=self.dt, shape=list(self.r.shape))
        src = Path(self.r.filename).resolve().parent \
            if isinstance(self.r, np.memmap) else None
        here = src is not None and src == folder.resolve()
        if self.shared and not copy:
            meta["shm"] = self._table.where["shm"]
        elif src is not None and not here and not copy:
            meta["dir"] = str(src)                   # files elsewhere
        elif not here:
            np.save(folder / "orbits_r.npy", self.r)
            np.save(folder / "orbits_v.npy", self.v)
        (folder / "orbits.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, folder, mmap: bool = True) -> "Orbits":
        folder = Path(folder)
        meta = json.loads((folder / "orbits.json").read_text())
        if "shm" in meta:
            tab = _Table.attach(meta)
            return cls(tab.r, tab.v, meta["t0_tt"], meta["grid_dt"], tab)
        mode = "r" if mmap else None
        d = Path(meta.get("dir", folder))
        return cls(np.load(d / "orbits_r.npy", mmap_mode=mode),
                   np.load(d / "orbits_v.npy", mmap_mode=mode),
                   meta["t0_tt"], meta["grid_dt"])

    def close(self) -> None:
        """Drop the arrays (a memory-mapped file can't be deleted on Windows
        while mapped) and let go of shared memory (freed with the last
        process holding it)."""
        self.r = self.v = None
        if self._table is not None:
            self._table.close(unlink=self._owner)
            self._table = None


    # ---------------------------------------------------------- lookup
    @property
    def t_end(self) -> float:
        return self.t0 + (len(self.r) - 1) * self.dt / 86400

    @property
    def n_objects(self) -> int:
        return self.r.shape[1]

    def at(self, tt: float, idx=None) -> np.ndarray:
        """(n, 3) positions at one TT jd (all objects, or rows `idx`)."""
        x = (tt - self.t0) * 86400 / self.dt
        i = int(np.clip(np.floor(x), 0, len(self.r) - 2))
        u = x - i
        sl = slice(None) if idx is None else idx
        p0, p1 = self.r[i][sl], self.r[i + 1][sl]
        m0, m1 = self.v[i][sl] * self.dt, self.v[i + 1][sl] * self.dt
        return _hermite(u, p0, m0, p1, m1)

    def at_many(self, tt: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Positions of objects `j` at TT jds `tt` (same shapes) -> (..., 3)."""
        x = (np.asarray(tt) - self.t0) * 86400 / self.dt
        i = np.clip(np.floor(x).astype(np.int64), 0, len(self.r) - 2)
        u = (x - i)[..., None]
        p0, p1 = self.r[i, j], self.r[i + 1, j]
        m0, m1 = self.v[i, j] * self.dt, self.v[i + 1, j] * self.dt
        return _hermite(u, p0, m0, p1, m1)


def _hermite(u, p0, m0, p1, m1):
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0
            + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1)


def _blocks(where: dict, lines, t_start: float, n_t: int, grid_dt: float,
            workers: int) -> list:
    """Propagation jobs: blocks of ~`ROWS` table rows x a share of the
    objects, ~8 per worker so the slower cores (efficiency cores, a second
    thread on a core) don't leave the rest waiting at the end. More
    objects groups rather than more row blocks: each job parses its
    objects' elements."""
    n_rows = -(-n_t // ROWS)
    n_obj = max(1, min(-(-8 * workers // n_rows), len(lines) // 500))
    cols = np.linspace(0, len(lines), n_obj + 1).astype(int)
    return [(where, lines[a:b], int(a), t_start, lo, min(lo + ROWS, n_t),
             grid_dt)
            for a, b in zip(cols[:-1], cols[1:]) if b > a
            for lo in range(0, n_t, ROWS)]
