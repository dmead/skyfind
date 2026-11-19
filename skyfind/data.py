"""Downloaded reference data (star catalogue, ephemerides, TLEs, map layers)
lives outside git.

Location: $SKYFIND_DATA if set, else the platform cache directory
(%LOCALAPPDATA%\\skyfind on Windows, ~/.cache/skyfind elsewhere). Files are
fetched once, on first use, and reused after that.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path


def data_dir() -> Path:
    env = os.environ.get("SKYFIND_DATA")
    if env:
        d = Path(env)
    elif sys.platform == "win32":
        d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "skyfind"
    else:
        d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) \
            / "skyfind"
    d.mkdir(parents=True, exist_ok=True)
    return d


def skyfield_loader():
    """skyfield Loader rooted in the data directory: ephemerides (de421.bsp,
    ~17 MB, from JPL) download there once, on first use."""
    from skyfield.api import Loader
    return Loader(str(data_dir()), verbose=False)


def timescale():
    """skyfield timescale from its built-in tables (no download)."""
    from skyfield.api import load
    return load.timescale(builtin=True)


def times(tt, samples: int = 257):
    """A skyfield Time for many TT jds, its IAU 2000A nutation (by far the
    dearest part of turning them into Earth-fixed positions: a 1365-term
    series per time) interpolated from `samples` times across their range
    when that's under 3 days. The series' shortest periods are days, so
    that's good to ~1e-6 arcsec."""
    import numpy as np
    from skyfield.nutationlib import iau2000a_radians
    ts = timescale()
    tt = np.asarray(tt, float)
    t = ts.tt_jd(tt)
    if tt.ndim == 1 and tt.size > 2 * samples and \
            float(tt.max() - tt.min()) < 3.0:
        grid = ts.tt_jd(np.linspace(float(tt.min()), float(tt.max()),
                                    samples))
        dpsi, deps = iau2000a_radians(grid)
        t._nutation_angles_radians = (np.interp(tt, grid.tt, dpsi),
                                      np.interp(tt, grid.tt, deps))
    return t


def fetch(url: str, name: str, *, quiet: bool = False) -> Path:
    """Return data_dir()/name, downloading it from `url` if missing.

    Downloads to a temporary name and renames on success, so an interrupted
    download never leaves a truncated file that looks complete.
    """
    dest = data_dir() / name
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    if not quiet:
        print(f"downloading {name} from {url} -> {dest.parent}",
              file=sys.stderr, flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest
