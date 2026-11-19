"""Bright-star reference catalogue for wide-field plate solving.

Hipparcos (ESA 1997, CDS I/239): ~118k stars, complete to V ~ 7.3, with
proper motions. That suits wide-field nightscapes, which solve on the
brightest stars — exactly where Gaia DR3 is incomplete (G < ~3).

The raw catalogue (hip_main.dat, ~51 MB) is downloaded once into the data
directory (see `skyfind.data`) and reduced to a small cached
extract; nothing is stored in the repository. A local file can be used
instead via `path=` — a .npz or .csv with ra, dec (deg, ICRS) and mag
columns (optionally pmra, pmdec in mas/yr, cos(dec) included in pmra).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data

HIP_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"
HIP_EPOCH = 1991.25
_EXTRACT = "hipparcos_v10.npz"          # stars to V = 10, a few MB


@dataclass(frozen=True)
class Stars:
    ra: np.ndarray        # deg, ICRS, at `epoch`
    dec: np.ndarray       # deg
    mag: np.ndarray       # V (or the file's magnitude)
    ident: np.ndarray     # catalogue number (HIP) or row index
    epoch: float

    def __len__(self) -> int:
        return len(self.ra)

    def name(self, i: int) -> str:
        return f"HIP {int(self.ident[i])}"


def parse_hip_main(lines) -> dict[str, np.ndarray]:
    """Parse hip_main.dat rows (pipe-separated, CDS I/239 byte layout).

    Fields used: H1 HIP, H5 Vmag, H8/H9 RA/Dec (deg, ICRS, J1991.25),
    H12/H13 pmRA*cos(dec)/pmDec (mas/yr). Rows without a position or a
    magnitude are skipped (a few hundred lack Hipparcos astrometry).
    """
    hip, ra, de, pmra, pmde, vmag = [], [], [], [], [], []
    for ln in lines:
        f = ln.split("|")
        if len(f) < 14:
            continue
        try:
            r, d, v = float(f[8]), float(f[9]), float(f[5])
        except ValueError:
            continue
        try:
            pa, pd = float(f[12]), float(f[13])
        except ValueError:
            pa = pd = 0.0
        hip.append(int(f[1])); ra.append(r); de.append(d)
        pmra.append(pa); pmde.append(pd); vmag.append(v)
    return dict(hip=np.array(hip, np.int32), ra=np.array(ra), dec=np.array(de),
                pmra=np.array(pmra), pmdec=np.array(pmde),
                mag=np.array(vmag, np.float32))


def _hipparcos() -> dict[str, np.ndarray]:
    ext = data.data_dir() / _EXTRACT
    if ext.exists():
        return dict(np.load(ext))
    raw = data.fetch(HIP_URL, "hip_main.dat")
    with open(raw, encoding="ascii", errors="ignore") as fh:
        cat = parse_hip_main(fh)
    keep = cat["mag"] <= 10.0
    cat = {k: v[keep] for k, v in cat.items()}
    np.savez(ext, **cat)
    raw.unlink()                      # only the extract is ever read again
    return cat


def _local(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        z = dict(np.load(path))
    else:
        z = np.genfromtxt(path, delimiter=",", names=True)
        z = {k: np.asarray(z[k]) for k in z.dtype.names}
    out = dict(ra=z["ra"], dec=z["dec"], mag=z["mag"],
               pmra=z.get("pmra", np.zeros_like(z["ra"])),
               pmdec=z.get("pmdec", np.zeros_like(z["ra"])),
               hip=z.get("hip", np.arange(len(z["ra"]))))
    return out


def bright_stars(maglim: float = 6.5, *, epoch: float | None = None,
                 path: str | Path | None = None,
                 path_epoch: float = 2000.0) -> Stars:
    """Stars brighter than `maglim`, propagated to `epoch` (decimal year).

    Proper motion matters here: over three decades the fastest bright stars
    move arcminutes — a pixel or more on a 4K wide-field frame. A local
    file's positions are taken to be at `path_epoch` (Gaia DR3: 2016.0).
    """
    cat = _local(Path(path)) if path else _hipparcos()
    base = path_epoch if path else HIP_EPOCH
    keep = cat["mag"] <= maglim
    ra, dec = cat["ra"][keep], cat["dec"][keep]
    if epoch is not None:
        dt = epoch - base
        cosd = np.cos(np.radians(dec))
        ra = ra + cat["pmra"][keep] * dt / 3.6e6 / np.maximum(cosd, 1e-6)
        dec = dec + cat["pmdec"][keep] * dt / 3.6e6
    order = np.argsort(cat["mag"][keep])
    return Stars(ra=(ra % 360)[order], dec=dec[order],
                 mag=cat["mag"][keep][order], ident=cat["hip"][keep][order],
                 epoch=epoch if epoch is not None else base)
