import numpy as np

from skyfind import data
from skyfind import catalog

# two genuine-layout hip_main.dat rows (Betelgeuse, Aldebaran) and one row
# without astrometry, which must be skipped
ROWS = [
    "H|       27989| |05 55 10.29|+07 24 25.3| 0.45|3|H|088.79287163|+07.40703634| |   7.63|   27.33|   10.86|  1.64|  1.02|  1.64|  1.17|  1.01| ",
    "H|       21421| |04 35 55.20|+16 30 35.1| 0.87| |H|068.98000195|+16.50976164| |  50.09|   62.78| -189.36|  0.95|  0.63|  0.95|  0.84|  0.61| ",
    "H|      120404| |01 23 45.67|-12 34 56.7|11.20| |H|            |            | |       |        |        |      |      |      |      |      | ",
]


def test_parse_hip_main_rows():
    cat = catalog.parse_hip_main(ROWS)
    assert list(cat["hip"]) == [27989, 21421]
    assert np.allclose(cat["ra"], [88.79287163, 68.98000195])
    assert np.allclose(cat["pmdec"], [10.86, -189.36])


def test_bright_stars_from_local_file_with_proper_motion(tmp_path):
    cat = catalog.parse_hip_main(ROWS)
    f = tmp_path / "mine.npz"
    np.savez(f, ra=cat["ra"], dec=cat["dec"], mag=cat["mag"],
             pmra=cat["pmra"], pmdec=cat["pmdec"], hip=cat["hip"])
    s = catalog.bright_stars(1.0, path=f, path_epoch=1991.25)
    assert len(s) == 2 and s.name(0) == "HIP 27989"      # brightest first
    assert np.isclose(s.dec[1], 16.50976164)             # no epoch: as stored
    # Aldebaran's -189 mas/yr in dec: ~1.9 arcmin south after 33.5 years
    moved = catalog.bright_stars(1.0, path=f, path_epoch=1991.25,
                                 epoch=2024.75)
    assert np.isclose(moved.dec[1] - 16.50976164, -189.36 * 33.5 / 3.6e6)


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYFIND_DATA", str(tmp_path / "d"))
    assert data.data_dir() == tmp_path / "d"
    assert (tmp_path / "d").is_dir()
