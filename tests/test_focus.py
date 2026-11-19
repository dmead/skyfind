"""Focus by star size on rendered star fields: a lens whose blur grows
either side of a best position the sweep has to find."""

import numpy as np
import pytest

from skyfind import focus


def field(x, x0, *, seed=3, n=140, w=960, h=720, best=0.7, slope=9.0,
          noise=2.0, shot=0):
    """A star field through a lens at position `x` whose focus is at `x0`:
    Gaussian stars of sigma sqrt(best^2 + (slope (x - x0))^2) px, the same
    stars and fluxes at every position."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([20, 20], [w - 20, h - 20], (n, 2))
    flux = 400 * 10 ** rng.uniform(0, 1.6, n)
    s = float(np.hypot(best, slope * (x - x0)))
    img = np.full((h, w), 14.0, np.float32)
    r = int(np.ceil(4 * s)) + 2
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    for (px, py), f in zip(xy, flux):
        ix, iy = int(round(px)), int(round(py))
        if ix - r < 0 or iy - r < 0 or ix + r >= w or iy + r >= h:
            continue
        g = np.exp(-((ox + ix - px) ** 2 + (oy + iy - py) ** 2) / (2 * s * s))
        img[iy - r:iy + r + 1, ix - r:ix + r + 1] += f * g / (2 * np.pi * s * s)
    img += np.random.default_rng(1000 * seed + shot).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run(x0, lo=0.0, hi=1.5, **kw):
    sw = focus.Sweep(lo, hi)
    shot = 0
    while (x := sw.next()) is not None:
        shot += 1
        sw.add([field(x, x0, shot=shot * 10 + i, **kw) for i in range(2)])
    return sw


def test_hfr_follows_the_blur():
    sizes = [focus.measure(field(x, 0.5))["hfr"] for x in (0.5, 0.7, 0.9)]
    assert sizes[0] < sizes[1] < sizes[2]
    # a Gaussian's flux-weighted mean radius is sigma sqrt(pi / 2)
    assert sizes[2] == pytest.approx(np.hypot(0.7, 9 * 0.4) * 1.2533, rel=0.2)
    sharp = [focus.measure(field(x, 0.5))["sharp"] for x in (0.5, 0.7, 0.9)]
    assert sharp[0] > sharp[1] > sharp[2]


def test_no_stars_measures_nothing():
    blank = np.random.default_rng(0).normal(14, 2, (480, 640))
    m = focus.measure(np.clip(blank, 0, 255).astype(np.uint8))
    assert m["hfr"] is None and m["n"] < focus.MIN_STARS


@pytest.mark.parametrize("x0", [0.18, 0.47, 0.9])
def test_the_sweep_finds_focus(x0):
    sw = run(x0)
    # a tenth of a coarse step: 0.11 px of blur sigma here
    assert sw.best == pytest.approx(x0, abs=0.0125)
    assert sw.hfr is not None and sw.hfr < 1.5
    # the lens ends there, come from the near side like every stop
    last = [s["x"] for s in sw.stops if s["stage"] == "fine"]
    assert last == sorted(last, reverse=True)


def test_focus_beyond_the_far_end_stops_there():
    # infinity past the lens's travel: the far end is the best there is
    sw = run(-0.08)
    assert sw.best == pytest.approx(0.0, abs=1e-6)
    assert "far end" in sw.summary()


def test_the_v_curve():
    x = np.linspace(0, 1.5, 13)
    x0, h0 = focus.v_curve(x, np.hypot(1.0, 9 * (x - 0.4)))
    assert x0 == pytest.approx(0.4, abs=1e-6) and h0 == pytest.approx(1.0)
    assert focus.v_curve(x, 5 - x) is None            # no V in the range


def test_cloud_says_so():
    sw = focus.Sweep(0.0, 1.5)
    blank = np.full((480, 640), 14, np.uint8)
    while sw.next() is not None:
        sw.add(blank)
    assert sw.best is None and "no stars" in sw.summary()


def test_drift():
    assert focus.drifted(2.0, 1.2) and not focus.drifted(1.4, 1.2)
    assert not focus.drifted(None, 1.2)
