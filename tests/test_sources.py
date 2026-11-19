"""The workbench's orbit sources: order, on/off, what the solve is told."""

from datetime import datetime, timezone

import pytest

from skyfind import sources, tles
from skyfind.sources import SPACETRACK, Source

from .synth import constellation


@pytest.fixture
def datadir(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYFIND_DATA", str(tmp_path))
    return tmp_path


def _write(path, els):
    path.write_text("".join(f"{e.name}\n{e.line1}\n{e.line2}\n" for e in els))
    return path


def test_list_round_trips_and_starts_with_space_track_off(datadir):
    got, user = sources.load()
    assert [(s.path, s.enabled) for s in got] == [(SPACETRACK, False)]
    lst = [Source(SPACETRACK, True), Source("a.3le"), Source("b.tle", False)]
    sources.save(lst, "someone")
    got, user = sources.load()
    assert got == lst and user == "someone"
    assert "password" not in (datadir / "sources.json").read_text().lower()


def test_solve_args_keep_the_order_and_skip_what_is_off():
    lst = [Source(SPACETRACK, True), Source("cat.3le"), Source("x.tle", False),
           Source("sup.3le")]
    assert sources.solve_args(lst) == ["--tle", "cat.3le", "--tle", "sup.3le",
                                       "--spacetrack"]
    lst[0].enabled = False
    assert "--spacetrack" not in sources.solve_args(lst)
    # nothing but Space-Track: the solve's default already is Space-Track
    assert sources.solve_args([Source(SPACETRACK, True)]) == []


def _jd(t):
    return t.timestamp() / 86400 + 2440587.5


def test_a_later_file_wins_where_it_is_current(tmp_path):
    epoch = datetime(2025, 2, 14, tzinfo=timezone.utc)
    els, _ = constellation(epoch)
    a = _write(tmp_path / "a.3le", els[:6])
    b = _write(tmp_path / "b.3le", [tles.Element("BETTER", e.line1, e.line2)
                                    for e in els[3:8]])
    got = tles.nearest_epoch(tles.read_files([a, b]), _jd(epoch))
    assert len(got) == 8
    assert {e.name for e in got if e.norad == els[4].norad} == {"BETTER"}
    assert {e.name for e in got if e.norad == els[0].norad} == {els[0].name}


def test_a_later_file_never_displaces_current_elements_with_stale_ones(
        tmp_path):
    """July's supplemental Starlink orbits, ticked for a March clip: March's
    catalogue elements still win."""
    from .synth import make_tle
    march = datetime(2026, 3, 10, tzinfo=timezone.utc)
    july = datetime(2026, 7, 21, tzinfo=timezone.utc)
    now = make_tle(44714, "SL MARCH", march, 53.0, 10, 0.0001, 0, 0, 15.06)
    old = make_tle(44714, "SL JULY", july, 53.0, 10, 0.0001, 0, 0, 15.06)
    cat = _write(tmp_path / "cat.3le", [now])
    sup = _write(tmp_path / "sup.3le", [old])
    (got,) = tles.nearest_epoch(tles.read_files([cat, sup]), _jd(march))
    assert got.name == "SL MARCH"
    (got,) = tles.nearest_epoch(tles.read_files([cat, sup]), _jd(july))
    assert got.name == "SL JULY"


def test_summary_says_what_a_file_holds(tmp_path):
    epoch = datetime(2025, 2, 14, tzinfo=timezone.utc)
    els, _ = constellation(epoch)
    s = sources.summarize(_write(tmp_path / "c.3le", els))
    assert s.objects == len({e.norad for e in els}) and s.sets == len(els)
    assert s.first.date() == epoch.date()
    assert s.covers(datetime(2025, 2, 15, tzinfo=timezone.utc))
    assert not s.covers(datetime(2025, 3, 15, tzinfo=timezone.utc))
    assert "objects" in s.text()


def test_the_password_only_travels_in_the_environment():
    assert sources.environment("u", None) == {}
    assert sources.environment("u", "p") == {"SPACETRACK_USER": "u",
                                             "SPACETRACK_PASSWORD": "p"}
