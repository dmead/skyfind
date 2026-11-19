from datetime import datetime, timezone

import pytest

from skyfind import tles

ISS = """0 ISS (ZARYA)
1 25544U 98067A   24276.46436926  .00016717  00000-0  30271-3 0  9993
2 25544  51.6394 196.2476 0007143  47.7536  72.6436 15.49859307474999
1 25544U 98067A   24277.10000000  .00016717  00000-0  30271-3 0  9994
2 25544  51.6394 193.0000 0007143  47.7536  72.6436 15.49859307475090
"""


def test_parse_mixed_2line_3line_and_nearest_epoch():
    els = tles.parse(ISS)
    assert [e.norad for e in els] == [25544, 25544]
    assert els[0].name == "ISS (ZARYA)"
    # 2024 day 277.1 is later; pick the set nearest 2024-10-03 00:00 UTC
    when = datetime(2024, 10, 3, tzinfo=timezone.utc).timestamp() / 86400 + 2440587.5
    (best,) = tles.nearest_epoch(els, when)
    assert best.line1.endswith("9994")


def test_without_credentials_the_message_says_exactly_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYFIND_DATA", str(tmp_path))
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    a = datetime(2024, 10, 2, 6, 4, tzinfo=timezone.utc)
    b = datetime(2024, 10, 2, 13, 14, tzinfo=timezone.utc)
    with pytest.raises(tles.TLEsNeeded) as e:
        tles.acquire(a, b)
    msg = str(e.value)
    assert "SPACETRACK_USER" in msg and "--tle" in msg
    assert "EPOCH/2024-10-01--2024-10-03" in msg
    assert tles.ST_SIGNUP in msg


def test_cached_download_is_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYFIND_DATA", str(tmp_path))
    a = datetime(2024, 10, 2, 6, tzinfo=timezone.utc)
    start, end = tles._range(a, a)
    tles.cache_path(start, end).write_text(ISS, encoding="ascii")
    assert len(tles.acquire(a, a)) == 2


def test_alpha5_catalogue_numbers():
    assert tles.catalog_number("25544") == 25544
    assert tles.catalog_number("    5") == 5
    assert tles.catalog_number("A0000") == 100000
    assert tles.catalog_number("T0000") == 270000
    assert tles.catalog_number("Z9999") == 339999
    from sgp4.api import Satrec
    line1 = "1 T0000U 24001A   24276.46436926  .00016717  00000-0  30271-3 0  9993"
    line2 = "2 T0000  51.6394 196.2476 0007143  47.7536  72.6436 15.49859307474999"
    e = tles.Element("X", line1, line2)
    assert e.norad == Satrec.twoline2rv(line1, line2).satnum == 270000


def test_a_refused_sign_in_says_why(monkeypatch):
    """Space-Track answers a bad password with HTTP 401 and the reason in
    the body; the user sees the reason, not a traceback. (No network: the
    server is a stand-in.)"""
    import io
    import urllib.error
    import urllib.request

    class Opener:
        def open(self, url, data=None, timeout=None):
            raise urllib.error.HTTPError(
                url, 401, "Unauthorized", {},
                io.BytesIO(b'{"Login":"Failed"}'))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(tles.TLEsNeeded) as e:
        tles._space_track("someone", "wrong", "https://example.invalid/q")
    msg = str(e.value)
    assert "refused the sign-in" in msg and "Failed" in msg
    assert "wrong" not in msg                     # never echo the password


def test_elements_far_from_the_night_are_left_out():
    """A set a month or more from the night is noise there (or a satellite
    not yet launched): leave the object out rather than guess."""
    (iss,) = tles.parse(ISS)[:1]
    when = iss.epoch_jd
    assert tles.nearest_epoch([iss], when + 10)
    assert not tles.nearest_epoch([iss], when + 45)
    assert tles.nearest_epoch([iss], when + 45, max_age_days=None)
