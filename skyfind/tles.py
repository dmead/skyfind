"""Orbital elements for the night of a clip: find them with as little
friction as possible, and say exactly what's needed when we can't.

Order of preference:
  1. files the user passes (--tle, repeatable): 2- or 3-line TLE text,
     any source, later files preferred for their objects where current;
  2. a cached download for the same UTC range in the data directory;
  3. Space-Track (historical elements for any date), when the user has set
     SPACETRACK_USER and SPACETRACK_PASSWORD — a free account from
     https://www.space-track.org/auth/createAccount. Never anything else:
     no shared or made-up credentials;
  4. otherwise stop with TLEsNeeded, whose message names the account, the
     environment variables, and the exact query URL for a manual download.

Space-Track is the only public source of *historical* elements; CelesTrak
serves current ones only, so it can't cover a clip shot more than a day or
two ago and is not used here.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import os
import re
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import data

ST_BASE = "https://www.space-track.org"
ST_SIGNUP = "https://www.space-track.org/auth/createAccount"


class TLEsNeeded(RuntimeError):
    """No elements available without the user's help; the message says how.
    `reason`: "login" (no Space-Track login given), "refused" (Space-Track
    refused it; `detail` says why) or "download" (signed in, no elements);
    `url`: the query that fetches them. A front end can say it in its own
    words (the phone app's login fields, not environment variables)."""

    def __init__(self, msg: str, *, reason: str | None = None,
                 url: str | None = None, detail: str | None = None):
        super().__init__(msg)
        self.reason = reason
        self.url = url
        self.detail = detail


@dataclass(frozen=True)
class Element:
    name: str
    line1: str
    line2: str
    rank: int = 0             # its source's priority (`nearest_epoch`)

    @property
    def norad(self) -> int:
        return catalog_number(self.line1[2:7])

    @property
    def epoch_jd(self) -> float:
        yy = int(self.line1[18:20])
        day = float(self.line1[20:32])
        year = 2000 + yy if yy < 57 else 1900 + yy
        return (datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() / 86400
                + 2440587.5 + day - 1)


_ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"          # no I or O: A = 10 ... Z = 33


def catalog_number(field: str) -> int:
    """NORAD number from a TLE's 5-character field, Alpha-5 included:
    past 99999 the first digit becomes a letter (A0000 = 100000, T0000 =
    270000), skipping I and O."""
    f = field.strip()
    if f and f[0].isalpha():
        return (10 + _ALPHA5.index(f[0].upper())) * 10000 + int(f[1:])
    return int(f)


def parse(text: str) -> list[Element]:
    """2-line or 3-line (name + 2 lines) element sets; '0 NAME' accepted."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out, name, i = [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("1 ") and i + 1 < len(lines) \
                and lines[i + 1].startswith("2 "):
            out.append(Element(name or ln[2:7].strip(), ln, lines[i + 1]))
            name, i = "", i + 2
        else:
            name = ln[2:].strip() if ln.startswith("0 ") else ln.strip()
            i += 1
    return out


FRESH_DAYS = 3.0          # elements this close to the night count as current
MAX_AGE_DAYS = 30.0       # further than this from the night they're noise


def nearest_epoch(elements: list[Element], when_jd: float, *,
                  fresh_days: float = FRESH_DAYS,
                  max_age_days: float | None = MAX_AGE_DAYS) -> list[Element]:
    """One element set per object for the time `when_jd`: among its sets
    within `fresh_days`, the highest-ranked source's (then the nearest
    epoch); with none that fresh, the nearest epoch whatever the source.
    So a better source wins only where it is current — July's Starlink
    orbits never displace March's for a March clip. An object whose
    nearest set is over `max_age_days` away is left out: propagated that
    far its position is noise (and a satellite launched after the night
    wasn't there at all), and noise fits tracks by chance."""
    if isinstance(elements, Catalog):
        return elements.nearest(when_jd, fresh_days, max_age_days)
    best: dict[int, tuple] = {}
    for e in elements:
        d = abs(e.epoch_jd - when_jd)
        if max_age_days is not None and d > max_age_days:
            continue
        key = (0, -e.rank, d) if d <= fresh_days else (1, 0, d)
        b = best.get(e.norad)
        if b is None or key < b[0]:
            best[e.norad] = (key, e)
    return [best[k][1] for k in sorted(best)]


def query_url(start: datetime, end: datetime) -> str:
    """Space-Track gp_history query covering [start, end] (UTC dates)."""
    a, b = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    return (f"{ST_BASE}/basicspacedata/query/class/gp_history/EPOCH/{a}--{b}"
            f"/orderby/NORAD_CAT_ID/format/3le")


def _range(first_utc: datetime, last_utc: datetime) -> tuple[datetime, datetime]:
    # elements are published about daily; a day either side covers the night
    return (first_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0), \
        (last_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0)


def cache_path(start: datetime, end: datetime) -> Path:
    return data.data_dir() / \
        f"tle_{start:%Y%m%d}-{end:%Y%m%d}.3le"


def sign_in(user: str, password: str):
    """A session signed in to Space-Track (an opener holding its cookie),
    or TLEsNeeded(reason="refused") with Space-Track's reason; no network
    raises urllib's URLError."""
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    body = urllib.parse.urlencode({"identity": user,
                                   "password": password}).encode()
    # a refused sign-in comes back as HTTP 401 with the reason in the body
    # ({"Login": "Failed"}, {"Login": "Password does not meet ..."})
    try:
        with op.open(f"{ST_BASE}/ajaxauth/login", body, timeout=60) as r:
            reply = r.read()
    except urllib.error.HTTPError as e:
        why = _reason(e.read())
        raise TLEsNeeded(f"Space-Track refused the sign-in for {user!r} "
                         f"(HTTP {e.code}): {why}. Check the "
                         "user name and password (the workbench's Sources…, "
                         "or SPACETRACK_USER / SPACETRACK_PASSWORD) by "
                         f"signing in at {ST_BASE}.",
                         reason="refused", detail=why) from None
    if b"Failed" in reply:
        why = _reason(reply)
        raise TLEsNeeded(f"Space-Track refused the sign-in for {user!r}: "
                         f"{why}.", reason="refused", detail=why)
    return op


def _space_track(user: str, password: str, url: str) -> str:
    op = sign_in(user, password)
    try:
        with op.open(url, timeout=600) as r:
            return r.read().decode("ascii", errors="ignore")
    except urllib.error.HTTPError as e:
        raise TLEsNeeded(f"Space-Track signed in but refused the download "
                         f"(HTTP {e.code}: {_reason(e.read())}) for {url}",
                         reason="download", url=url) from None


def _reason(reply: bytes) -> str:
    import json
    try:
        d = json.loads(reply)
        return str(d.get("Login") or d.get("error") or d) if \
            isinstance(d, dict) else str(d)
    except ValueError:
        return reply.decode("utf-8", errors="replace").strip()[:200] or \
            "no reason given"


def read_files(paths, first_rank: int = 1) -> "Catalog":
    """Element sets from several files, each ranked by its place in the
    list: later files are the better sources for their objects (operators'
    supplemental orbits for Starlink or OneWeb, amateur-tracked classified
    objects) where their elements are current (`nearest_epoch`)."""
    out = Catalog.empty()
    for i, p in enumerate(paths):
        out = out + Catalog.read(p, first_rank + i)
    return out


def n_objects(elements) -> int:
    """How many distinct objects the element sets cover."""
    if isinstance(elements, Catalog):
        return elements.n_objects()
    return len({e.norad for e in elements})


SPAN_DAYS = 7             # a download's length when fetching weeks


def _download(start: datetime, end: datetime) -> str:
    """Space-Track's history from `start` to `end` with the user's account
    (SPACETRACK_USER / SPACETRACK_PASSWORD), or TLEsNeeded saying how."""
    url = query_url(start, end)
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not (user and password):
        raise TLEsNeeded(
            "Identifying satellites needs their orbital elements for "
            f"{start:%Y-%m-%d} to {end:%Y-%m-%d} (UTC), and only Space-Track "
            "keeps historical ones. Either:\n"
            f"  - make a free account at {ST_SIGNUP}, then set "
            "SPACETRACK_USER and SPACETRACK_PASSWORD and re-run "
            "(the download is then automatic and cached), or\n"
            f"  - log in on the website, open\n    {url}\n"
            "    save the result, and pass it with --tle FILE.",
            reason="login", url=url)
    text = _space_track(user, password, url)
    if not parse(text):
        raise TLEsNeeded(f"Space-Track returned no elements for {url}",
                         reason="download", url=url)
    return text


def acquire_span(first_utc: datetime, last_utc: datetime, *, log=print
                 ) -> list[tuple[datetime, datetime, Path]]:
    """Space-Track's elements over weeks, fetched `SPAN_DAYS` at a time
    (a month in one query is ~400 MB) and cached: [(start, end, file)]."""
    start, end = _range(first_utc, last_utc)
    out = []
    a = start
    while a < end:
        b = min(a + timedelta(days=SPAN_DAYS), end)
        p = cache_path(a, b)
        if not p.exists():
            if log:
                log(f"  Space-Track: elements for {a:%Y-%m-%d} to "
                    f"{b:%Y-%m-%d}...")
            p.write_text(_download(a, b), encoding="ascii")
        out.append((a, b, p))
        a = b
    return out


def acquire(first_utc: datetime, last_utc: datetime, *,
            path=None, spacetrack: bool = False) -> "Catalog":
    """Element sets for a clip spanning first_utc..last_utc (aware UTC).
    `path`: a file, or a list of files (`read_files`); with `spacetrack`
    too, the files go over Space-Track's elements for the night."""
    if path:
        files = read_files(path if isinstance(path, (list, tuple))
                           else [path])
        return acquire(first_utc, last_utc) + files if spacetrack else files
    start, end = _range(first_utc, last_utc)
    cached = cache_path(start, end)
    if not cached.exists():
        cached.write_text(_download(start, end), encoding="ascii")
    return Catalog.read(cached, 0)


# ---------------------------------------------------------------- columns
INDEX_VERSION = 1


class Catalog:
    """Element sets as columns — object number, epoch, source rank, and
    where each set's lines are — so a month of history (millions of sets)
    loads, and has one set per object picked, in a fraction of a second.
    `Element`s are made only for the sets picked (`nearest_epoch`). Each
    file's columns are indexed once and kept in the data directory. Adds,
    counts and iterates like the list of Elements it stands for."""

    def __init__(self, norad, epoch, rank, src, off, sources):
        self.norad = np.asarray(norad, np.int64)
        self.epoch = np.asarray(epoch, float)
        self.rank = np.asarray(rank, np.int32)
        self.src = np.asarray(src, np.int32)
        # (n, 5): name start (-1: none), name length, line 1 start, line 2
        # start, line lengths packed as l1 * 65536 + l2
        self.off = np.asarray(off, np.int64).reshape(-1, 5)
        self.sources = list(sources)     # file paths, or bytes in memory
        self._data: dict = {}

    # ---------------------------------------------------------- build
    @classmethod
    def empty(cls) -> "Catalog":
        return cls([], [], [], [], np.zeros((0, 5)), [])

    @classmethod
    def read(cls, path, rank: int = 0) -> "Catalog":
        """A file's element sets (ranked `rank`), from its index when the
        file hasn't changed since it was made."""
        path = Path(path).resolve()
        st = path.stat()
        key = hashlib.sha1(str(path).lower().encode()).hexdigest()[:16]
        idx = data.data_dir() / "element-index" / f"{key}.npz"
        cols = None
        if idx.exists():
            try:
                z = np.load(idx)
                if int(z["version"]) == INDEX_VERSION and \
                        int(z["size"]) == st.st_size and \
                        int(z["mtime_ns"]) == st.st_mtime_ns:
                    cols = {k: z[k] for k in ("norad", "epoch", "off")}
            except (OSError, ValueError, KeyError):
                cols = None
        if cols is None:
            cols = _index(path.read_bytes())
            try:
                idx.parent.mkdir(parents=True, exist_ok=True)
                tmp = idx.with_suffix(".part.npz")
                np.savez(tmp, version=INDEX_VERSION, size=st.st_size,
                         mtime_ns=st.st_mtime_ns, **cols)
                tmp.replace(idx)
            except OSError:
                pass                      # no room: index it again next time
        n = len(cols["norad"])
        return cls(cols["norad"], cols["epoch"], np.full(n, rank),
                   np.zeros(n), cols["off"], [path])

    @classmethod
    def of(cls, elements) -> "Catalog":
        """A Catalog of Elements (kept in memory)."""
        if isinstance(elements, Catalog):
            return elements
        elements = list(elements)
        text = "".join(f"0 {e.name}\n{e.line1}\n{e.line2}\n"
                       for e in elements).encode("ascii", errors="ignore")
        cols = _index(text)
        return cls(cols["norad"], cols["epoch"], [e.rank for e in elements],
                   np.zeros(len(elements)), cols["off"], [text])

    def __add__(self, other) -> "Catalog":
        other = Catalog.of(other)
        return Catalog(np.concatenate([self.norad, other.norad]),
                       np.concatenate([self.epoch, other.epoch]),
                       np.concatenate([self.rank, other.rank]),
                       np.concatenate([self.src,
                                       other.src + len(self.sources)]),
                       np.concatenate([self.off, other.off]),
                       self.sources + other.sources)

    def __radd__(self, other) -> "Catalog":
        return Catalog.of(other) + self

    # ---------------------------------------------------------- use
    def __len__(self) -> int:
        return len(self.norad)

    def __iter__(self):
        for i in range(len(self)):
            yield self.element(i)

    def n_objects(self) -> int:
        return len(np.unique(self.norad))

    def _bytes(self, s: int):
        if s not in self._data:
            src = self.sources[s]
            self._data[s] = src if isinstance(src, bytes) else \
                Path(src).read_bytes()
        return self._data[s]

    def element(self, i: int) -> Element:
        b = self._bytes(int(self.src[i]))
        n0, nl, a1, a2, ll = (int(x) for x in self.off[i])
        l1 = b[a1:a1 + ll // 65536].decode("ascii", errors="ignore")
        l2 = b[a2:a2 + ll % 65536].decode("ascii", errors="ignore")
        name = b[n0:n0 + nl].decode("ascii", errors="ignore") if n0 >= 0 \
            else l1[2:7].strip()
        return Element(name, l1, l2, int(self.rank[i]))

    def nearest(self, when_jd: float, fresh_days: float,
                max_age_days: float | None) -> list[Element]:
        """`nearest_epoch`, in columns: the same choice (ties to the first
        listed), sorted by object."""
        if not len(self):
            return []
        d = np.abs(self.epoch - when_jd)
        keep = np.ones(len(d), bool) if max_age_days is None else \
            d <= max_age_days
        i = np.flatnonzero(keep)
        stale = d[i] > fresh_days
        rk = np.where(stale, 0, -self.rank[i])
        o = np.lexsort((i, d[i], rk, stale, self.norad[i]))
        nor = self.norad[i][o]
        first = np.r_[True, nor[1:] != nor[:-1]]
        return [self.element(int(k)) for k in i[o][first]]


def _index(b: bytes) -> dict:
    """`parse`'s element sets in a text, as columns (see `Catalog`)."""
    lines = []                                   # (start, stripped line)
    for m in re.finditer(rb"[^\r\n]+", b):
        ln = m.group().rstrip()
        if ln.strip():
            lines.append((m.start(), ln))
    off, l1s = [], []
    name = (-1, 0)
    i = 0
    while i < len(lines):
        o, ln = lines[i]
        if ln.startswith(b"1 ") and i + 1 < len(lines) \
                and lines[i + 1][1].startswith(b"2 "):
            o2, ln2 = lines[i + 1]
            off.append((name[0], name[1], o, o2,
                        len(ln) * 65536 + len(ln2)))
            l1s.append(ln)
            name, i = (-1, 0), i + 2
        else:
            body, at = (ln[2:], o + 2) if ln.startswith(b"0 ") else (ln, o)
            lead = len(body) - len(body.lstrip())
            text = body.strip()
            name = (at + lead, len(text)) if text else (-1, 0)
            i += 1
    n = len(l1s)
    norad = np.zeros(n, np.int64)
    epoch = np.zeros(n)
    if n:
        a = np.array([ln[:32].ljust(32) for ln in l1s], "S32")
        a = a.view("S1").reshape(n, 32)
        field = np.char.strip(a[:, 2:7].copy().view("S5").ravel())
        digits = np.char.isdigit(field)
        norad[digits] = field[digits].astype(np.int64)
        for k in np.flatnonzero(~digits):              # Alpha-5, or odd
            norad[k] = catalog_number(field[k].decode("ascii", "ignore"))
        yy = a[:, 18:20].copy().view("S2").ravel().astype(np.int64)
        day = a[:, 20:32].copy().view("S12").ravel().astype(float)
        year = np.where(yy < 57, 2000 + yy, 1900 + yy)
        jan1 = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]")
        epoch = jan1.astype(np.int64).astype(float) + 2440587.5 + day - 1
    return dict(norad=norad, epoch=epoch,
                off=np.array(off, np.int64).reshape(-1, 5))
