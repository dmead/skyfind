"""Where the orbital elements come from — the workbench's list of sources.

An ordered list: Space-Track's history for the clip's night (downloaded once
with the user's own account, then cached), then element files, each on or
off. Order matters: an object in a later source replaces its sets from
earlier ones (`tles.read_files`), so the general catalogue goes first and
the better sources for some objects after it — operators' supplemental
orbits for Starlink or OneWeb, amateur-tracked classified objects.

The list lives in the data directory (`sources.json`). The Space-Track
password never does: it goes to the system keyring (Windows Credential
Manager) when the `keyring` package is there, and to the solve only through
its environment (`tles` reads SPACETRACK_USER / SPACETRACK_PASSWORD).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import data, tles

SPACETRACK = "space-track"             # the list's entry for the download
KEYRING_SERVICE = "skyfind: space-track.org"


@dataclass
class Source:
    path: str                          # a file, or SPACETRACK
    enabled: bool = True

    @property
    def is_spacetrack(self) -> bool:
        return self.path == SPACETRACK


def _file() -> Path:
    return data.data_dir() / "sources.json"


def load() -> tuple[list[Source], str]:
    """(sources, Space-Track user name) as last saved; by default just the
    Space-Track entry, off."""
    try:
        d = json.loads(_file().read_text())
        out = [Source(**s) for s in d.get("sources", [])]
        user = d.get("spacetrack_user", "")
    except (OSError, ValueError, TypeError):
        out, user = [], ""
    if not any(s.is_spacetrack for s in out):
        out.insert(0, Source(SPACETRACK, False))
    return out, user


def save(sources: list[Source], user: str = "") -> None:
    _file().write_text(json.dumps(dict(
        sources=[asdict(s) for s in sources], spacetrack_user=user),
        indent=1))


def solve_args(sources: list[Source]) -> list[str]:
    """The `skyfind solve` options for the enabled sources, in order. The
    Space-Track entry counts only first in the list (it is the general
    catalogue the files go over); no files and no Space-Track = the
    default (cached, or Space-Track)."""
    on = [s for s in sources if s.enabled]
    files = [s.path for s in on if not s.is_spacetrack]
    args = [a for f in files for a in ("--tle", f)]
    if files and any(s.is_spacetrack for s in on):
        args.append("--spacetrack")
    return args


# ---------------------------------------------------------------- files
@dataclass
class Summary:
    objects: int
    sets: int
    first: datetime | None             # epochs, UTC
    last: datetime | None

    def text(self) -> str:
        if not self.sets:
            return "no element sets"
        span = f"{self.first:%Y-%m-%d}" if self.first.date() == \
            self.last.date() else f"{self.first:%Y-%m-%d} to {self.last:%Y-%m-%d}"
        return f"{self.objects:,} objects, {self.sets:,} sets, epochs {span}"

    def covers(self, when: datetime, days: float = 3.0) -> bool:
        """Epochs close enough to `when` for SGP4 to be useful (LEO elements
        go stale in days)."""
        if not self.sets:
            return False
        lo = self.first.timestamp() - days * 86400
        hi = self.last.timestamp() + days * 86400
        return lo <= when.timestamp() <= hi


def summarize(path) -> Summary:
    els = tles.parse(Path(path).read_text(encoding="ascii", errors="ignore"))
    if not els:
        return Summary(0, 0, None, None)
    jds = [e.epoch_jd for e in els]

    def dt(jd):
        return datetime.fromtimestamp((jd - 2440587.5) * 86400, timezone.utc)
    return Summary(len({e.norad for e in els}), len(els), dt(min(jds)),
                   dt(max(jds)))


def cached_summary(path) -> Summary:
    """`summarize`, remembered by the file's size and time (a night's
    history is ~50 MB and takes seconds to read)."""
    p = Path(path)
    st = p.stat()
    key = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    f = data.data_dir() / "source_summaries.json"
    try:
        cache = json.loads(f.read_text())
    except (OSError, ValueError):
        cache = {}
    c = cache.get(key)
    if c is not None:
        return Summary(c["objects"], c["sets"],
                       datetime.fromisoformat(c["first"]) if c["first"] else None,
                       datetime.fromisoformat(c["last"]) if c["last"] else None)
    s = summarize(p)
    cache[key] = dict(objects=s.objects, sets=s.sets,
                      first=s.first.isoformat() if s.first else None,
                      last=s.last.isoformat() if s.last else None)
    try:
        f.write_text(json.dumps(cache))
    except OSError:
        pass
    return s


# ---------------------------------------------------------------- account
def _keyring():
    try:
        import keyring
        return keyring
    except ImportError:
        return None


def can_remember() -> bool:
    return _keyring() is not None


def password(user: str) -> str | None:
    kr = _keyring()
    if kr is None or not user:
        return None
    try:
        return kr.get_password(KEYRING_SERVICE, user)
    except Exception:                          # a locked or missing backend
        return None


def remember(user: str, pw: str) -> bool:
    """Keep the password in the system keyring; False if there is none."""
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(KEYRING_SERVICE, user, pw)
        return True
    except Exception:
        return False


def forget(user: str) -> None:
    kr = _keyring()
    if kr is not None and user:
        try:
            kr.delete_password(KEYRING_SERVICE, user)
        except Exception:
            pass


def environment(user: str, pw: str | None) -> dict[str, str]:
    """What the solve's process needs to reach Space-Track."""
    if not (user and pw):
        return {}
    return {"SPACETRACK_USER": user, "SPACETRACK_PASSWORD": pw}


def signup_url() -> str:
    return tles.ST_SIGNUP


def has_env_account() -> bool:
    return bool(os.environ.get("SPACETRACK_USER")
                and os.environ.get("SPACETRACK_PASSWORD"))
