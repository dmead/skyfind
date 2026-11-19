"""Parallel work: processes on the desktop, threads where a program can't
start more of itself (the Android app sets SKYFIND_PROCESSES=0). Either
way the answers are the same (Parallel = serial); threads overlap only
where numpy and OpenCV let go of the GIL."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def processes() -> bool:
    return os.environ.get("SKYFIND_PROCESSES", "1") != "0"


def executor(n: int):
    """A pool of `n` workers: processes, or threads without them."""
    return (ProcessPoolExecutor if processes() else ThreadPoolExecutor)(
        max(1, n))
